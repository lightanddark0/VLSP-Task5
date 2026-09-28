# Qwen3-VL-8B-Instruct QLoRA for ViSPARTQA (YN/FR/FB/CO)

Text-only 4-bit QLoRA fine-tuning of `Qwen/Qwen3-VL-8B-Instruct` on the
Vietnamese SPARTQA task, targeting balanced quality across all four question
types (YN, FR, FB, CO) rather than one blended score. Full rationale, phased
schedule, GPU budget rule and non-regression gate are in the planning notes;
this document is the practical setup/run reference.

**Nothing in this document has been executed.** It is written to run on
[Modal](https://modal.com) against a persistent Volume named `qwen`, on one
A100-80GB GPU, within roughly a 6-12 hour total budget. Every version pin and
architectural assumption below must be verified once against the actual
container before committing the full budget — see "Unverified assumptions"
at the end.

## Repository layout added by this slice

```text
spartqa/
  qwen_data.py     Dependency-free split/prompt/answer contract (Phase 1, already tested offline)
  qwen_model.py     Quantization + LoRA setup, restricted to the language decoder (needs torch/peft/bitsandbytes)
  qwen_dataset.py   Tokenization, prompt/completion masking, collation, stage A/B sampling (needs torch)
  qwen_eval.py      Constrained-decoding generation, eight-cell metrics, regression gate (needs torch)
  qwen_train.py     Orchestrates the above into stage A/B training, evaluate, predict (needs torch)
qwen_finetune.py    Local CLI: manifest, audit, inspect-modules, train-stage-a/b, evaluate, predict
modal_app.py        Modal App/Image/Volume wiring; local_entrypoints dispatch the same CLI to a remote Modal Function
modal_runner.py     In-process alternative: calls the same CLI directly, for a notebook/shell already inside the GPU environment
requirements-qlora.txt  Training dependency versions (for a bare-metal/non-Modal run)
requirements-modal.txt  `modal` client only, for the machine issuing `modal run` commands
```

`qwen_finetune.py` defers all heavy imports into each subcommand, so
`python qwen_finetune.py --help` and `python -m compileall` work without
torch/transformers/peft/bitsandbytes installed, matching `gpt_experiment.py`'s
convention and keeping the existing offline CI unaffected.

## Two ways to run this

- **`modal_app.py`** — run from a separate machine that only has the `modal`
  client installed (`requirements-modal.txt`). Each step is dispatched with
  `modal run modal_app.py::<step>`, building a fresh container image (with
  `requirements-qlora.txt` baked in) and mounting the `qwen` Volume. Use this
  if you are not already inside a GPU container.
- **`modal_runner.py`** — call its functions directly (e.g.
  `import modal_runner; modal_runner.manifest()`) from a Python process/notebook
  that is *already* running inside the target environment (GPU attached, this
  repo's files present, `requirements-qlora.txt` already installed). It runs
  the exact same `qwen_finetune.py` commands in-process, with no `modal`
  client, image build or Volume mount involved. Paths default to `./Data` and
  `./outputs/qwen_qlora` (siblings of `qwen_finetune.py`); pass `root_dir=` to
  point elsewhere. Run `modal_runner.list_root()` first to see what is
  actually where before assuming a path.

The rest of this document uses `modal_app.py`'s command names; substitute the
same keyword arguments into the matching `modal_runner.py` function if you are
running in-process instead.

## One-time setup

On the machine that will issue `modal run` commands (not necessarily the GPU host):

```powershell
python -m pip install -r requirements-modal.txt
modal setup
```

Upload the organizer JSON files into the Volume once (either works):

```powershell
modal run modal_app.py::upload_data
# or, without any Python:
modal volume put qwen ./Data /Data
```

This expects `Data/human_train.json` and `Data/auto_train.json` locally, matching
[Data/README.md](../Data/README.md). Public test files can be uploaded the same way
and referenced later by `predict`.

## Command sequence

Run each step from the machine with the `modal` CLI; each one dispatches to a
Modal Function running against the `qwen` Volume, mounted at `/vol` in the
container. Every entrypoint accepts `--root-dir` (default `/vol`), which is
combined with the fixed subpaths `Data/` and `outputs/qwen_qlora/`. **If you
uploaded the repo/Data folder yourself to a different location on the
Volume, run `list_volume` first and pass the matching `--root-dir` to every
other command** — do not assume `/vol/Data` is correct.

```powershell
modal run modal_app.py::list_volume
modal run modal_app.py::manifest --root-dir /vol
modal run modal_app.py::inspect_modules
modal run modal_app.py::audit --root-dir /vol
modal run modal_app.py::evaluate --root-dir /vol --output-name metrics_base.json
modal run modal_app.py::train_stage_a --root-dir /vol --auto-subset-size 30000 --max-steps 800
modal run modal_app.py::evaluate --root-dir /vol --adapter /vol/outputs/qwen_qlora/stage_a/adapter --output-name metrics_stage_a.json --baseline-metrics /vol/outputs/qwen_qlora/metrics_base.json
modal run modal_app.py::train_stage_b --root-dir /vol --max-steps 300 --human-passes 3.0
modal run modal_app.py::evaluate --root-dir /vol --adapter /vol/outputs/qwen_qlora/stage_b/adapter --output-name metrics_stage_b.json --baseline-metrics /vol/outputs/qwen_qlora/metrics_base.json
modal run modal_app.py::predict --input-path /vol/Data/human_public_test.json --adapter /vol/outputs/qwen_qlora/stage_b/adapter --output-name human_public_test_predictions.json
modal run modal_app.py::predict --input-path /vol/Data/auto_public_test.json --adapter /vol/outputs/qwen_qlora/stage_b/adapter --output-name auto_public_test_predictions.json
```

0. **list_volume** — cheap, GPU-free listing of every file actually present
   under a given path on the Volume. Use this to confirm where your own
   upload landed before running anything else; a wrong `--root-dir` fails
   fast with `FileNotFoundError` rather than silently reading stale data.
1. **manifest** — story-disjoint 80/20 split per source (`human`/`auto`), with
   duplicate stories across sources kept in one split and coverage repair so
   every (source, task) cell has validation support. Writes
   `manifest.json`; reruns with the same seed reproduce it exactly, so this
   file — not the raw JSON — is the source of truth for train/val membership.
2. **inspect_modules** — loads only the quantized base model and prints the
   resolved LoRA target modules. Run this before any training and manually
   confirm the printed names are all inside the language decoder (no
   `visual`/`merger`/`projector` names). Adjust
   `NON_LANGUAGE_NAME_FRAGMENTS` in `spartqa/qwen_model.py` if the actual
   revision uses different naming.
3. **audit** — label distribution (counts, cardinalities, per-label
   frequencies including Human `DK`) and chat-templated token-length
   percentiles. Use the reported max/p99 to size `--max-steps` and confirm no
   record needs truncation.
4. **evaluate** (no `--adapter`) — quantized base-model baseline on the
   validation split, using the same constrained decoding as later checkpoints.
   This is the reference for the regression gate below.
5. **train_stage_a** — Auto-only foundation stage, LR `1e-4`, one bounded pass
   over a deterministic `--auto-subset-size` subset balanced across YN/FR/FB/CO.
6. **train_stage_b** — continues the *same* adapter, Human:Auto sampled 1:1,
   capped at `--human-passes` effective passes over Human TRAIN, LR `3e-5`.
7. **evaluate** after each stage, always against the fixed manifest validation
   split and always passing `--baseline-metrics` from the quantized-base run.
8. **predict** — fills `answer` for every question in an unlabeled public-test
   file using constrained decoding; fails loudly if any prediction is missing
   or invalid rather than silently leaving/guessing an answer.

## Evaluation and the non-regression gate

`evaluate` reports, per source (`human`, `auto`):

- Accuracy for YN and CO.
- Exact Match (primary) and Jaccard (secondary) for FR and FB.
- `_invalid_count` / `_total_count`: unparsable or schema-invalid generations
  count as wrong, never as a lucky empty-set match (see
  `spartqa/qwen_eval.py::_MISSING_PREDICTION_SENTINEL`).

`summary.official_task_source_average` is `(Human + Auto) / 2` per task/metric,
matching [Docs/task_description.md](task_description.md). There is no single
official all-task scalar; `summary.internal_macro_of_eight_cells` and
`summary.worst_cell` are internal selection aids only.

When `--baseline-metrics` is given, `regression_gate` fails (`passed: false`)
if any (source, task) primary cell drops more than 2 percentage points below
the quantized base model. Treat a failed gate as "goal not met for that
cell" — do not select a checkpoint based on macro score alone if the gate
fails.

## Resuming and budget

- Training checkpoints under `--output-dir/checkpoint-*` (HF `Trainer`
  default) support `--resume-from-checkpoint`.
- `train-stage-b --init-adapter` must point at a `stage_a/adapter` (or a
  previous `stage_b/adapter` for a bounded continuation run) — the same
  adapter continues training, it is not reinitialized.
- Reserve roughly 20-30% of the total GPU budget for the two `evaluate` runs
  and `predict`, ~10% for `inspect_modules`/`audit`/a short smoke run, and the
  remainder split across stage A (larger share) and stage B (capped by
  `--human-passes`, not wall-clock).
- `--max-steps` is required and must be picked from measured
  tokens/sec after a short pilot (e.g. `--max-steps 20`) on the target GPU —
  this repository does not claim a throughput number since it has not run.

## Unverified assumptions (check before the full run)

These follow directly from `Qwen/Qwen3-VL-8B-Instruct`'s model card and
Modal's documentation as of this writing, but must be confirmed on the actual
container before relying on them:

- **Verified 2026-09-27 (live run):** `AutoProcessor.from_pretrained` for
  Qwen3-VL raises `ImportError: AutoVideoProcessor requires the Torchvision
  library` even for pure text usage, because the processor also builds a
  video-processor backend. Fix: `torchvision` is now in
  `requirements-qlora.txt` and `modal_app.py`'s `pip_install`; if you built an
  image/venv before this fix, add `torchvision` and retry `audit` or
  `inspect-modules` first (cheap, no GPU needed) before spending stage-A time.
- **Verified 2026-09-27 (live run, second incident):** adding `torchvision`
  unpinned let pip re-resolve `torch`'s own CUDA/NCCL sub-wheels to a
  mismatched set, breaking even `import torch` with
  `ImportError: .../libtorch_cuda.so: undefined symbol: ncclCommResume`.
  Fix: `torch`, `torchvision` and `nvidia-nccl-cu12` are now pinned to an
  exact, previously-verified-compatible triple (`2.8.0` / `0.23.0` /
  `2.27.3`) in both `requirements-qlora.txt` and `modal_app.py`, instead of
  open ranges. If this exact triple ever fails again on a rebuilt image (e.g.
  the mirror no longer serves these exact wheels), the standard fallback is
  to install `torch`/`torchvision` from PyTorch's own CUDA wheel index
  (`--extra-index-url https://download.pytorch.org/whl/cu128`, matching the
  CUDA 12.8 build already seen in the install log) instead of the plain PyPI
  mirror, which guarantees an ABI-matched set. Always smoke-test with
  `inspect-modules` (cheap: loads only the model, no training) before running
  a full stage, and treat any `import torch` failure as blocking, not
  something to retry through.
- `transformers>=4.57` includes `Qwen3VLForConditionalGeneration`; if
  `load_base_model` raises `ImportError`, install
  `pip install git+https://github.com/huggingface/transformers` instead.
- **Verified 2026-09-27 (live run, third incident):** Qwen3-VL's `AutoProcessor`
  is multimodal, so its `apply_chat_template` requires each message's
  `content` to be a list of parts (`[{"type": "text", "text": ...}]`), not a
  bare string — a bare string is iterated character-by-character, crashing
  with `TypeError: string indices must be integers, not 'str'` while it scans
  for image/video parts. Fixed in `spartqa/qwen_data.py::build_prompt_messages`
  (system/user messages) and `spartqa/qwen_dataset.py::encode_example`
  (assistant completion message). No dummy image or `pixel_values` is needed —
  only the content-parts structure — but this was not provable without a live
  call; re-check the first few `input_ids` from `audit` decode back to plain
  text with no stray image/video placeholder tokens.
- The resolved LoRA target module names from `inspect-modules` are indeed
  vision-free (see step 2 above).
- **Verified 2026-09-27 (live run, fourth incident):** batched constrained
  generation in `evaluate`/`predict` crashed with `ValueError:
  prefix_allowed_tokens_fn returned an empty list for batch ID 1` after a
  `right-padding was detected!` warning. With right padding, `model.generate`
  appends each new token at the same padded length for every row, which for a
  shorter sequence in the batch is a pad-token position, not its real last
  token — the model then continues from garbage, so no legal completion in
  `LegalAnswerTrie` ever matches. Fixed by setting
  `processor.tokenizer.padding_side = "left"` in
  `spartqa/qwen_model.py::load_processor`. This only affects batched
  generation tokenization; training is unaffected because
  `spartqa/qwen_dataset.py::collate_batch` builds its own right-padded
  tensors manually and never reads `padding_side`.
- **Verified 2026-09-27 (live run, fifth incident):** the identical
  `prefix_allowed_tokens_fn returned an empty list` error persisted even
  after the padding fix above — a second, unrelated bug. HF's batched
  `generate()` keeps calling `prefix_allowed_tokens_fn` for every row at
  every step until the WHOLE batch is done, including rows that already
  emitted their own EOS for a short answer (e.g. YN) while another row in the
  same batch (e.g. FR) is still generating. `LegalAnswerTrie.allowed_tokens`
  used to filter out the `None` terminal marker and return `[]` once a row
  had already reached a complete answer, which is a hard crash, not a no-op.
  Fixed in `spartqa/qwen_eval.py::LegalAnswerTrie.allowed_tokens`: once the
  walk reaches a terminal leaf (or an out-of-trie token, defensively), return
  `[eos_token_id]` instead of `[]`, letting HF pad that row while the rest of
  the batch keeps decoding. `skip_special_tokens=True` in the decode step
  already strips the extra trailing EOS, so parsed answers are unaffected.
- `optim="paged_adamw_8bit"` is available from the installed `bitsandbytes`
  build inside the Modal image.
- Effective batch size (`micro_batch_size x gradient_accumulation_steps`, `4x8`
  by default) actually fits 80 GB with the longest audited record; reduce
  `--micro-batch-size` and raise `--gradient-accumulation-steps` if not,
  without truncating any record.

## Excluded from this round

GPT rationale generation or distillation, PoT/graph pipeline changes, image
rendering or vision training, GRPO/DPO, four independent per-task adapters,
multi-GPU, fixing the existing XLNet baseline, retraining on the full
combined dataset after validation, and broad hyperparameter sweeps. See the
planning notes for the full phased rationale and GPU budget derivation.
