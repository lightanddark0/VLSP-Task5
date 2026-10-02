# Vietnamese Spatial QA Experiments

Baselines for the VLSP ViSpatialQA task (Vietnamese SPARTQA): a shared XLNet
encoder with task-specific heads and a GPT-4.1-mini API experiment using a
spatial reasoning prompt. Both support YN, FR, FB, and CO questions.

## Repository Layout

```text
spartqa/
  data.py                 JSON I/O, answer labels, validation, API input records
  metrics.py              Accuracy, exact-match, and Jaccard scoring
  api.py                  Structured requests and sanitized API response handling
  splits.py               Seeded story-level train/dev splitting
  submission.py           Submission structure/format checks and fallback filling
  repro.py                set_seed and run metadata for fine-tuning runs
  prompting.py            Qwen3 chat prompts (F and L-CoT) and answer parsing
  predictions.py          Shared JSONL prediction format, resume, and filling
  postprocess.py          FR constraints, Human YN DK policy, indifinite rules
  voting.py               Self-consistency and weighted source voting
  inference.py            Inference jobs, fallback answers, dev scoring
  hub.py / tracking.py    Hugging Face Hub storage and W&B logging (optional)
  symbolic/               Branch S: story/question parser and spatial reasoner
  augment.py              Converse augmentation of Human FR/YN training questions (E2)
  lp_prompting.py         Branch LP (E3) prompts: story -> world JSON, question -> form JSON
make_splits.py            Create or rebuild Data/splits from the committed manifest
evaluate.py               Official metrics, Final Score, and story bootstrap CIs
validate_submission.py    Check (and optionally fill) a prediction file
explore.py                Dataset statistics and pipeline assumption checks
prepare_sft.py            Branch F training JSONL (Auto stage, Human stage)
train_lora.py             Branch F: Qwen3-8B LoRA fine-tuning with Hub resume
infer_f.py                Branch F: vLLM inference with the LoRA adapter
llm_cot.py                Branch L-CoT: Qwen3-32B-FP8 thinking + self-consistency
ensemble.py               Dev-tuned ensemble, post-processing, submission files
solve_symbolic.py         Branch S: rule-based answers for Auto (CPU), abstains otherwise
compare_branches.py       Scores of each branch, each pair, and all three, from saved predictions
merge_predictions.py      Replace some question types of one prediction file with another's
llm_parse.py              Branch LP (E3): Qwen3-32B writes worlds/forms as JSON, S reasoner answers
solve_parsed.py           Branch LP (E3), CPU: re-answer saved parses, tune conventions on Human train
crossfit.py               A1: story folds of Human train, out-of-fold merge, train+dev tuning split
analyze_errors.py         Human error analysis on out-of-fold predictions (recoverable errors, FR kinds, LP causes)
stack_ensemble.py         Stacking: per-type logistic regression over source confidences, same folds as cv
run_colab_e1e2.ipynb      Colab notebook for experiments E1 (FR checklist) and E2 (converse data)
run_colab_e3.ipynb        Colab notebook for experiment E3 (branch LP) and its comparison
run_colab_e4.ipynb        Colab notebook for E4: LP+ (CPU) and F2C cross-fit, ensembles tuned on 613 Human questions
run_analysis_e5.ipynb     CPU notebook: error analysis and stacking on the E4 predictions
run_colab_e6.ipynb        E6: FR DK fix, nested stacking, group A/B scores, LP re-sampling of unreadable worlds
run_colab_e7.ipynb        E7: extra L-CoT source L2 (Qwen3-30B-A3B-Thinking-2507), three-variant comparison, final submission
run_colab.ipynb           Colab H100 notebook that runs the whole pipeline
gpt_experiment.py         API experiment CLI, cache, exports, and configuration
XLNER.py                  XLNet training/evaluation/prediction CLI
test_gpt_experiment.py    Mocked API and resume tests
test_spartqa.py           Shared utilities and Git publication tests
test_evaluation.py        Split, metric, submission, and seed tests
test_pipeline.py          Prompt, post-processing, voting, and ensemble tests
test_symbolic.py          Branch S parser, reasoner, and question tests (synthetic stories)
Docs/
  spartqa_cot.txt          Versioned default prompt
  gpt_experiment.md        API experiment options and output details
  task_description.md     Original task description
Data/                     Organizer-provided JSON files (local only)
outputs/                  Results and checkpoints (local only)
.env.example              Public configuration template with no credentials
.github/workflows/        Offline tests on Windows and Linux
```

Run commands from the repository root with Python 3.11. Code modules in
`spartqa/` have no third-party import requirements. Running API experiments does
not require PyTorch; running XLNet does not require an OpenAI account.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. Obtain the
datasets from the organizers and follow [Data/README.md](Data/README.md).
Competition data is not redistributed here.

## GPT API Experiment

```powershell
python -m pip install -r requirements-api.txt
```

Create your local `.env` from `.env.example` only if it does not already exist.
Set `OPENAI_API_KEY` privately. Never paste keys into chat, logs, or commits.

- `OPENAI_BASE_URL`: your authorized OpenAI-compatible endpoint, including its
  API path if required. The default is the official OpenAI API.
- `OPENAI_MODEL`: exact model/deployment ID accepted by that endpoint.
- Precedence: `--model`, then `OPENAI_MODEL`, then `BASE_MODEL_ID`, then the
  pinned public snapshot `gpt-4.1-mini-2025-04-14`.
- Existing shell variables override `.env`. Use `--env-file` for another file.

Some gateways use `azure-4.1-mini` instead of `gpt-4.1-mini`. Check the permitted
model IDs on your provider; a public OpenAI alias may not be a valid deployment.
Do not use an internal API key against an unrelated endpoint.

```powershell
python gpt_experiment.py --dry-run
python gpt_experiment.py --max-questions 10 --output-dir outputs/gpt41mini_cot_human_train
python gpt_experiment.py --output-dir outputs/gpt41mini_cot_human_train
```

The final command resumes the same experiment on all questions. Successful
responses are cached, so they are not requested again. Failed requests can be
retried. Use a new output directory when changing the prompt, endpoint, model,
generation settings, or dataset. CLI names and existing cache configuration
version 2 remain compatible with runs made before the shared-module refactor.

Requests send stories and questions to the configured service and may incur
charges. Gold answers and reasoning metadata are excluded from requests.
Outputs include predictions in the input JSON structure, per-question
explanations, recorded token usage, and per-task metrics. An incomplete run is
marked `partial`; questions without predictions do not retain their gold answers.

See [Docs/gpt_experiment.md](Docs/gpt_experiment.md) for details. To experiment
with another prompt, pass `--prompt-file` and a new `--output-dir`; to use another
dataset, pass `--input`. The default run evaluates Human train, not a holdout.

## XLNet Baseline

Install the appropriate CPU/CUDA build of PyTorch for your machine, then:

```powershell
python -m pip install -r requirements.txt
python XLNER.py train --train-files Data/human_train.json Data/auto_train.json --output-dir outputs/xlnet_large
python XLNER.py predict --checkpoint-dir outputs/xlnet_large --output-dir outputs/xlnet_large_predictions
```

Defaults: `xlnet/xlnet-large-cased`, 3 epochs, batch size 2, accumulation 8,
maximum length 384, seed 42, and internal eval ratio 0.2. The train command
predicts both public test files by default. Pass `--test-files` without values
to skip public-test prediction. Use `--help` for hyperparameter options.

Model weights require a download and substantial memory. Only load checkpoints
from trusted sources. Checkpoint keys, head dimensions, and task-ID order are
unchanged by the refactor. The two CLIs retain their previous arguments.

## Splits, Evaluation, and Submission

These tools need only the standard library, so any fine-tuning code can use them.

```powershell
python make_splits.py                   # first time: writes Data/splits/* and manifest.json
python make_splits.py --from-manifest   # elsewhere: rebuilds byte-identical files
```

Splits are by **story**, never by question, so no story appears in both train
and dev. Defaults: seed 42, Human dev 20% (7 stories, 116 questions), Auto dev
10% (1,308 stories, 10,201 questions). Among 100 seeded shuffles, the dev set
closest to the full file's question-type and label distribution is kept.
`Data/splits/manifest.json` stores only story indices and source SHA-256
checksums, so it is committed; the split JSON files stay local like the data.
`--from-manifest` refuses to run if the source files have changed.

Fine-tune on `*_train.json`, select checkpoints and thresholds on `*_dev.json`,
and never tune on the public test. Predictions must keep the input JSON
structure and only fill `answer`:

```powershell
python evaluate.py --human outputs/run/human_dev_pred.json Data/splits/human_dev.json `
                   --auto outputs/run/auto_dev_pred.json Data/splits/auto_dev.json `
                   --bootstrap 1000 --output outputs/run/dev_eval.json
python validate_submission.py outputs/run/human_public_test_pred.json --reference Data/human_public_test.json
```

`evaluate.py` reports YN/CO accuracy and FR/FB exact match and Jaccard per
dataset, the official Final Score `(Human + Auto) / 2` per metric, breakdowns by
`reasoning_type`, and optional 95% intervals from resampling whole stories.
Missing or invalid answers count as wrong. `primary_macro_unofficial`, the mean
primary metric over the four types, is only a model-selection convenience.
Human dev is small: its intervals are about ±10 points, so small differences
there are not meaningful.

`validate_submission.py` fails if anything other than `answer` changed or an
answer is missing, mistyped, outside the candidates, or mixes FR label 7 with
others. With `--fill-from Data/human_train.json --output FILE`, invalid answers
are replaced by the most frequent training answer of that type and listed.
This majority fallback scores Final ≈ 0.33 (mean primary) on dev, a floor any
model should exceed.

In training scripts, call `spartqa.repro.set_seed(seed)` before creating
models and loaders, and save `spartqa.repro.run_metadata(seed=seed, ...)` with
each run (git commit, dirty flag, package versions, time).

## Qwen3 Pipeline on Colab (H100)

`run_colab.ipynb` clones this repository, installs `requirements-colab.txt`,
and runs every step as a separate `python script.py` process, so no model stays
in the notebook kernel. Edit the `CONFIG` cell or keep its defaults, then Run
all. Secrets: `HF_TOKEN` (write), `WANDB_API_KEY`, optionally `WANDB_ENTITY`
and `GITHUB_TOKEN`. Put the four data files in the Drive folder `DRIVE_DATA_DIR`.

| Step | Script | Output |
| --- | --- | --- |
| Statistics and checks | `explore.py` | `outputs/explore/report.json` |
| F stage 1 (all Auto train, 1 epoch) | `prepare_sft.py`, `train_lora.py` | adapter on the Hub |
| F1 inference, backup submission | `infer_f.py`, `ensemble.py --only F1` | `outputs/submission_backup/` |
| F stage 2 (Human + 2k Auto) | `prepare_sft.py`, `train_lora.py --init-adapter` | adapter on the Hub |
| L-CoT for Human (n=8) | `llm_cot.py` | vote shares per label |
| Ensemble and submission | `ensemble.py` | `outputs/submission/<dataset>/` |

Every step resumes: training continues from the newest checkpoint in
`DRIVE_CKPT_DIR` on Google Drive, and inference appends predictions to
`outputs/predictions/<source>/<dataset>_<split>.jsonl` in chunks, which the
notebook backs up to Drive and restores in a new session.

Storage: the only thing sent to Hugging Face is each final adapter, as a public
repo with a one-line model card and a `run_config.json` holding the base model
and prompt version (`HF_PUBLIC_MODELS=False` sends nothing). Checkpoints, data,
predictions, and submissions stay on Drive, since they contain the organizers'
stories and questions. W&B receives metrics, configs, timings, and small tables
only; runs record the adapter repo and commit hash.

Branch S (`solve_symbolic.py`, section 17 of the notebook) reads a generated
story into objects and facts, closes them under converse, symmetry,
transitivity, block inheritance, and opposite-edge rules, and answers the four
question types from that closure. It abstains when a story or question does
not parse, which is the case for all Human stories (free text). On Auto dev it
answers about 99.5% of questions, with accuracy on those of about 97.8% (YN),
99.5% (FR), 98.1% (FB), and 99.5% (CO). `compare_branches.py` then tunes and
scores F, S, L, every pair, and all three on dev (in-sample and 5-fold
cross-validated over stories) and writes a submission per combination, without
running any model.

Experiments E1/E2 for the Human subset (`run_colab_e1e2.ipynb`, branch
`feat/fr-checklist-augment`) reuse the saved F1/F2/S/LCOT predictions:
- E1: `llm_cot.py --fr-mode checklist` asks each FR axis (left/right,
  above/below, near/far, touch) as its own question with its own samples and
  merges them; it is scored on Human train and dev (L is not trained on Human),
  and `merge_predictions.py` puts its FR answers into the L predictions (LCOTX).
- E2: `prepare_sft.py --augment-converse FR,YN` adds converse copies of Human
  train questions (111 FR, 29 YN) before retraining stage 2 (source F2C).
- `compare_branches.py --branches` compares baseline, E1, E2, and E1+E2.

Result on Human dev (cv): E1 did not help (Human FR 0.714 -> 0.643 in the best
combination); E2 raised Human cv from 0.764 to 0.790 (Final cv 0.8756 -> 0.8889),
within the noise of 116 questions but kept (source F2C).

Experiment E3, branch LP (`run_colab_e3.ipynb`, branch `feat/e3-llm-parse`),
separates reading from reasoning for Human, where the S parser cannot read
the free text:
- `llm_parse.py`: Qwen3-32B samples `--n-world` worlds per story (objects,
  stated facts, edge contacts) and `--n-form` logical forms per question
  (schema in `spartqa/symbolic/structured.py`). The S reasoner answers every
  (world, form) pair; answers are voted, and a question with no usable pair
  is abstained. All JSON is kept in `<output>.parses.jsonl`.
- `solve_parsed.py --grid`: re-answers the parses under every convention
  (closed/open YN, FR over all/any object pairs, FB scope, FAR along one
  direction, FAR across blocks) on Human train, without the GPU, and writes
  the best one; `--set` applies it to dev/test (source LPT).
- Hand annotations of three Human train stories
  (`spartqa/symbolic/lp_annotations.json`, facts only, text read from Data/)
  are the prompt example (story 0, skipped when predicting) and an oracle test:
  with perfect transcription the reasoner answers 39/48 questions, 42/48 with
  FAR across blocks (FB 12/12). The rest are gold labels the text contradicts.
- `compare_branches.py --branches "F=F2C,F1 S=S L=LCOT P=LPT"` adds LP as a branch.

Result (Human dev cv): LP alone 0.790 (FB 0.931); F+S+L+P 0.850 against 0.790
without LP, Final cv 0.8889 -> 0.9185.

E4 (`run_colab_e4.ipynb`, branch `feat/e4-lp-crossfit`):
- LP+ on CPU from the saved E3 parses (`solve_parsed.py`): rule-based forms for
  the common Human question shapes (`spartqa/symbolic/human_forms.py`, 84% of
  Human questions; `forms_mode`), relaxed matching (`relax`), `definite`,
  `fr_block_share`, fact-level merging of sampled worlds (`world_merge`), and a
  learned FR near/far table (`--fit-fr-calibration`, scored out of fold with
  `--oof-calibration`). Switches are chosen on Human train; the result is source LPX.
- A1: `crossfit.py` makes story folds of Human train; F2C is retrained per fold
  and predicts its held-out stories, F1 and L also predict Human train, and
  `crossfit.py combine` builds `human_trdev` (613 questions) so
  `compare_branches.py --tune-splits human=trdev` tunes and cross-validates
  ensembles on train + dev instead of 116 dev questions.

Result (Human cv, ensembles tuned on train + dev, 613 questions): F+S+L 0.795,
+LP 0.836, +LPX 0.846 (Final cv 0.8910 / 0.9115 / 0.9165). LPX chose
`forms_mode=rule_plus` and `world_merge=both`; the FR near/far table did not
help out of fold and is off. Per type with LPX: YN 0.857, FR 0.785, FB 0.922,
CO 0.819. F2C held-out folds range 0.63-0.79, so 7-story dev scores are noisy.

Public-test scores (Codabench overall rank score, mean of the 8 Human/Auto metrics):
E4 0.8887, E6 0.8913, **E7 0.8963 (best so far)**. E7 (`run_colab_e7.ipynb`) runs
L-CoT with a second thinking model on Human train/dev/test (source LCOT2, tolerant
answer parsing for the 2507 template), compares base / +L2 / L2-replaces-L on the 613
Human questions and writes the best variant. The 0.8963 file was made with
`HUMAN_YN_DK_FINAL="no"`; E4, which kept DK, scored Human YN 47/51 against 44/51,
so the notebook default is now `"keep"` (not yet re-submitted). Human public test
has only 132 questions, so differences of 1-2 questions are noise.

`ensemble.py` picks, per question type, the best single source or weighted vote
on dev, applies post-processing, validates the submission structure, and writes
an ablation table. Two flags are decisions for the organizers' rules:

- `--use-indifinite` (default off): the question field `indifinite` is True
  exactly when YN is DK or FR is [7] in both training files, and it is present
  in the public test files. The rule forces DK there and forbids it elsewhere.
  It is not a documented input, so enable it only if the organizers allow it.
  The ablation always reports the dev score with the other setting.
- `--human-yn-dk keep|no`: Human YN is documented as Yes/No, but Human train
  has 16 DK answers. `no` maps DK to the likelier of Yes/No.

Local smoke test without a GPU (small model, CPU):

```powershell
python prepare_sft.py --stage auto --limit 200 --output-dir outputs/smoke/sft
python train_lora.py --train-file outputs/smoke/sft/train.jsonl --model-name Qwen/Qwen3-0.6B `
    --output-dir outputs/smoke/ckpt --limit 16 --max-steps 3 --batch-size 2 --no-push
```

vLLM inference (`infer_f.py`, `llm_cot.py`) needs a CUDA GPU.

## Metrics and Caveats

| Task | Answer | Metrics |
| --- | --- | --- |
| YN | One of `Yes`, `No`, `DK` | Accuracy |
| CO | One integer from 0 to 3 | Accuracy |
| FR | Relation indices from 0 to 7 | Exact match and Jaccard |
| FB | Zero or more candidate block IDs | Exact match and Jaccard |

FR/FB are compared as sets. Two empty FB sets have Jaccard 1. Missing API
predictions score 0, even when the gold FB answer is empty. Human train contains
DK labels despite the binary-YN description in the original task document.

This is a structural refactor, not a change to experimental methodology:

- Story-level dev splits now exist (`make_splits.py`); the API runner and XLNet
  script do not use them yet.
- XLNet still splits questions internally; stories may overlap between train
  and eval. Its combined-file metrics are pooled, not Human/Auto macro-averaged.
- XLNet still truncates the story/question text as one sequence; long stories
  can remove the question. Check the tokenizer lengths before interpreting scores.
- The existing scheduler counts batches rather than optimizer updates under
  gradient accumulation. Automatic post-training prediction uses the last
  in-memory epoch; standalone `predict` loads the saved best checkpoint.
- Public test files have no gold answers, so local public-test accuracy cannot
  be calculated. API scores on the full training file are not held-out results.

Use a fixed, story-disjoint split per dataset and report Human and Auto scores
separately before comparing future configurations. Consult the original
[task description](Docs/task_description.md) for the official metric averaging.

## Offline Checks

```powershell
python -m unittest discover -v
python -m compileall -q spartqa *.py
```

The tests need only Python's standard library. Git is needed for the ignore-rule
test. No API calls, credentials, datasets, or model downloads are required.
GitHub Actions runs these checks on Windows and Linux. It does not validate
live gateway access, GPU training, or full checkpoint loading.

## Publish Safely

Keep `.env`, private data, model weights, scratch notebooks, reference PDFs in
`Docs/`, and `outputs/` local; `.gitignore` excludes them. The public `.env.example` contains no key or
internal endpoint. Do not force-add excluded artifacts. Review custom prompts
and documents too, since they can contain private examples or infrastructure URLs.

After initializing your own Git repository, inspect `git status --short`,
`git status --ignored --short`, and `git diff --cached --stat` before committing.
If sensitive files were already tracked, ignore rules alone will not remove
them from history: untrack them and rotate exposed keys before publishing.
Do not assume the organizer data or third-party task text can be redistributed
under a new license. This repository does not grant rights to those materials.Skip to main content

Association for Vietnamese Language and Speech Processing
A chapter of VAIP - Vietnam Association for Information Processing

English
Vietnamese
Home
Main navigation
About
News and Events
Conferences
Evaluation Campaigns
Resources
Home
VLSP 2026
VLSP 2026 Evaluation Campaign
VLSP 2026 challenge on Spatial Reasoning on Textual Question Answering
VLSP 2026 challenge on Spatial Reasoning on Textual Question Answering
Important dates 
Sep 10, 2026: Registration open
Sep 18, 2026: Training data release
Sep 25, 2026: Public test release
Sep 30, 2026: System submission deadline
Oct 9, 2026: Private test results release
Oct 15, 2026: Result announcement
Oct 25, 2026: Paper submission
Nov 5, 2026: Notification of acceptance
Nov 12, 2026: Camera-ready deadline
Nov 15, 2026: Workshop date
Task Description
Objective: Build a system to answer spatial questions in Vietnamese. The system must understand spatial descriptions, identify relationships among objects and blocks, and perform reasoning over the information provided in a story to produce the correct answer.

The ViSpatialQA task consists of four question types: Yes/No (YN), Find Relation (FR), Find Block (FB), and Choose Object (CO).

Question Type 1: Yes/No (YN)
Description: Determine whether a proposed spatial relationship is supported by the story. Depending on the dataset, the answer may be Yes, No, or DK when the relationship cannot be determined from the available information.
Focus: Verify explicit or inferred spatial relationships through single-label classification.
Question Type 2: Find Relation (FR)
Description: Identify all spatial relationships between two entities mentioned in the question. A pair of entities may have more than one valid spatial relationship.
Focus: Predict one or more labels from a fixed inventory of eight possible relation labels.
Question Type 3: Find Block (FB)
Description: Identify all blocks that satisfy the condition expressed in the question. Depending on the story, the answer may contain one block, multiple blocks, or no block.
Focus: Reason about objects, blocks, quantifiers, and containment information to select the correct set of blocks.
Question Type 4: Choose Object (CO)
Description: Determine which of two candidate objects satisfies the spatial condition stated in the question. The correct answer may be the first object, the second object, both objects, or neither object.
Focus: Compare candidate objects and select the correct answer from four possible choices.
Questions may be answered using information explicitly stated in the story or through reasoning operations such as converse, symmetry, transitivity, quantification, negation, and list/set reasoning.

Dataset
The ViSpatialQA shared task uses two complementary datasets: ViSPARTQA-Human and ViSPARTQA-Auto. Both datasets contain Vietnamese spatial stories and questions covering the four question types: YN, FR, FB, and CO.

ViSPARTQA-Human: This dataset is derived from the human-authored portion of SPARTQA and localized into Vietnamese through machine translation followed by manual post-editing. This process preserves the original spatial information, questions, and answers while improving the naturalness and correctness of the Vietnamese text. The dataset contains 86 stories and 1,123 questions.
ViSPARTQA-Auto: This dataset is constructed using a Vietnamese-adapted generation pipeline based on language-independent spatial scene representations. Its stories, questions, and answers are generated directly in Vietnamese. The dataset contains 16,954 stories and 132,971 questions, providing large-scale supervision with controlled spatial reasoning structures.
Together, the two datasets combine relatively natural linguistic expressions from human-authored data with large-scale and systematically generated spatial reasoning examples.

Data Format
The datasets are stored as JSON files. Each file contains a name field and a data list. Each item in the list consists of a spatial story and its associated questions.

{

  "name": "SPaRTQA",

  "data": [

    {

      "story": [

        "Có một khối tên là A. Trong A có ..."

      ],

      "questions": [

        {

          "q_id": 1,

          "q_type": "FR",

          "question": "Mối liên hệ giữa hình vuông màu vàng và vật thể màu đen là gì?",

          "candidate_answers": ["bên trái", "bên phải", "bên trên", "bên dưới", "gần tới", "xa khỏi", "chạm vào", "DK"],

          "answer": [2, 5]

        }

      ]

    }

  ]

}

The principal fields are:

story: A list containing the Vietnamese spatial description.
questions: A list of questions associated with the story.
q_id: The question identifier.
q_type: The question type: YN, FR, FB, or CO.
question: The question written in Vietnamese.
candidate_answers: The available answer options, when applicable.
answer: The gold answer, represented as a list. Its format depends on the question type.
Answer Formats
Yes/No (YN): The answer is a list containing exactly one string. The possible values are "Yes" and "No" in ViSPARTQA-Human. ViSPARTQA-Auto additionally includes "DK", indicating that the answer cannot be determined from the story.
Find Relation (FR): The answer is a list containing one or more integer indices. Each integer refers to a label in the fixed relation inventory. Multiple relations may be correct for the same question.
Find Block (FB): The answer is a list containing zero or more block identifiers, such as "A", "B", or "C". The valid block identifiers are provided in candidate_answers. An empty list indicates that no candidate block satisfies the condition.
Choose Object (CO): The answer is a list containing exactly one integer: 0 for the first candidate object, 1 for the second candidate object, 2 for both objects, and 3 for neither object.
FR Relation Inventory
FR questions use the following fixed list of eight labels. The index of each label is used in the answer field.

0: bên trái, 1: bên phải, 2: bên trên, 3: bên dưới, 4: gần tới, 5: xa khỏi, 6: chạm vào, 7: DK

Evaluation
ViSPARTQA-Human and ViSPARTQA-Auto are evaluated separately. For each evaluation metric, the final score is calculated as the arithmetic mean of the corresponding scores obtained on the two datasets:

Final Score = (Score on ViSPARTQA-Human + Score on ViSPARTQA-Auto) / 2 

Different metrics are used depending on the question type.

Evaluation Metrics
Accuracy: Used for YN and CO questions. A prediction is counted as correct only when the predicted label matches the gold label. YN is treated as binary classification in ViSPARTQA-Human and three-class classification in ViSPARTQA-Auto. CO is treated as four-class classification. 
Accuracy = Number of correctly answered questions / Total number of questions
Exact Match: The primary evaluation metric for FR and FB questions. A prediction is counted as correct only when the complete predicted answer set is identical to the gold answer set. The order of elements does not affect the result. 
Exact Match = Number of questions with exactly matched answer sets / Total number of questions
Jaccard Score: A supplementary evaluation metric for FR and FB questions. It measures the overlap between the predicted answer set and the gold answer set and gives partial credit to partially correct predictions. 
Jaccard(P, G) = |P ∩ G| / |P ∪ G| 
where P denotes the predicted answer set and G denotes the gold answer set.
Accuracy is reported for YN and CO, while Exact Match and Jaccard Score are reported for FR and FB. Each metric is first calculated separately on ViSPARTQA-Human and ViSPARTQA-Auto. The corresponding final result is then obtained by averaging the two dataset-level scores.

Example for YN or CO
Gold answer: ["Yes"]

System prediction: ["Yes"]

The predicted label matches the gold label.
→ Accuracy = 1.0
Example for FR or FB
Gold answer: [2, 5]

System prediction: [2]

Exact Match: The predicted answer set does not contain all labels in the gold answer set.
→ Exact Match = 0.0
Jaccard Score: The intersection contains one label, while the union contains two labels.
→ Jaccard = 1 / 2 = 0.5
If both the predicted answer and the gold answer for an FB question are empty sets, they are considered an exact match and assigned a Jaccard score of 1.0.

Training and Test Data
The organizers will provide:

Training and Development Data: Vietnamese spatial stories and questions in JSON format, together with their gold answers.
Test Data: Vietnamese spatial stories and questions following the same JSON structure. Gold answers will not be included in the released test files.
ViSPARTQA-Human and ViSPARTQA-Auto will be released and evaluated as separate datasets.

Submission
Participants must submit predictions using the same JSON structure as the corresponding test file. Questions of all four types must remain in their original stories and question lists; separate submission files for YN, FR, FB, and CO are not required.

For each question, participants must fill the answer field with the system prediction while preserving the original data structure, question identifiers, question types, questions, and candidate answers. The predicted answer must follow the format required by its question type.

A separate prediction file must be submitted for each test dataset: one for ViSPARTQA-Human and one for ViSPARTQA-Auto. Detailed file-naming conventions and submission procedures will be announced later.

Registration
Click here to register 

Contact
Zalo Group: ......

Organizers
Nguyen Thi Minh Huyen, email: ntmhuyen@gmail.com
Ha My Linh, email: halinh.hus@gmail.com
Pham Thi Duc, email: phamthiduc@hus.edu.vn
Le Ngoc Toan, email: lengoctoan@hus.edu.vn
References
 
 Share This Page
Sponsors and Partners
VinBIGDATA  VinIF  AIMESOFT  bee  Dagoras            

 

  zalo   VTCC  VCCorp

 

 

IOIT HUS  USTH  UET    TLU  UIT  INT2  jaist  VIETLEX

 


 

© 2026 Association for Vietnamese Language and Speech Processing, All rights reserved.
This site uses cookies. By continuing to browse the site you are agreeing to our use of cookies. I agree

