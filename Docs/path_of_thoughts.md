# Path-of-Thoughts on ViSPARTQA

## Paper and scope

Reference: Ge Zhang et al., [Extracting and Following Paths for Robust Relational
Reasoning with Large Language Models](https://arxiv.org/abs/2412.17963v2),
arXiv:2412.17963v2, 23 March 2026, TMLR 2026. Version 1 used the title
"Path-of-Thoughts: Extracting and Following Paths ...".

This is a **method adaptation**, not a reproduction of the paper's numerical
results or its original prompts. The closest paper experiment is PoT-LLM on
SPARTUN **Find Relation** questions. The paper evaluates the first 1,000 FR test
questions, not Vietnamese SPARTQA or its YN/FB/CO tasks. It does not evaluate
PoT-Symbolic on SPARTUN. The original implementation (pot_version 1) supported
FR only. **PoT v2 now supports YN, FR, FB and CO by default**, using multi-pair
query plans and joint reasoning over the selected paths. This is an additional
adaptation, not the paper's independent-path FR union procedure.

The relevant parts of the paper are sections 4.1-4.3 and 5.1, appendix A.2
(multiple paths versus one shortest path), A.5/table 29 (spatial extraction),
and A.6/table 34 (spatial LLM reasoning).

## Pipeline

1. **Extract:** one LLM call per question produces entity IDs/descriptions,
  explicit spatial triples, and a query with source/target, `pairs`, and `focus`.
  Extraction sees only story, question, task and candidates, never gold labels
  or reasoning annotations. The prompt audits active-block membership, pronouns,
  repeated objects, ordinals, group expansion, clause direction and side contact.
  Boundary entities preserve named sides without counting as inventory objects.
  Only synthetic examples are in the prompt, not evaluation stories or answers.
2. **Identify paths:** iterative DFS searches each distinct queried pair.
  Traversal can follow either edge direction, but statements retain their
  original meaning. Parallel relations are preserved and identical triples are
  deduplicated. Each pair gets the path/hop cap; all pairs share one expansion
  budget. Searches record disconnected, unresolved and budget-limited cases.
3. **Assemble evidence:** provide the paths, their distinct facts, full entity
  inventory, containment facts, facts incident to focused entities, and original
  story. If any pair lacks a path, no pairs resolve, or any limit is hit, include
  the full graph as `fallback_graph` evidence and record the fallback. Missing
  paths are not negative facts. The original story remains authoritative.
4. **Reason jointly:** one LLM call evaluates the complete question across the
  evidence. YN uses Yes/No/DK, FB checks every candidate block, CO checks both
  alternatives before selecting 0/1/2/3, and FR returns supported relation labels.
  No union or vote over independently predicted answers is used. This is needed
  for quantifiers and combined choices and avoids automatically merging
  contradictory path predictions. It does not guarantee semantic correctness.

A successful uncached question normally uses **two API calls**, extraction plus
joint reasoning, regardless of path count (SDK retries can add requests). Long
graphs and many pairs still increase token usage. **Extraction is deduped per
story within a run**: the first question of a story runs a real extraction call;
every later question about the same story reuses that graph (entities/triples
only) instead of re-extracting, with its own `query` reset to unresolved (this
is free in practice: `fallback`/`pot_diagnostics` show query-restricted path
search already falls back to the full graph in effectively every real run seen
so far, so an unresolved query costs nothing). Reused extractions are marked
`reused_extraction: true` in `responses.jsonl`, same as `--graph-cache-dir`
imports, and counted under `pot_diagnostics.reused_extractions`. Pass
`--extract-only` to build just the extraction cache (one real call per unique
story) and skip every reasoning call entirely; rerunning the same command
without `--extract-only` on the same `--output-dir` fills in reasoning for the
already-extracted questions without re-extracting.

The graph uses the paper's 15 spatial relation names, including containment
and topology. Output uses ViSPARTQA's fixed inventory:

| Index | Relation |
| --- | --- |
| 0, 1 | left, right |
| 2, 3 | above, below |
| 4, 5 | near, far |
| 6 | touching |
| 7 | unknown (DK), alone |

Containment is useful intermediate evidence but is not an FR output label.
Overlap is not mapped to touching. Directional relations are not unit-distance
vectors: unstated coordinates or distances must not be invented.

## Differences from the paper

- Data/language: ViSPARTQA, not SPARTUN; eight output labels instead of fifteen.
- Backbone: the configured deployment, not necessarily the paper's snapshots.
  The pilot used `azure-4.1-mini`; response model IDs are saved for inspection.
- Prompts: structured JSON with two synthetic extraction examples and the
  repository's existing CoT reasoner plus task-aware joint-path instructions. These
  are not the paper's few-shot reasoning examples. No evaluation examples were
  inserted into prompts.
- Sampling: repository defaults are temperature 0 and 1,200 reasoning output
  tokens; extraction gets at most 8,192 in PoT v2 (previously 4,096). Paper
  version 2 section 5.1 uses temperature 0.3 and
  4,096 output tokens for general-purpose models. Version 1 had inconsistent
  temperature descriptions; use version 2 when specifying a paper-like run.
- Search: default at most 16 paths per pair and 10,000 DFS edge examinations per question.
  `--max-paths 0 --max-hops 0` removes path-count and hop caps, but the expansion
  budget remains. Check `limits_hit` before describing any run as exhaustive.
  Path enumeration can be exponential; limited DFS order is not shortest-first.
- Metrics: the paper counts a hit when predicted and gold relation sets overlap.
  Here `hit_accuracy` is supplementary; exact match and Jaccard remain primary.
  Their disagreement matters for multi-label prediction.

## Run all four question types

The default input is still `Data/human_train.json`. To run the active public
test file, use a new output directory. On the previously verified gateway:

```powershell
python gpt_experiment.py --method pot --input Data/human_public_test.json --model azure-4.1-mini --dry-run
python gpt_experiment.py --method pot --input Data/human_public_test.json --model azure-4.1-mini --max-paths 4 --max-output-tokens 2048 --output-dir outputs/pot_v2_human_public_test
```

Omitting `--tasks` selects YN, FR, FB and CO. `--tasks YN FR FB CO` is equivalent.
The model ID must match your configured provider. The second command sends data
to that provider and incurs charges; dry-run makes no API calls or output files.
Rerun the identical command to resume. The prediction file will be
`outputs/pot_v2_human_public_test/human_public_test_predictions.json`.
Require `status: complete` and full coverage before using it as a full-file
prediction. Public test has no gold labels, so no local accuracy is available.

For labeled Human evaluation or Auto prediction, change the input and output:

```powershell
python gpt_experiment.py --method pot --input Data/human_train.json --model azure-4.1-mini --max-paths 4 --max-output-tokens 2048 --output-dir outputs/pot_v2_human_train
python gpt_experiment.py --method pot --input Data/auto_public_test.json --model azure-4.1-mini --max-paths 4 --max-output-tokens 2048 --output-dir outputs/pot_v2_auto_public_test
```

These full API runs have not been performed for v2. Pilot on a small development
subset before a large Auto run. `--max-questions` truncates the selected sequence;
it is not stratified sampling and a short prefix may not contain all four types.

## Run a matched FR pilot

Install `requirements-api.txt` and configure the existing private `.env` as
described in [gpt_experiment.md](gpt_experiment.md). No additional dependencies,
GPU, training, or data download are needed. Run from the repository root:

```powershell
python gpt_experiment.py --method pot --tasks FR --max-questions 5 --max-paths 4 --dry-run
python gpt_experiment.py --method cot --tasks FR --max-questions 5 --output-dir outputs/cot_fr_v2_pilot5
python gpt_experiment.py --method pot --tasks FR --max-questions 5 --max-paths 4 --output-dir outputs/pot_v2_fr_pilot5
python gpt_experiment.py --method pot-no-path --tasks FR --max-questions 5 --graph-cache-dir outputs/pot_v2_fr_pilot5 --output-dir outputs/pot_v2_no_path_fixed_graph_fr_pilot5
```

Task filtering happens **before** `--max-questions`, so all commands use the
same five FR questions. The explicit `--tasks FR` is now required to restrict
PoT to FR. FR-only exports leave other
questions unanswered and are **not complete competition submissions**.

`pot-no-path` supplies the whole graph in one reasoning call. Use
`--graph-cache-dir` to hold the graph and query extraction fixed. Even
temperature 0 does not guarantee identical extractions across API calls.
Without this option, the ablation independently re-extracts each graph and is
only an end-to-end pipeline comparison, not an isolated path-stage comparison.

Cache reuse checks the input hash, extraction prompt, model, endpoint,
temperature, extraction token limit, and PoT version. Every selected question
must have a successful source extraction. A graph-cache fingerprint is stored
in the destination configuration; changing source extractions requires a new
output directory. Keep the source run frozen during a controlled ablation.

## Historical v1 results observed on 26 September 2026

Five Human-train FR questions, from the first two stories; **not a held-out
benchmark**. All methods answered 5/5. These numbers are from the old FR-only
prompt and independent-path union, before the v2 prompt revision. They are NOT
measurements of the current pipeline. No live v2 improvement has been established.

| Method | Exact match | Mean Jaccard | Hit accuracy | New API calls | Recorded tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoT | 20% | 43.33% | 80% | 5 | 6,224 |
| PoT, at most 4 paths | 0% | 33.33% | 80% | 16 | 26,083 |
| No paths, same cached graphs | 0% | 33.33% | 100% | 5 | 9,694 |

Cached ablation cost is **incremental reasoning only**: its five extraction
calls were already paid for in PoT. Do not compare that number to end-to-end
PoT cost without adding the source extraction usage. An earlier independent
extraction/no-path pilot also completed: 0% exact match, 33.33% Jaccard, 80% hit
accuracy, 10 calls and 19,012 tokens. Across these four runs, 36 API calls and
61,013 tokens were recorded. Provider billing can differ from recorded usage.

PoT used 11 reasoning calls for 5 extractions, with no fallbacks. One question
hit the four-path cap and two had disagreeing path predictions. For key `0:10`,
gold was `[1,4]`, one path returned `[1,4]`, another `[0]`, and union produced
`[0,1,4]`: a hit but an exact-match failure. This illustrates a risk of using
the paper's permissive hit metric for this task. It is not evidence that gold
labels should be changed or contradictory predictions silently discarded.

There is **no observed improvement over CoT in this pilot**, and five clustered
questions cannot establish statistical significance or overall performance.
All raw local records and metrics are under the four `outputs/*fr_pilot5`
directories. They are excluded from Git; source data was not modified.

## Larger experiments

Freeze prompts and create story-disjoint development/evaluation data before
tuning. Keep all questions of a story together, separately for Human and Auto.
The first two Human training stories and the public-test context used for this
revision have been inspected and must not later be claimed as untouched holdout
data. This CLI does not create those splits.

For a paper-like generation setting, pass the **same labeled evaluation file**
to every command, use new output directories, and explicitly match temperature,
model and reasoning token limit. For example, on the existing labeled file:

```powershell
python gpt_experiment.py --input Data/human_train.json --method cot --tasks FR --temperature 0.3 --max-output-tokens 4096 --output-dir outputs/cot_fr_v2_human_t03
python gpt_experiment.py --input Data/human_train.json --method pot --tasks FR --temperature 0.3 --max-output-tokens 4096 --max-paths 0 --max-hops 0 --output-dir outputs/pot_v2_fr_human_t03
python gpt_experiment.py --input Data/human_train.json --method pot-no-path --tasks FR --temperature 0.3 --max-output-tokens 4096 --graph-cache-dir outputs/pot_v2_fr_human_t03 --output-dir outputs/pot_v2_no_path_fr_human_t03
```

These full-file commands have **not been run**. The currently supplied Human
file has 149 FR questions. Full-file evaluation is still not a holdout, and
unlimited path counts can increase cost substantially. Replace `--input` with
your frozen evaluation file for a proper benchmark. Repeat independently for
Auto; report Human and Auto separately. Public-test files have no gold labels,
so their local accuracy cannot be computed.

Inspect extraction quality, query entity resolution, path-limit frequency,
fallback rate and extra-label errors before changing prompts or aggregation.
Tune only on development stories. Report repeated-run uncertainty (preferably
resampling by story), exact match/Jaccard, hit accuracy, coverage and token cost.
The current joint-evidence decision is a new variant, not the paper's
union-of-possible-answers behavior; compare it separately from historical v1.

## Recovery and artifacts

`responses.jsonl` checkpoints extraction and the `reason:joint` call
immediately, as well as the final record. Rerunning skips successful stages,
retries failed stages and does not re-bill cached successes. A failed joint
response is not converted to DK or silently accepted; the question
remains unanswered until resumed successfully. An interruption before a response
can be saved may still incur provider charges.

Final records include graph, edge-index paths, per-pair search diagnostics,
`limits_hit`, fallback, and one joint result in `path_results`. The legacy
`path_disagreement` and `path_disagreement_questions` fields are null because
independent path predictions are no longer produced. `metrics.json` includes ordinary
task metrics plus `pot_diagnostics`; missing predictions count as incorrect.
Usage sums stage events only once, across recorded attempts. Imported graph
events have zero new usage and are counted under `reused_extractions`, which
now also includes same-run per-story reuse, not only `--graph-cache-dir`
imports. `--extract-only` writes a different, smaller `metrics.json` shape
(`extracted_questions`, `unique_stories`, `real_extraction_calls`,
`reused_story_extractions`; no `by_task`/predictions file) since no answers
are produced.

Input, prompts, settings and selected task types are pinned in `config.json`.
Use a new output directory when changing them. Existing default CoT version-2
caches remain compatible; PoT uses format version 3 with `pot_version: 2`.
Old PoT extractions/outputs are incompatible; use a new output directory and
re-extract rather than mixing prompt or reasoning versions. The default PoT
directory is now `outputs/gpt41mini_pot_v2_human_train`.
Increasing `--max-questions` may
resume a run, but complete/freeze its source before starting cached ablations.

Offline checks: `python -m unittest discover -v`. The suite covers directed
evidence during undirected traversal, parallel edges, cycles, bounds, invalid
graphs, boundary contact, multi-pair budgets, all four tasks, joint reasoning,
prompt example schemas, fallback, leakage, resumability, old-cache rejection and
fixed-graph ablation accounting. These tests validate implementation behavior,
not the LLM's semantic accuracy.