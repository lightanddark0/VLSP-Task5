# GPT-4.1-mini spatial QA experiment

This is zero-shot API evaluation on Human train, not fine-tuning and not a
held-out benchmark result. Gold answers are used locally for scoring only.
The validation split task is separate; this script does not alter any data splits.

This describes the default `--method cot` run. For the all-task
`--method pot` and `--method pot-no-path` experiments, graph-stage caching,
controlled ablations, and observed results, see
[path_of_thoughts.md](path_of_thoughts.md). All methods default to YN, FR, FB and
CO. Use `--tasks FR` on both methods for a matched FR-only comparison before
applying `--max-questions`.

## Run

Install only the API dependency; PyTorch and a GPU are not needed:

```powershell
python -m pip install -r requirements-api.txt
python gpt_experiment.py --dry-run
```

The script reads `.env` in the project root automatically (or `--env-file PATH`).
Existing shell variables take precedence over the file. `OPENAI_API_KEY` is
required. `OPENAI_BASE_URL` may point to an OpenAI-compatible gateway; if unset,
the official OpenAI endpoint is used. Never paste keys into chat or commit them.
The project `.gitignore` excludes environment files, but does not untrack files
that were already committed. Rotate any key that has been exposed.

Alternatively, configure the key privately in the same terminal. This PowerShell
input hides the key while you type:

```powershell
$secret = Read-Host "OPENAI_API_KEY" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
Remove-Variable secret
```

Start with ten questions, then resume to evaluate the full file (613 questions
in the current dataset). This sends the Vietnamese story and question to the
configured endpoint and incurs usage charges. Successful cached calls are skipped.
Use the same terminal for both commands so the environment variable is available.

```powershell
python gpt_experiment.py --max-questions 10
python gpt_experiment.py
```

Model selection is `--model`, then `OPENAI_MODEL`, then `BASE_MODEL_ID`, then the
pinned snapshot `gpt-4.1-mini-2025-04-14`. `OPENAI_MODEL_2`, `DIVERSITY_MODEL_ID`,
and `AUGMENT_MODEL_ID` are not used by this experiment. Defaults are temperature
0 and at most 1200 output tokens per request. One question is sent
per request. The SDK retries transient failures at most twice; persistent API
errors stop the run to avoid repeatedly spending on a broken configuration.
No exact USD estimate is reported: consult your provider's billing for current
pricing and cached-token discounts. API error records include the HTTP status
and available message/code/parameter fields, with API keys and tokens redacted.

## Gateway Model IDs

An OpenAI-compatible gateway may require a deployment ID such as
`azure-4.1-mini` rather than the public alias `gpt-4.1-mini`. HTTP 400 with
"Invalid model name" means you should check the IDs available to your key,
often through `/v1/models`. Set `OPENAI_MODEL` or pass `--model` explicitly.
Leave model settings for unrelated pipelines unchanged.

For a gateway that advertises `azure-4.1-mini`, these commands start or resume
an experiment in its own output directory:

```powershell
python gpt_experiment.py --model azure-4.1-mini --max-questions 10 --output-dir outputs/gpt41mini_cot_azure
python gpt_experiment.py --model azure-4.1-mini --output-dir outputs/gpt41mini_cot_azure
```

For another gateway, use its model listing to find the permitted ID. Do not
automatically substitute a different model when a deployment is unavailable.

## Compare Configurations

Edit a copy of `Docs/spartqa_cot.txt` and use a new output directory:

```powershell
python gpt_experiment.py --prompt-file Docs/spartqa_cot.txt --temperature 0.2 --output-dir outputs/gpt41mini_cot_t02
```

Other options: `--input`, `--env-file`, `--model`, `--max-output-tokens`, `--max-questions`.
Use `--model gpt-4.1-mini` for the rolling alias instead of the pinned snapshot.
Endpoint, model, prompt, generation settings, and input hash are recorded; keys
are not recorded. Reusing an output directory with different settings, endpoint,
or changed data is rejected. Older configurations without endpoint metadata need
a new output directory. Resume
skips successful responses and retries previously unsuccessful questions.

## Outputs

Default directory: `outputs/gpt41mini_cot_human_train/`.

- `config.json`: endpoint, exact prompt, model settings, and input SHA-256.
- `responses.jsonl`: append-only per-call explanations, answers, token usage,
  response model/ID, or sanitized API error details. Never contains the API key.
- `human_train_predictions.json`: original JSON structure with gold answers
  removed and successful predictions inserted. Unanswered questions have no
  answer field, so a partial run must not be treated as a complete submission.
- `metrics.json`: YN/CO accuracy, FR/FB exact match and mean per-question
  Jaccard, supplementary FR hit accuracy, counts, coverage, run status, and
  recorded token usage. Hit accuracy means at least one predicted label is gold;
  it does not penalize extra labels and must not replace exact match/Jaccard.

For the selected questions, missing/invalid predictions count as incorrect and
receive Jaccard 0. Valid empty FB prediction and gold sets receive Jaccard 1.
A `--max-questions` run reports metrics only for that prefix, not the full file.
Usage includes recorded failed-format attempts; requests without a returned
usage object cannot be included. Actual billing can differ, especially if a
connection is lost after the server finishes a request.

The prompt supports all four tasks, partial spatial information, quantifiers,
and DK. It does not force unit-distance grids. Human train contains DK labels
despite the binary-YN description in the task README; these labels are retained.
Do not tune a prompt on this entire file and later describe a subset of the same
file as an untouched holdout. The existing XLNet run is not directly comparable
because its evaluation subset and training exposure differ.

## Local Tests

```powershell
python -m unittest discover -v
```

These use a mocked API and cost nothing. They validate payload isolation, answer
types, failed responses, metrics, JSON preservation, resume behavior, and Git
publication rules. Reusable data/answer logic lives in `spartqa/data.py`, scoring
in `spartqa/metrics.py`, and API transport in `spartqa/api.py`. A real
API smoke test is still required to verify credentials and service compatibility.