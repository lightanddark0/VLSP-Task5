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
  pot.py                  All-task graph extraction, path search, and joint reasoning
  qwen_data.py            Story-disjoint split, prompt/answer contract (no torch dependency)
  qwen_model.py           QLoRA quantization + LoRA setup, restricted to the language decoder
  qwen_dataset.py         Tokenization, completion masking, collation, stage A/B sampling
  qwen_eval.py            Constrained-decoding generation, eight-cell metrics, regression gate
  qwen_train.py           Stage A/B training, evaluate, predict orchestration
gpt_experiment.py         API experiment CLI, cache, exports, and configuration
XLNER.py                  XLNet training/evaluation/prediction CLI
qwen_finetune.py          Local CLI for Qwen3-VL-8B-Instruct QLoRA (manifest/audit/train/evaluate/predict)
modal_app.py              Modal App/Volume wiring to run qwen_finetune.py on a GPU
test_gpt_experiment.py    Mocked API and resume tests
test_spartqa.py           Shared utilities and Git publication tests
Docs/
  spartqa_cot.txt          Versioned default prompt
  gpt_experiment.md        API experiment options and output details
  path_of_thoughts.md      PoT adaptation, controlled ablation, and pilot report
  qwen_qlora.md            Qwen3-VL-8B-Instruct QLoRA setup, Modal commands, and evaluation gate
  spartqa_pot_extract.txt  Structured spatial graph extraction prompt
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

## Qwen3-VL-8B-Instruct QLoRA

A text-only 4-bit QLoRA fine-tune targeting balanced quality across YN, FR, FB
and CO, designed to run on [Modal](https://modal.com) (one A100-80GB GPU) via
`modal_app.py`, or on any CUDA host via `qwen_finetune.py` directly. If you are
already running inside the target GPU environment (e.g. a notebook with this
repo and `requirements-qlora.txt` already in place), use `modal_runner.py`
instead — it calls the same commands in-process, with no `modal` CLI/client
needed. See [Docs/qwen_qlora.md](Docs/qwen_qlora.md) for the full command
sequence, the story-disjoint manifest, the eight-cell (Human/Auto x task)
evaluation, the non-regression gate against the quantized base model, and
assumptions that must be verified before spending GPU budget. Nothing in
that flow has been executed yet.

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

- The fixed story-level 80/20 validation files have **not yet been created**.
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
python -m compileall -q spartqa gpt_experiment.py XLNER.py test_gpt_experiment.py test_spartqa.py
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

