# Local Data

Obtain the Vietnamese SPARTQA files from the VLSP task organizers through an
authorized channel and place them in this directory:

| File | Use |
| --- | --- |
| `human_train.json` | Labeled Human stories and questions |
| `auto_train.json` | Labeled Auto stories and questions |
| `human_public_test.json` | Unlabeled Human public test |
| `auto_public_test.json` | Unlabeled Auto public test |

These competition JSON files are intentionally excluded from Git. Do not assume
redistribution is permitted. No dataset download, copying of private data, or API
access is required for the unit tests, which construct synthetic examples.

Each JSON file contains `name` and `data`. Each data item has `story` and
`questions`; each question has `q_id`, `q_type`, `question`, `candidate_answers`,
and an `answer` list when labeled. Preserve the stories and their question lists.

YN uses a single string, FR uses relation indices, FB uses block identifiers, and
CO uses a single integer. Human train includes DK labels; do not discard them.
When creating a validation split, keep all questions of a story together.

`python make_splits.py --from-manifest` recreates the story-level train/dev files
in `Data/splits/` from the committed `manifest.json`. See the main readme.
