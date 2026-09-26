"""XLNet baseline for ViSPARTQA / SPaRTQA Vietnamese spatial QA.

The script trains one shared XLNet encoder with four task-specific heads:
YN and CO are single-label classification; FR and FB are multi-label
classification. It keeps the public-test JSON structure and fills the answer
field for submission-style predictions.

Example:
	python XLNER.py train --train-files Data/human_train.json Data/auto_train.json \
		--test-files Data/human_public_test.json Data/auto_public_test.json \
		--output-dir outputs/xlnet_large
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from spartqa.data import read_json, write_json as write_data_json
from spartqa.metrics import finalize_metrics, set_jaccard, update_metric_state

TASKS = ("YN", "CO", "FR", "FB")
TASK_TO_ID = {task: index for index, task in enumerate(TASKS)}
ID_TO_TASK = {index: task for task, index in TASK_TO_ID.items()}

YN_LABELS = ("Yes", "No", "DK")
CO_LABELS = (0, 1, 2, 3)
FR_LABELS = tuple(range(8))
FB_LABELS = ("A", "B", "C")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def write_json(data: dict[str, Any], path: str | Path) -> None:
	write_data_json(path, data)


def normalize_story(story: str | list[str]) -> str:
	if isinstance(story, list):
		return " ".join(part.strip() for part in story if part.strip())
	return story.strip()


def normalize_answer(answer: Any, q_type: str) -> list[Any]:
	if answer is None:
		return []
	if not isinstance(answer, list):
		answer = [answer]
	if q_type in {"FR", "CO"}:
		return [int(item) for item in answer]
	return answer


def format_input(story: str, question: str, candidate_answers: list[Any]) -> str:
	candidate_text = " | ".join(str(answer) for answer in candidate_answers)
	if candidate_text:
		return f"Câu chuyện: {story}\nCâu hỏi: {question}\nCác lựa chọn: {candidate_text}"
	return f"Câu chuyện: {story}\nCâu hỏi: {question}"


@dataclass(frozen=True)
class QAExample:
	dataset_name: str
	source_file: str
	story_index: int
	question_index: int
	q_id: int | str | None
	q_type: str
	text: str
	candidate_answers: list[Any]
	answer: list[Any] | None


def flatten_dataset(path: str | Path, require_answer: bool) -> list[QAExample]:
	path = Path(path)
	data = read_json(path)
	examples: list[QAExample] = []
	for story_index, item in enumerate(data.get("data", [])):
		story = normalize_story(item.get("story", ""))
		for question_index, question_item in enumerate(item.get("questions", [])):
			q_type = question_item.get("q_type")
			if q_type not in TASK_TO_ID:
				raise ValueError(f"Unsupported q_type={q_type!r} in {path}")
			has_answer = "answer" in question_item
			if require_answer and not has_answer:
				continue
			candidate_answers = list(question_item.get("candidate_answers", []))
			answer = normalize_answer(question_item.get("answer"), q_type) if has_answer else None
			examples.append(
				QAExample(
					dataset_name=str(data.get("name", path.stem)),
					source_file=str(path),
					story_index=story_index,
					question_index=question_index,
					q_id=question_item.get("q_id"),
					q_type=q_type,
					text=format_input(story, question_item.get("question", ""), candidate_answers),
					candidate_answers=candidate_answers,
					answer=answer,
				)
			)
	return examples


class SpatialQADataset(Dataset[dict[str, Any]]):
	def __init__(self, examples: list[QAExample], tokenizer: AutoTokenizer, max_length: int) -> None:
		self.examples = examples
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict[str, Any]:
		example = self.examples[index]
		encoded = self.tokenizer(
			example.text,
			truncation=True,
			max_length=self.max_length,
			padding=False,
		)
		item: dict[str, Any] = {
			"input_ids": encoded["input_ids"],
			"attention_mask": encoded["attention_mask"],
			"task_id": TASK_TO_ID[example.q_type],
			"example_index": index,
		}
		if example.answer is not None:
			item.update(make_label_tensors(example))
		return item


def make_label_tensors(example: QAExample) -> dict[str, torch.Tensor | int]:
	yn_label = -100
	co_label = -100
	fr_label = torch.zeros(len(FR_LABELS), dtype=torch.float)
	fr_mask = torch.zeros(len(FR_LABELS), dtype=torch.float)
	fb_label = torch.zeros(len(FB_LABELS), dtype=torch.float)
	fb_mask = torch.zeros(len(FB_LABELS), dtype=torch.float)

	answer = example.answer or []
	if example.q_type == "YN":
		yn_label = YN_LABELS.index(str(answer[0])) if answer else YN_LABELS.index("DK")
	elif example.q_type == "CO":
		co_label = int(answer[0]) if answer else 3
	elif example.q_type == "FR":
		fr_mask[:] = 1.0
		for label in answer:
			fr_label[int(label)] = 1.0
	elif example.q_type == "FB":
		candidate_set = set(str(candidate) for candidate in example.candidate_answers) or set(FB_LABELS)
		for index, label in enumerate(FB_LABELS):
			if label in candidate_set:
				fb_mask[index] = 1.0
		for label in answer:
			if label in FB_LABELS:
				fb_label[FB_LABELS.index(label)] = 1.0

	return {
		"yn_label": yn_label,
		"co_label": co_label,
		"fr_label": fr_label,
		"fr_mask": fr_mask,
		"fb_label": fb_label,
		"fb_mask": fb_mask,
	}


class SpatialQACollator:
	def __init__(self, tokenizer: AutoTokenizer) -> None:
		self.tokenizer = tokenizer

	def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
		token_features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features]
		batch = self.tokenizer.pad(token_features, padding=True, return_tensors="pt")
		batch["task_id"] = torch.tensor([item["task_id"] for item in features], dtype=torch.long)
		batch["example_index"] = torch.tensor([item["example_index"] for item in features], dtype=torch.long)
		if "yn_label" in features[0]:
			batch["yn_label"] = torch.tensor([item["yn_label"] for item in features], dtype=torch.long)
			batch["co_label"] = torch.tensor([item["co_label"] for item in features], dtype=torch.long)
			batch["fr_label"] = torch.stack([item["fr_label"] for item in features])
			batch["fr_mask"] = torch.stack([item["fr_mask"] for item in features])
			batch["fb_label"] = torch.stack([item["fb_label"] for item in features])
			batch["fb_mask"] = torch.stack([item["fb_mask"] for item in features])
		return batch


class XLNetSpatialQA(nn.Module):
	def __init__(self, model_name: str, dropout: float) -> None:
		super().__init__()
		config = AutoConfig.from_pretrained(model_name)
		self.encoder = AutoModel.from_pretrained(model_name, config=config)
		hidden_size = config.hidden_size
		self.dropout = nn.Dropout(dropout)
		self.yn_head = nn.Linear(hidden_size, len(YN_LABELS))
		self.co_head = nn.Linear(hidden_size, len(CO_LABELS))
		self.fr_head = nn.Linear(hidden_size, len(FR_LABELS))
		self.fb_head = nn.Linear(hidden_size, len(FB_LABELS))

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
		outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
		hidden = outputs.last_hidden_state
		mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
		pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
		pooled = self.dropout(pooled)
		return {
			"YN": self.yn_head(pooled),
			"CO": self.co_head(pooled),
			"FR": self.fr_head(pooled),
			"FB": self.fb_head(pooled),
		}


def compute_loss(logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
	losses: list[torch.Tensor] = []
	task_id = batch["task_id"]
	ce_loss = nn.CrossEntropyLoss()
	bce_loss = nn.BCEWithLogitsLoss(reduction="none")

	yn_mask = task_id == TASK_TO_ID["YN"]
	if yn_mask.any():
		losses.append(ce_loss(logits["YN"][yn_mask], batch["yn_label"][yn_mask]))

	co_mask = task_id == TASK_TO_ID["CO"]
	if co_mask.any():
		losses.append(ce_loss(logits["CO"][co_mask], batch["co_label"][co_mask]))

	fr_mask = task_id == TASK_TO_ID["FR"]
	if fr_mask.any():
		raw_loss = bce_loss(logits["FR"][fr_mask], batch["fr_label"][fr_mask])
		losses.append((raw_loss * batch["fr_mask"][fr_mask]).sum() / batch["fr_mask"][fr_mask].sum().clamp(min=1.0))

	fb_mask = task_id == TASK_TO_ID["FB"]
	if fb_mask.any():
		raw_loss = bce_loss(logits["FB"][fb_mask], batch["fb_label"][fb_mask])
		losses.append((raw_loss * batch["fb_mask"][fb_mask]).sum() / batch["fb_mask"][fb_mask].sum().clamp(min=1.0))

	if not losses:
		raise ValueError("Batch does not contain a supported task.")
	return torch.stack(losses).mean()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
	return {key: value.to(device) for key, value in batch.items()}


def split_indices(examples: list[QAExample], eval_ratio: float, seed: int) -> tuple[list[int], list[int]]:
	grouped: dict[str, list[int]] = {task: [] for task in TASKS}
	for index, example in enumerate(examples):
		grouped[example.q_type].append(index)

	rng = random.Random(seed)
	train_indices: list[int] = []
	eval_indices: list[int] = []
	for indices in grouped.values():
		rng.shuffle(indices)
		eval_size = max(1, int(round(len(indices) * eval_ratio))) if len(indices) > 1 else 0
		eval_indices.extend(indices[:eval_size])
		train_indices.extend(indices[eval_size:])
	rng.shuffle(train_indices)
	rng.shuffle(eval_indices)
	return train_indices, eval_indices


def prediction_from_logits(logits: dict[str, torch.Tensor], task: str, example: QAExample, threshold: float) -> list[Any]:
	if task == "YN":
		label_index = int(logits["YN"].argmax().item())
		return [YN_LABELS[label_index]]
	if task == "CO":
		return [int(logits["CO"].argmax().item())]
	if task == "FR":
		probs = torch.sigmoid(logits["FR"])
		selected = [index for index, value in enumerate(probs.tolist()) if value >= threshold]
		if not selected:
			selected = [int(probs.argmax().item())]
		return selected
	if task == "FB":
		probs = torch.sigmoid(logits["FB"])
		candidates = [str(candidate) for candidate in example.candidate_answers] or list(FB_LABELS)
		return [label for label in candidates if label in FB_LABELS and probs[FB_LABELS.index(label)].item() >= threshold]
	raise ValueError(f"Unsupported task {task!r}")


@torch.no_grad()
def predict_examples(
	model: XLNetSpatialQA,
	data_loader: DataLoader,
	examples: list[QAExample],
	device: torch.device,
	threshold: float,
) -> dict[int, list[Any]]:
	model.eval()
	predictions: dict[int, list[Any]] = {}
	for batch in tqdm(data_loader, desc="Predict", leave=False):
		batch = move_batch(batch, device)
		logits = model(batch["input_ids"], batch["attention_mask"])
		for row, example_index in enumerate(batch["example_index"].tolist()):
			example = examples[example_index]
			row_logits = {task: value[row].detach().cpu() for task, value in logits.items()}
			predictions[example_index] = prediction_from_logits(row_logits, example.q_type, example, threshold)
	return predictions


def evaluate(
	model: XLNetSpatialQA,
	data_loader: DataLoader,
	examples: list[QAExample],
	device: torch.device,
	threshold: float,
) -> dict[str, dict[str, float]]:
	predictions = predict_examples(model, data_loader, examples, device, threshold)
	state: dict[str, dict[str, float]] = {}
	for example_index, prediction in predictions.items():
		gold = examples[example_index].answer
		if gold is not None:
			update_metric_state(state, examples[example_index].q_type, prediction, gold)
	return finalize_metrics(state)


def apply_predictions_to_json(input_path: str | Path, predictions: dict[int, list[Any]], examples: list[QAExample]) -> dict[str, Any]:
	data = read_json(input_path)
	output = copy.deepcopy(data)
	for example_index, prediction in predictions.items():
		example = examples[example_index]
		output["data"][example.story_index]["questions"][example.question_index]["answer"] = prediction
	return output


def save_checkpoint(model: XLNetSpatialQA, tokenizer: AutoTokenizer, output_dir: str | Path, args: argparse.Namespace) -> None:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	model.encoder.save_pretrained(output_dir / "encoder")
	tokenizer.save_pretrained(output_dir / "tokenizer")
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"model_name": args.model_name,
			"dropout": args.dropout,
			"max_length": args.max_length,
			"threshold": args.threshold,
		},
		output_dir / "model.pt",
	)


def load_checkpoint(checkpoint_dir: str | Path, device: torch.device) -> tuple[XLNetSpatialQA, AutoTokenizer, dict[str, Any]]:
	checkpoint_dir = Path(checkpoint_dir)
	checkpoint = torch.load(checkpoint_dir / "model.pt", map_location=device)
	encoder_path = checkpoint_dir / "encoder"
	model_name = str(encoder_path) if encoder_path.exists() else checkpoint.get("model_name", "xlnet/xlnet-large-cased")
	tokenizer_path = checkpoint_dir / "tokenizer"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else model_name)
	model = XLNetSpatialQA(model_name, dropout=float(checkpoint.get("dropout", 0.1)))
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	return model, tokenizer, checkpoint


def train_command(args: argparse.Namespace) -> None:
	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	examples = [example for path in args.train_files for example in flatten_dataset(path, require_answer=True)]
	if not examples:
		raise ValueError("No training examples with answers were found.")

	dataset = SpatialQADataset(examples, tokenizer, args.max_length)
	train_indices, eval_indices = split_indices(examples, args.eval_ratio, args.seed)
	collator = SpatialQACollator(tokenizer)
	train_loader = DataLoader(
		Subset(dataset, train_indices),
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=collator,
		num_workers=args.num_workers,
	)
	eval_loader = DataLoader(
		dataset,
		batch_size=args.eval_batch_size,
		sampler=torch.utils.data.SubsetRandomSampler(eval_indices),
		collate_fn=collator,
		num_workers=args.num_workers,
	)

	model = XLNetSpatialQA(args.model_name, args.dropout).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	total_steps = max(len(train_loader) * args.epochs, 1)
	warmup_steps = int(total_steps * args.warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

	best_score = -1.0
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	for epoch in range(1, args.epochs + 1):
		model.train()
		total_loss = 0.0
		optimizer.zero_grad(set_to_none=True)
		progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
		for step, batch in enumerate(progress, start=1):
			batch = move_batch(batch, device)
			with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
				logits = model(batch["input_ids"], batch["attention_mask"])
				loss = compute_loss(logits, batch) / args.gradient_accumulation_steps
			scaler.scale(loss).backward()
			if step % args.gradient_accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)
			total_loss += loss.item() * args.gradient_accumulation_steps
			progress.set_postfix(loss=total_loss / step)
		if len(train_loader) % args.gradient_accumulation_steps != 0:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			optimizer.zero_grad(set_to_none=True)

		eval_metrics = evaluate(model, eval_loader, examples, device, args.threshold)
		write_json(eval_metrics, output_dir / f"eval_metrics_epoch_{epoch}.json")
		score = mean_primary_score(eval_metrics)
		print(f"Epoch {epoch}: eval primary score={score:.4f}")
		if score > best_score:
			best_score = score
			save_checkpoint(model, tokenizer, output_dir, args)
			write_json(eval_metrics, output_dir / "eval_metrics_best.json")

	if args.test_files:
		predict_command(args, model=model, tokenizer=tokenizer, device=device)


def mean_primary_score(metrics: dict[str, dict[str, float]]) -> float:
	scores: list[float] = []
	for task, values in metrics.items():
		if task in {"YN", "CO"}:
			scores.append(values.get("accuracy", 0.0))
		else:
			scores.append(values.get("exact_match", 0.0))
	return float(np.mean(scores)) if scores else 0.0


def predict_command(
	args: argparse.Namespace,
	model: XLNetSpatialQA | None = None,
	tokenizer: AutoTokenizer | None = None,
	device: torch.device | None = None,
) -> None:
	device = device or torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	checkpoint: dict[str, Any] = {}
	if model is None or tokenizer is None:
		if not args.checkpoint_dir:
			raise ValueError("--checkpoint-dir is required for predict when no in-memory model is provided.")
		model, tokenizer, checkpoint = load_checkpoint(args.checkpoint_dir, device)
	threshold = args.threshold if hasattr(args, "threshold") and args.threshold is not None else checkpoint.get("threshold", 0.5)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	all_metrics: dict[str, dict[str, dict[str, float]]] = {}
	for test_file in args.test_files:
		examples = flatten_dataset(test_file, require_answer=False)
		dataset = SpatialQADataset(examples, tokenizer, args.max_length)
		loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=SpatialQACollator(tokenizer))
		predictions = predict_examples(model, loader, examples, device, threshold)
		output = apply_predictions_to_json(test_file, predictions, examples)
		output_path = output_dir / f"{Path(test_file).stem}_predictions.json"
		write_json(output, output_path)

		if any(example.answer is not None for example in examples):
			metrics = evaluate(model, loader, examples, device, threshold)
			all_metrics[Path(test_file).stem] = metrics
			write_json(metrics, output_dir / f"{Path(test_file).stem}_metrics.json")
		print(f"Wrote {output_path}")
	if all_metrics:
		write_json(all_metrics, output_dir / "prediction_metrics.json")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="XLNet multi-task baseline for ViSPARTQA.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Train with an 80/20 split by default, evaluate, and optionally predict tests.")
	add_common_args(train_parser)
	train_parser.add_argument("--train-files", nargs="+", default=["Data/human_train.json", "Data/auto_train.json"])
	train_parser.add_argument("--test-files", nargs="*", default=["Data/human_public_test.json", "Data/auto_public_test.json"])
	train_parser.add_argument("--eval-ratio", type=float, default=0.2)
	train_parser.add_argument("--epochs", type=int, default=3)
	train_parser.add_argument("--batch-size", type=int, default=2)
	train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
	train_parser.add_argument("--learning-rate", type=float, default=2e-5)
	train_parser.add_argument("--weight-decay", type=float, default=0.01)
	train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
	train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
	train_parser.add_argument("--dropout", type=float, default=0.1)
	train_parser.add_argument("--fp16", action="store_true")
	train_parser.set_defaults(func=train_command)

	predict_parser = subparsers.add_parser("predict", help="Load a checkpoint and fill answers for public/private test JSON files.")
	add_common_args(predict_parser)
	predict_parser.add_argument("--checkpoint-dir", required=True)
	predict_parser.add_argument("--test-files", nargs="+", default=["Data/human_public_test.json", "Data/auto_public_test.json"])
	predict_parser.set_defaults(func=predict_command)
	return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--model-name", default="xlnet/xlnet-large-cased")
	parser.add_argument("--output-dir", default="outputs/xlnet_baseline")
	parser.add_argument("--max-length", type=int, default=384)
	parser.add_argument("--eval-batch-size", type=int, default=8)
	parser.add_argument("--threshold", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--cpu", action="store_true")


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
