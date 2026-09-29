"""Local CLI for Qwen3-VL-8B-Instruct QLoRA SFT on ViSPARTQA (YN/FR/FB/CO).

Subcommands: manifest, audit, inspect-modules, train-stage-a, train-stage-b,
evaluate, predict (single-agent, end-to-end); distill-graphs, train-agent-extraction,
train-agent-reasoning, evaluate-pipeline, predict-pipeline (two-agent: an extraction
agent producing structured facts, then a reasoning agent answering with them).
Heavy ML imports are deferred into each command so --help and syntax checks stay
usable without torch/transformers/peft/bitsandbytes installed, matching
gpt_experiment.py's lazy-import convention.

See Docs/qwen_qlora.md for the full command sequence, budget notes and
unverified assumptions to check before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spartqa.data import write_json
from spartqa.qwen_data import build_manifest

ROOT = Path(__file__).resolve().parent


def command_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest({"human": args.human, "auto": args.auto}, seed=args.seed, val_ratio=args.val_ratio)
    write_json(args.output, manifest)
    print(json.dumps({
        "warnings": manifest["warnings"],
        "label_conflicts": len(manifest["label_conflicts"]),
        "counts": manifest["counts"],
    }, indent=2, ensure_ascii=False))
    return 1 if manifest["warnings"] else 0


def command_audit(args: argparse.Namespace) -> int:
    from spartqa.data import read_json
    from spartqa.qwen_data import label_distribution, load_source_examples
    from spartqa.qwen_dataset import encode_example
    from spartqa.qwen_model import load_processor

    processor = load_processor(args.model_id)
    examples = []
    for source, path in (("human", args.human), ("auto", args.auto)):
        examples.extend(load_source_examples(source, read_json(path)))
    labeled = [example for example in examples if example["gold"] is not None]
    lengths = sorted(len(encode_example(processor, example)["input_ids"]) for example in labeled)
    report = {
        "label_distribution": label_distribution(labeled),
        "token_length": {
            "count": len(lengths),
            "max": lengths[-1] if lengths else 0,
            "p50": lengths[len(lengths) // 2] if lengths else 0,
            "p99": lengths[int(len(lengths) * 0.99)] if lengths else 0,
        },
    }
    write_json(args.output, report)
    print(json.dumps(report["token_length"], indent=2))
    return 0


def command_inspect_modules(args: argparse.Namespace) -> int:
    from spartqa.qwen_model import load_base_model, resolve_language_lora_target_modules

    model = load_base_model(args.model_id, args.revision)
    targets = resolve_language_lora_target_modules(model)
    all_names = [name for name, _module in model.named_modules()]
    print(json.dumps({
        "resolved_target_modules_count": len(targets),
        "resolved_target_modules_sample": targets[:20],
        "total_module_count": len(all_names),
    }, indent=2))
    return 0


def command_train_stage_a(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_stage_a

    run_stage_a(args)
    return 0


def command_train_stage_b(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_stage_b

    run_stage_b(args)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_evaluate

    run_evaluate(args)
    return 0


def command_predict(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_predict

    run_predict(args)
    return 0


def command_distill_graphs(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_distill_graphs

    run_distill_graphs(args)
    return 0


def command_train_agent_extraction(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_train_agent_extraction

    run_train_agent_extraction(args)
    return 0


def command_train_agent_reasoning(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_train_agent_reasoning

    run_train_agent_reasoning(args)
    return 0


def command_evaluate_pipeline(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_evaluate_pipeline

    run_evaluate_pipeline(args)
    return 0


def command_predict_pipeline(args: argparse.Namespace) -> int:
    from spartqa.qwen_train import run_predict_pipeline

    run_predict_pipeline(args)
    return 0


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--human", type=Path, default=ROOT / "Data/human_train.json")
    parser.add_argument("--auto", type=Path, default=ROOT / "Data/auto_train.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Build the story-disjoint train/val split manifest.")
    manifest_parser.add_argument("--human", type=Path, default=ROOT / "Data/human_train.json")
    manifest_parser.add_argument("--auto", type=Path, default=ROOT / "Data/auto_train.json")
    manifest_parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen_qlora/manifest.json")
    manifest_parser.add_argument("--seed", type=int, default=42)
    manifest_parser.add_argument("--val-ratio", type=float, default=0.2)
    manifest_parser.set_defaults(handler=command_manifest)

    audit_parser = subparsers.add_parser("audit", help="Report label distribution and chat-templated token lengths.")
    audit_parser.add_argument("--human", type=Path, default=ROOT / "Data/human_train.json")
    audit_parser.add_argument("--auto", type=Path, default=ROOT / "Data/auto_train.json")
    audit_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    audit_parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen_qlora/audit.json")
    audit_parser.set_defaults(handler=command_audit)

    inspect_parser = subparsers.add_parser("inspect-modules", help="Print resolved LoRA target modules before training.")
    inspect_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    inspect_parser.add_argument("--revision")
    inspect_parser.set_defaults(handler=command_inspect_modules)

    stage_a_parser = subparsers.add_parser("train-stage-a", help="Auto spatial foundation stage.")
    _add_common_train_args(stage_a_parser)
    stage_a_parser.add_argument("--auto-subset-size", type=int, default=30000)
    stage_a_parser.add_argument("--learning-rate", type=float, default=1e-4)
    stage_a_parser.set_defaults(handler=command_train_stage_a)

    stage_b_parser = subparsers.add_parser("train-stage-b", help="Human adaptation stage with Auto replay, same adapter.")
    _add_common_train_args(stage_b_parser)
    stage_b_parser.add_argument("--init-adapter", type=Path, required=True)
    stage_b_parser.add_argument("--human-passes", type=float, default=3.0)
    stage_b_parser.add_argument("--learning-rate", type=float, default=3e-5)
    stage_b_parser.set_defaults(handler=command_train_stage_b)

    evaluate_parser = subparsers.add_parser("evaluate", help="Constrained-decoding evaluation on the manifest validation split.")
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--human", type=Path, default=ROOT / "Data/human_train.json")
    evaluate_parser.add_argument("--auto", type=Path, default=ROOT / "Data/auto_train.json")
    evaluate_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    evaluate_parser.add_argument("--adapter", type=Path, help="Omit to evaluate the quantized base model only (baseline).")
    evaluate_parser.add_argument("--baseline-metrics", type=Path, help="Prior metrics.json to apply the regression gate against.")
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=64)
    evaluate_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_parser.add_argument("--max-questions", type=int, help="Cap per source (human/auto) for a timing pilot; omit for the full validation split.")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(handler=command_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Fill answers for an unlabeled public-test JSON file.")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    predict_parser.add_argument("--adapter", type=Path, required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=64)
    predict_parser.add_argument("--batch-size", type=int, default=4)
    predict_parser.add_argument("--max-questions", type=int, help="Timing pilot only: produces an incomplete file, not a valid submission.")
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.set_defaults(handler=command_predict)

    distill_parser = subparsers.add_parser(
        "distill-graphs",
        help="Convert a completed `gpt_experiment.py --method pot` run's responses.jsonl into a compact graphs.json for agent training.",
    )
    distill_parser.add_argument("--pot-log", type=Path, required=True, help="responses.jsonl from a gpt_experiment.py --method pot/pot-no-path run.")
    distill_parser.add_argument("--source", choices=("human", "auto"), required=True, help="Which manifest source this pot-log's --input matched; re-keys graphs to match qwen_data.py's source-qualified keys.")
    distill_parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen_qlora/graphs.json")
    distill_parser.set_defaults(handler=command_distill_graphs)

    agent_extraction_parser = subparsers.add_parser(
        "train-agent-extraction", help="Agent 1: story+question -> structured facts JSON, distilled from GPT.",
    )
    _add_common_train_args(agent_extraction_parser)
    agent_extraction_parser.add_argument("--graphs", type=Path, required=True, help="graphs.json produced by distill-graphs.")
    agent_extraction_parser.add_argument("--max-examples", type=int, help="Cap the number of TRAIN examples with a distilled graph.")
    agent_extraction_parser.add_argument("--learning-rate", type=float, default=1e-4)
    agent_extraction_parser.set_defaults(handler=command_train_agent_extraction)

    agent_reasoning_parser = subparsers.add_parser(
        "train-agent-reasoning", help="Agent 2: story+question[+facts] -> answer JSON.",
    )
    _add_common_train_args(agent_reasoning_parser)
    agent_reasoning_parser.add_argument(
        "--graphs", type=Path,
        help="graphs.json produced by distill-graphs; TRAIN examples without an entry train the no-facts fallback.",
    )
    agent_reasoning_parser.add_argument("--human-passes", type=float, default=3.0)
    agent_reasoning_parser.add_argument("--learning-rate", type=float, default=3e-5)
    agent_reasoning_parser.set_defaults(handler=command_train_agent_reasoning)

    evaluate_pipeline_parser = subparsers.add_parser(
        "evaluate-pipeline", help="Two-agent (extraction -> reasoning) constrained-decoding evaluation on the manifest validation split.",
    )
    evaluate_pipeline_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_pipeline_parser.add_argument("--human", type=Path, default=ROOT / "Data/human_train.json")
    evaluate_pipeline_parser.add_argument("--auto", type=Path, default=ROOT / "Data/auto_train.json")
    evaluate_pipeline_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    evaluate_pipeline_parser.add_argument("--extraction-adapter", type=Path, required=True)
    evaluate_pipeline_parser.add_argument("--reasoning-adapter", type=Path, required=True)
    evaluate_pipeline_parser.add_argument("--baseline-metrics", type=Path, help="Prior metrics.json to apply the regression gate against.")
    evaluate_pipeline_parser.add_argument("--graph-max-new-tokens", type=int, default=512)
    evaluate_pipeline_parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    evaluate_pipeline_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_pipeline_parser.add_argument("--max-questions", type=int, help="Cap per source (human/auto) for a timing pilot; omit for the full validation split.")
    evaluate_pipeline_parser.add_argument("--output", type=Path, required=True)
    evaluate_pipeline_parser.set_defaults(handler=command_evaluate_pipeline)

    predict_pipeline_parser = subparsers.add_parser(
        "predict-pipeline", help="Two-agent (extraction -> reasoning) prediction for an unlabeled public-test JSON file.",
    )
    predict_pipeline_parser.add_argument("--input", type=Path, required=True)
    predict_pipeline_parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    predict_pipeline_parser.add_argument("--extraction-adapter", type=Path, required=True)
    predict_pipeline_parser.add_argument("--reasoning-adapter", type=Path, required=True)
    predict_pipeline_parser.add_argument("--graph-max-new-tokens", type=int, default=512)
    predict_pipeline_parser.add_argument("--answer-max-new-tokens", type=int, default=64)
    predict_pipeline_parser.add_argument("--batch-size", type=int, default=4)
    predict_pipeline_parser.add_argument("--max-questions", type=int, help="Timing pilot only: produces an incomplete file, not a valid submission.")
    predict_pipeline_parser.add_argument("--output", type=Path, required=True)
    predict_pipeline_parser.set_defaults(handler=command_predict_pipeline)

    return parser


def main_with_argv(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def main() -> int:
    return main_with_argv()


if __name__ == "__main__":
    raise SystemExit(main())
