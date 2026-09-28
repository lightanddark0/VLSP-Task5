"""Branch L-CoT: Qwen3-32B-FP8 in thinking mode with self-consistency voting.

    python llm_cot.py --n 8 \\
        --job human,Data/splits/human_dev.json,outputs/pred/LCOT/human_dev.jsonl \\
        --job human,Data/human_public_test.json,outputs/pred/LCOT/human_test.jsonl

Each question is sampled ``n`` times (Qwen3 thinking settings: temperature 0.6,
top_p 0.95, top_k 20). The answer after the last "ĐÁP ÁN:" outside the
thinking block is parsed; truncated or unparseable samples are dropped.
YN/CO take the majority, FR/FB keep labels voted by at least half of the valid
samples; ``scores`` holds the vote shares. With no valid sample the question
is abstained (answer null). Solved examples of the same question type are
drawn from the train split only (never dev or test stories).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
import time
from pathlib import Path
from typing import Any

from spartqa import hub, tracking
from spartqa.data import read_json
from spartqa.inference import (common_arguments, configure_vllm_environment, fallback_answers, log_report,
                               overall_progress, pending, print_report, score_predictions, upload_predictions)
from spartqa.postprocess import PostprocessOptions
from spartqa.predictions import PredictionWriter, iter_questions, payload_of, read_predictions
from spartqa.prompting import (FR_AXES, build_cot_messages, build_fr_axis_messages, merge_axis_samples,
                               parse_axis_answer, parse_cot_answer, prompt_version, render_prompt)
from spartqa.voting import vote_samples


def example_pool(path: Path) -> dict[str, list[tuple[dict[str, Any], list[Any]]]]:
    pool: dict[str, list[tuple[dict[str, Any], list[Any]]]] = {}
    if not path.exists():
        return pool
    for _, item, question in iter_questions(read_json(path)):
        pool.setdefault(question["q_type"], []).append((payload_of(item, question), question["answer"]))
    return pool


def pick_examples(pool: dict[str, list[Any]], task: str, key: str, k: int, seed: int,
                  story: Any = None) -> list[Any]:
    """k solved examples of the same type, never from the question's own story (matters when the
    examples and the evaluated questions both come from Human train)."""
    candidates = [c for c in pool.get(task, []) if story is None or c[0]["story"] != story]
    return random.Random(f"{seed}:{key}").sample(candidates, min(k, len(candidates))) if k else []


def self_consistency_curve(data: dict[str, Any], records: dict[str, dict[str, Any]], dataset: str,
                           fallback: dict[str, list[Any]], n: int) -> dict[int, dict[str, Any]]:
    """Scores using only the first m samples, for m = 1, 2, 4, ... n (no new generation)."""
    curve = {}
    sizes = sorted({m for m in (1, 2, 4, 8, 16, 32) if m <= n} | {n})
    for m in sizes:
        subset = {}
        for key, record in records.items():
            answer, scores = vote_samples(record["q_type"], record.get("samples", [])[:m])
            subset[key] = {"answer": answer, "scores": scores}
        report = score_predictions(data, subset, dataset, PostprocessOptions(), fallback)
        if report is not None:
            curve[m] = report
    return curve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--fallback-model", default="Qwen/Qwen3-32B",
                        help="Loaded if --model-name fails to start (BF16, no FP8 kernels); '' disables.")
    parser.add_argument("--source", default="LCOT")
    parser.add_argument("--n", type=int, default=8, help="Samples per question.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--fewshot-k", type=int, default=2, help="Solved same-type examples in the prompt.")
    parser.add_argument("--save-raw", action="store_true", help="Also write every sampled text to *.raw.jsonl.")
    parser.add_argument("--q-types", default="YN,FR,FB,CO", help="Only these question types, e.g. FR.")
    parser.add_argument("--fr-mode", choices=("joint", "checklist"), default="joint",
                        help="checklist (E1): ask each FR axis (left/right, above/below, near/far, touch) "
                             "separately with its own samples, then merge.")
    common_arguments(parser)
    parser.set_defaults(chunk_size=32)
    args = parser.parse_args(argv)

    configure_vllm_environment()
    from vllm import LLM, SamplingParams

    smoke = bool(args.limit)
    datasets = sorted({job.dataset for job in args.job})
    splits = sorted({job.split for job in args.job})
    name = args.run_name or tracking.run_name("L-CoT", "infer", "+".join(datasets), "+".join(splits),
                                              f"n{args.n}")
    tracking.init_run("L-CoT", "infer", name, datasets + splits + (["smoke"] if smoke else []), {
        "method": "prompting", "model": args.model_name, "n": args.n, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs, "fewshot_k": args.fewshot_k,
        "prompt_version": prompt_version("COT"), "seed": args.seed, "q_types": args.q_types,
        "fr_mode": args.fr_mode,
        "jobs": [{"dataset": job.dataset, "input": str(job.input), "output": str(job.output)} for job in args.job]})

    def load(model: str) -> LLM:
        return LLM(model=model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                   max_num_seqs=args.max_num_seqs, enable_prefix_caching=True, seed=args.seed)

    try:
        llm = load(args.model_name)
    except RuntimeError as error:
        # FP8 kernels may need a JIT compiler newer than the machine has (e.g. SM 12.x GPUs on Colab).
        if not args.fallback_model or args.fallback_model == args.model_name:
            raise
        print(f"Could not start {args.model_name} ({error}); retrying with {args.fallback_model}.", flush=True)
        tracking.update_config({"model": args.fallback_model, "model_fallback_from": args.model_name})
        llm = load(args.fallback_model)
    tokenizer = llm.get_tokenizer()
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    sampling = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              max_tokens=args.max_tokens, seed=args.seed)

    for job in args.job:
        data = read_json(job.input)
        fallback = fallback_answers(args.splits_dir, job.dataset)
        pool = example_pool(args.splits_dir / f"{job.dataset}_train.json")
        todo, done = pending(data, job.output, args.limit)
        wanted = {t.strip().upper() for t in args.q_types.split(",")}
        todo = [entry for entry in todo if entry[2]["q_type"] in wanted]
        print(f"[{job.dataset}/{job.split}] {job.input}: {done} done, {len(todo)} to predict")
        writer = PredictionWriter(job.output)
        raw_writer = PredictionWriter(job.output.with_suffix(".raw.jsonl")) if args.save_raw else None
        stats = {"questions": 0, "abstain": 0, "samples": 0, "invalid_samples": 0, "prompt_tokens": 0,
                 "output_tokens": 0, "seconds": 0.0, "agreement": 0.0, "yn_co": 0}
        think_lengths: list[int] = []
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            payloads = [payload_of(item, question) for _, item, question in chunk]
            # One prompt per question, or one per axis for FR in checklist mode.
            units = []   # (question index, axis or None, prompt)
            for index, ((key, _, question), payload) in enumerate(zip(chunk, payloads)):
                task = payload["q_type"]
                examples = pick_examples(pool, task, key, args.fewshot_k, args.seed, payload["story"])
                if task == "FR" and args.fr_mode == "checklist":
                    for axis in FR_AXES:
                        units.append((index, axis, render_prompt(
                            tokenizer, build_fr_axis_messages(payload, job.dataset, axis, examples), enable_thinking=True)))
                else:
                    units.append((index, None, render_prompt(
                        tokenizer, build_cot_messages(payload, job.dataset, examples), enable_thinking=True)))
            began = time.time()
            overall_progress(f"{args.source} {job.dataset}/{job.split}", start // args.chunk_size + 1,
                             -(-len(todo) // args.chunk_size), done + start, done + len(todo), stats["seconds"],
                             start)
            outputs = llm.generate([prompt for _, _, prompt in units], sampling, use_tqdm=True)
            stats["seconds"] += time.time() - began
            by_question: dict[int, list[tuple[str | None, Any]]] = {}
            for (index, axis, _), output in zip(units, outputs):
                by_question.setdefault(index, []).append((axis, output))
            records, raws = [], []
            for index, ((key, _, question), payload) in enumerate(zip(chunk, payloads)):
                parts = by_question[index]
                if parts[0][0] is None:
                    output = parts[0][1]
                    samples = [None if sample.finish_reason == "length" else parse_cot_answer(sample.text, payload)
                               for sample in output.outputs]
                    invalid = sum(sample is None for sample in samples)
                else:
                    per_axis = {axis: [None if sample.finish_reason == "length" else parse_axis_answer(sample.text, axis)
                                       for sample in output.outputs] for axis, output in parts}
                    samples = merge_axis_samples(per_axis)
                    invalid = sum(value is None for values in per_axis.values() for value in values)
                answer, scores = vote_samples(question["q_type"], samples)
                for _, output in parts:
                    stats["prompt_tokens"] += len(output.prompt_token_ids)
                    for sample in output.outputs:
                        ids = list(sample.token_ids)
                        think_lengths.append(ids.index(think_end) if think_end in ids else len(ids))
                        stats["output_tokens"] += len(ids)
                        stats["samples"] += 1
                stats["invalid_samples"] += invalid
                stats["abstain"] += answer is None
                if scores and question["q_type"] in {"YN", "CO"}:
                    stats["agreement"] += max(scores.values())
                    stats["yn_co"] += 1
                record = {"key": key, "q_type": question["q_type"], "answer": answer, "scores": scores,
                          "source": args.source, "samples": samples}
                if parts[0][0] is not None:
                    record["fr_mode"] = "checklist"
                    record["axes"] = {axis: Counter(tuple(v) for v in values if v is not None).most_common(1)[0][0]
                                      if any(v is not None for v in values) else None
                                      for axis, values in per_axis.items()}
                records.append(record)
                if raw_writer is not None:
                    raws.append({"key": key, "texts": {str(axis): [sample.text for sample in output.outputs]
                                                       for axis, output in parts}})
            writer.write_many(records)
            if raw_writer is not None:
                raw_writer.write_many(raws)
            stats["questions"] += len(records)
            upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")
        overall_progress(f"{args.source} {job.dataset}/{job.split}", None, None, done + len(todo),
                         done + len(todo), stats["seconds"], len(todo))

        count = max(stats["questions"], 1)
        think_lengths.sort()
        tag = f"infer/{job.dataset}/{job.split}"
        tracking.log({
            f"{tag}/questions": stats["questions"], f"{tag}/seconds": stats["seconds"],
            f"{tag}/seconds_per_question": stats["seconds"] / count,
            f"{tag}/output_tokens_per_second": stats["output_tokens"] / max(stats["seconds"], 1e-9),
            f"{tag}/mean_prompt_tokens": stats["prompt_tokens"] / count,
            f"{tag}/mean_output_tokens": stats["output_tokens"] / max(stats["samples"], 1),
            f"{tag}/think_tokens_p50": think_lengths[len(think_lengths) // 2] if think_lengths else None,
            f"{tag}/think_tokens_p95": think_lengths[int(len(think_lengths) * 0.95)] if think_lengths else None,
            f"{tag}/invalid_sample_rate": stats["invalid_samples"] / max(stats["samples"], 1),
            f"{tag}/abstain_rate": stats["abstain"] / count,
            f"{tag}/mean_vote_share_yn_co": stats["agreement"] / max(stats["yn_co"], 1),
            f"{tag}/gpu_hours": stats["seconds"] / 3600})
        print(f"  invalid samples {stats['invalid_samples']}/{stats['samples']}, abstained {stats['abstain']}")
        if args.save_raw:
            upload_predictions(job.output.with_suffix(".raw.jsonl"), args.results_repo,
                               f"{args.results_prefix}/{args.source}")
        revision = upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")
        if revision:
            tracking.log_hf_link(hub.resolve_repo(args.results_repo), revision,
                                 f"pred_{job.dataset}_{job.split}", "dataset")

        records = read_predictions(job.output)
        report = score_predictions(data, records, job.dataset, PostprocessOptions(), fallback)
        if report is not None:
            print_report(report, f"[{args.source}] {job.dataset}/{job.split} (n={report['questions']})")
            log_report(report, job.dataset, job.split)
            for m, curve_report in self_consistency_curve(data, records, job.dataset, fallback, args.n).items():
                tracking.log_eval(curve_report, f"sc/{job.dataset}/{job.split}/n{m}")
                print(f"  self-consistency n={m}: mean primary {curve_report['primary_macro_unofficial']:.4f}")
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
