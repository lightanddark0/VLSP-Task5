"""Branch F inference: Qwen3-8B + LoRA adapter with vLLM (greedy, thinking off).

    python infer_f.py --adapter vispatialqa-f-qwen3-8b-auto --source F1 \\
        --job auto,Data/splits/auto_dev.json,outputs/pred/F1/auto_dev.jsonl \\
        --job auto,Data/auto_public_test.json,outputs/pred/F1/auto_test.jsonl

The adapter is a local directory or a Hub repo (``name`` or ``user/name``).
The model loads once for all jobs. Predictions are appended per chunk to the
JSONL files, so a rerun skips finished questions. An unparseable generation
gets the most frequent training answer of its type and ``fallback: true``.
Labeled inputs (dev) are scored with default post-processing and logged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spartqa import hub, tracking
from spartqa.data import read_json
from spartqa.inference import (common_arguments, fallback_answers, log_report, pending, print_report,
                               score_predictions, upload_predictions)
from spartqa.postprocess import PostprocessOptions
from spartqa.predictions import PredictionWriter, one_hot_scores, payload_of, read_predictions
from spartqa.prompting import build_messages, parse_answer, prompt_version, render_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Adapter directory or Hub repo id.")
    parser.add_argument("--model-name", help="Base model; defaults to the one recorded with the adapter.")
    parser.add_argument("--source", default="F", help="Source name stored in predictions, e.g. F1 or F2.")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/adapters")
                        if Path("/content").exists() else Path("outputs/adapters"))
    common_arguments(parser)
    args = parser.parse_args(argv)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_source = args.adapter if Path(args.adapter).exists() else hub.resolve_repo(args.adapter)
    adapter_dir, adapter_revision = hub.resolve_adapter(adapter_source, args.cache_dir)
    run_config = adapter_dir / "run_config.json"
    base = args.model_name or (json.loads(run_config.read_text())["base_model"] if run_config.exists()
                               else "Qwen/Qwen3-8B")
    lora_rank = json.loads((adapter_dir / "adapter_config.json").read_text()).get("r", 16)
    smoke = bool(args.limit)
    datasets = sorted({job.dataset for job in args.job})
    splits = sorted({job.split for job in args.job})
    name = args.run_name or tracking.run_name("F", "infer", "+".join(datasets), "+".join(splits), args.source)
    tracking.init_run("F", "infer", name, datasets + splits + (["smoke"] if smoke else []), {
        "method": "LoRA", "source": args.source, "base_model": base, "adapter": str(adapter_source),
        "temperature": 0.0, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs, "prompt_version": prompt_version("F"),
        "jobs": [vars(job) | {"input": str(job.input), "output": str(job.output)} for job in args.job]})
    tracking.log_hf_link(None if Path(args.adapter).exists() else adapter_source, adapter_revision, "adapter_in")

    llm = LLM(model=base, enable_lora=True, max_lora_rank=max(16, lora_rank), max_loras=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=True, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    lora = LoRARequest("f_adapter", 1, str(adapter_dir))
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for job in args.job:
        data = read_json(job.input)
        fallback = fallback_answers(args.splits_dir, job.dataset)
        todo, done = pending(data, job.output, args.limit)
        print(f"[{job.dataset}/{job.split}] {job.input}: {done} done, {len(todo)} to predict")
        writer = PredictionWriter(job.output)
        stats = {"questions": 0, "fallback": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            payloads = [payload_of(item, question) for _, item, question in chunk]
            prompts = [render_prompt(tokenizer, build_messages(payload, job.dataset)) for payload in payloads]
            began = time.time()
            outputs = llm.generate(prompts, sampling, lora_request=lora)
            stats["seconds"] += time.time() - began
            records = []
            for (key, _, question), payload, output in zip(chunk, payloads, outputs):
                text = output.outputs[0].text
                answer = parse_answer(text, payload)
                used_fallback = answer is None
                if used_fallback:
                    answer = list(fallback[question["q_type"]])
                    if question["q_type"] == "FB":
                        answer = [block for block in answer if block in question["candidate_answers"]]
                stats["fallback"] += used_fallback
                stats["prompt_tokens"] += len(output.prompt_token_ids)
                stats["output_tokens"] += len(output.outputs[0].token_ids)
                records.append({"key": key, "q_type": question["q_type"], "answer": answer,
                                "scores": one_hot_scores(question["q_type"], answer), "source": args.source,
                                "fallback": used_fallback, "raw": text[:200]})
            writer.write_many(records)
            stats["questions"] += len(records)
            print(f"  wrote {done + start + len(chunk)}/{done + len(todo)}", flush=True)
            upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")

        count = max(stats["questions"], 1)
        tag = f"infer/{job.dataset}/{job.split}"
        tracking.log({f"{tag}/questions": stats["questions"], f"{tag}/seconds": stats["seconds"],
                      f"{tag}/questions_per_second": stats["questions"] / max(stats["seconds"], 1e-9),
                      f"{tag}/output_tokens_per_second": stats["output_tokens"] / max(stats["seconds"], 1e-9),
                      f"{tag}/mean_prompt_tokens": stats["prompt_tokens"] / count,
                      f"{tag}/mean_output_tokens": stats["output_tokens"] / count,
                      f"{tag}/parse_error_rate": stats["fallback"] / count,
                      f"{tag}/gpu_hours": stats["seconds"] / 3600})
        print(f"  parse errors (fallback used): {stats['fallback']}/{stats['questions']}")
        revision = upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")
        if revision:
            tracking.log_hf_link(hub.resolve_repo(args.results_repo), revision,
                                 f"pred_{job.dataset}_{job.split}", "dataset")

        report = score_predictions(data, read_predictions(job.output), job.dataset, PostprocessOptions(), fallback)
        if report is not None:
            print_report(report, f"[{args.source}] {job.dataset}/{job.split} (n={report['questions']})")
            log_report(report, job.dataset, job.split)
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
