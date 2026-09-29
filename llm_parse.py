"""Branch LP (E3): Qwen3-32B transcribes stories and questions into JSON; the S reasoner answers.

    python llm_parse.py --n-world 4 --n-form 4 \\
        --job human,Data/splits/human_train.json,outputs/predictions/LP/human_train.jsonl \\
        --job human,Data/splits/human_dev.json,outputs/predictions/LP/human_dev.jsonl \\
        --job human,Data/human_public_test.json,outputs/predictions/LP/human_test.jsonl

For each story, ``n-world`` worlds are sampled (objects, stated facts, edge
contacts); for each question, ``n-form`` logical forms. Every usable
(world, form) pair is answered by the reasoner (spartqa.symbolic.structured)
and the answers are voted; with no usable pair the question is abstained.
All sampled JSON is saved in <output>.parses.jsonl, so solve_parsed.py can
re-answer under other conventions without the GPU. Stories used as prompt
examples (spartqa/symbolic/lp_annotations.json) are skipped.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from solve_parsed import answer_question, parse_setting, read_parses
from spartqa import hub, tracking
from spartqa.data import read_json
from spartqa.inference import (common_arguments, configure_vllm_environment, fallback_answers, log_report,
                               overall_progress, pending, print_report, score_predictions, upload_predictions)
from spartqa.lp_prompting import (extract_json, load_examples, prompt_version, question_messages, story_sha,
                                  world_messages)
from spartqa.postprocess import PostprocessOptions
from spartqa.predictions import PredictionWriter, payload_of, read_predictions
from spartqa.prompting import render_prompt
from spartqa.symbolic.structured import LPConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--fallback-model", default="Qwen/Qwen3-32B",
                        help="Loaded if --model-name fails to start (BF16, no FP8 kernels); '' disables.")
    parser.add_argument("--source", default="LP")
    parser.add_argument("--n-world", type=int, default=4, help="Sampled worlds per story.")
    parser.add_argument("--n-form", type=int, default=4, help="Sampled logical forms per question.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--no-think-forms", action="store_true", help="Parse questions without thinking (faster).")
    parser.add_argument("--examples", type=int, default=1, help="Annotated stories shown as worked examples.")
    parser.add_argument("--examples-data", type=Path, help="Data file of the annotations (default: their source).")
    parser.add_argument("--set", type=parse_setting, action="append", default=[], metavar="NAME=VALUE",
                        help="Answering convention (see solve_parsed.py), e.g. cross_block_far=true.")
    common_arguments(parser)
    parser.set_defaults(chunk_size=8)
    args = parser.parse_args(argv)
    config = replace(LPConfig(), **dict(args.set))
    examples = load_examples(data_path=args.examples_data, count=args.examples)
    example_shas = {example["sha"] for example in examples}

    configure_vllm_environment()
    from vllm import LLM, SamplingParams

    datasets = sorted({job.dataset for job in args.job})
    splits = sorted({job.split for job in args.job})
    name = args.run_name or tracking.run_name("LP", "infer", "+".join(datasets), "+".join(splits),
                                              f"w{args.n_world}f{args.n_form}")
    tracking.init_run("LP", "infer", name, datasets + splits + (["smoke"] if args.limit else []), {
        "method": "llm-parse + symbolic reasoner", "model": args.model_name, "n_world": args.n_world,
        "n_form": args.n_form, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "think_forms": not args.no_think_forms, "examples": args.examples, "prompt_version": prompt_version(),
        "lp_config": asdict(config), "seed": args.seed,
        "jobs": [{"dataset": job.dataset, "input": str(job.input), "output": str(job.output)} for job in args.job]})

    def load(model: str) -> LLM:
        return LLM(model=model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                   max_num_seqs=args.max_num_seqs, enable_prefix_caching=True, seed=args.seed)

    try:
        llm = load(args.model_name)
    except RuntimeError as error:
        if not args.fallback_model or args.fallback_model == args.model_name:
            raise
        print(f"Could not start {args.model_name} ({error}); retrying with {args.fallback_model}.", flush=True)
        tracking.update_config({"model": args.fallback_model, "model_fallback_from": args.model_name})
        llm = load(args.fallback_model)
    tokenizer = llm.get_tokenizer()
    world_sampling = SamplingParams(n=args.n_world, temperature=args.temperature, top_p=args.top_p,
                                    top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed)
    form_sampling = SamplingParams(n=args.n_form, temperature=args.temperature, top_p=args.top_p,
                                   top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed)

    for job in args.job:
        data = read_json(job.input)
        fallback = fallback_answers(args.splits_dir, job.dataset)
        parses_path = job.output.with_suffix(".parses.jsonl")
        worlds, forms = read_parses(parses_path)
        todo, done = pending(data, job.output, args.limit)
        skipped = [entry for entry in todo if story_sha(entry[1]["story"]) in example_shas]
        todo = [entry for entry in todo if story_sha(entry[1]["story"]) not in example_shas]
        by_story: dict[int, list] = defaultdict(list)
        for entry in todo:
            by_story[int(entry[0].split("_")[0])].append(entry)
        stories = sorted(by_story)
        print(f"[{job.dataset}/{job.split}] {job.input}: {done} done, {len(todo)} to predict in {len(stories)} "
              f"stories ({len(skipped)} questions of prompt-example stories skipped)")
        writer, parse_writer = PredictionWriter(job.output), PredictionWriter(parses_path)
        stats = {"questions": 0, "abstain": 0, "worlds": 0, "bad_worlds": 0, "forms": 0, "bad_forms": 0,
                 "votes": 0, "seconds": 0.0}
        for start in range(0, len(stories), args.chunk_size):
            chunk = stories[start:start + args.chunk_size]
            units, params = [], []      # ("world", story) or ("form", key)
            for story in chunk:
                item = by_story[story][0][1]
                if story not in worlds:
                    units.append(("world", story))
                    params.append((render_prompt(tokenizer, world_messages(item["story"], examples),
                                                 enable_thinking=True), world_sampling))
                for key, item, question in by_story[story]:
                    if key not in forms:
                        units.append(("form", key))
                        params.append((render_prompt(tokenizer, question_messages(payload_of(item, question), examples),
                                                     enable_thinking=not args.no_think_forms), form_sampling))
            began = time.time()
            overall_progress(f"{args.source} {job.dataset}/{job.split}", start // args.chunk_size + 1,
                             -(-len(stories) // args.chunk_size), done + stats["questions"], done + len(todo),
                             stats["seconds"], stats["questions"])
            outputs = llm.generate([prompt for prompt, _ in params], [sampling for _, sampling in params],
                                   use_tqdm=True) if params else []
            stats["seconds"] += time.time() - began
            parse_records = []
            for (kind, ident), output in zip(units, outputs):
                values = [None if sample.finish_reason == "length" else extract_json(sample.text)
                          for sample in output.outputs]
                if kind == "world":
                    worlds[ident] = values
                    stats["worlds"] += len(values)
                    stats["bad_worlds"] += sum(value is None for value in values)
                    parse_records.append({"kind": "world", "story": ident, "worlds": values})
                else:
                    forms[ident] = values
                    stats["forms"] += len(values)
                    stats["bad_forms"] += sum(value is None for value in values)
                    parse_records.append({"kind": "form", "key": ident, "forms": values})
            parse_writer.write_many(parse_records)
            records = []
            for story in chunk:
                for key, _, question in by_story[story]:
                    answer, scores, votes = answer_question(worlds[story], forms[key], question, config)
                    records.append({"key": key, "q_type": question["q_type"], "answer": answer, "scores": scores,
                                    "source": args.source, "votes": votes})
                    stats["abstain"] += answer is None
                    stats["votes"] += votes
            writer.write_many(records)
            stats["questions"] += len(records)
            upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")
        overall_progress(f"{args.source} {job.dataset}/{job.split}", None, None, done + len(todo),
                         done + len(todo), stats["seconds"], len(todo))

        count = max(stats["questions"], 1)
        tag = f"infer/{job.dataset}/{job.split}"
        tracking.log({f"{tag}/questions": stats["questions"], f"{tag}/seconds": stats["seconds"],
                      f"{tag}/abstain_rate": stats["abstain"] / count,
                      f"{tag}/invalid_world_rate": stats["bad_worlds"] / max(stats["worlds"], 1),
                      f"{tag}/invalid_form_rate": stats["bad_forms"] / max(stats["forms"], 1),
                      f"{tag}/mean_votes": stats["votes"] / count, f"{tag}/gpu_hours": stats["seconds"] / 3600})
        print(f"  abstained {stats['abstain']}/{stats['questions']}, unreadable worlds "
              f"{stats['bad_worlds']}/{stats['worlds']}, unreadable forms {stats['bad_forms']}/{stats['forms']}")
        upload_predictions(parses_path, args.results_repo, f"{args.results_prefix}/{args.source}")
        revision = upload_predictions(job.output, args.results_repo, f"{args.results_prefix}/{args.source}")
        if revision:
            tracking.log_hf_link(hub.resolve_repo(args.results_repo), revision,
                                 f"pred_{job.dataset}_{job.split}", "dataset")

        records = read_predictions(job.output)
        report = score_predictions(data, records, job.dataset, PostprocessOptions(), fallback)
        if report is not None:
            print_report(report, f"[{args.source}] {job.dataset}/{job.split} with fallback (n={report['questions']})")
            log_report(report, job.dataset, job.split)
    tracking.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
