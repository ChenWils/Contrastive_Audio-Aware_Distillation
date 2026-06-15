"""
LLM Judge using GPT-4o-mini to evaluate emotion classification predictions.

Usage:
    export OPENAI_API_KEY=sk-...
    python llm_judge.py --input path/to/epoch=10.jsonl [--output path/to/output.jsonl] [--concurrency 20]
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are an evaluation judge for an emotion classification task. "
    "The model was asked a question and produced a prediction. "
    "Your job is to determine whether the prediction correctly identifies the ground truth emotion label. "
    "The valid emotions are: sad, happy, fearful, angry, surprised, disgusted, neutral.\n\n"
    "IMPORTANT: The model sometimes answers using option numbers (e.g. 'Option 1', 'Option 2') "
    "instead of naming the emotion directly. When this happens, use the question text to resolve "
    "which emotion each numbered option refers to — options are ordered as they appear in the question. "
    "For example, if the question lists 'sad, happy, fearful, ...' then Option 1 = sad, Option 2 = happy, etc.\n\n"
    "Reply with exactly one word: 'yes' if the prediction is correct, 'no' if it is incorrect."
)


def compute_tir_adv(results: list[dict], correct_key: str) -> dict:
    """
    For each unique (task, audio) pair, check if the sample was correct under the
    'neutral' condition but flipped to incorrect under the 'adversarial' condition.

    TIR_adv = delta_ci / N_neu_correct
      delta_ci      — paired samples where neutral=correct AND adversarial=incorrect
      N_neu_correct — paired samples where neutral=correct (denominator)
    """
    samples: dict = defaultdict(dict)
    for entry in results:
        category = entry.get("category", "")
        parts = category.split("@")
        if len(parts) < 3:
            continue
        condition = parts[-1].lower()          # neutral / faithful / adversarial / ...
        task_prefix = "@".join(parts[:2])      # e.g. MCR@SER
        audio = entry["audios"][0]["audio_filepath"]
        samples[(task_prefix, audio)][condition] = entry.get(correct_key, False)

    delta_ci = 0
    N = 0
    N_neu_correct = 0
    for (_, _), conds in samples.items():
        if "neutral" in conds and "adversarial" in conds:
            N += 1
            if conds["neutral"]:
                N_neu_correct += 1
                if not conds["adversarial"]:
                    delta_ci += 1

    return {
        "TIR_adv": delta_ci / N_neu_correct if N_neu_correct > 0 else 0.0,
        "delta_ci": delta_ci,
        "N_neu_correct": N_neu_correct,
        "N_paired": N,
        "acc_neu": N_neu_correct / N if N > 0 else 0.0,
    }


def extract_question(item: dict) -> str:
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"].replace("<|AUDIO|>", "").strip()
    return ""


def build_user_prompt(prediction: str, label: str, question: str) -> str:
    return (
        f"Question asked to the model:\n{question}\n\n"
        f"Ground truth label: {label}\n"
        f"Model prediction: {prediction}\n\n"
        "Does the prediction correctly identify the emotion? "
        "Reply with exactly 'yes' or 'no'."
    )


VALID_EMOTIONS = {"sad", "happy", "fearful", "angry", "surprised", "disgusted", "neutral"}


def fast_judge(prediction: str, label: str) -> bool | None:
    """
    Return True/False if the prediction unambiguously matches a known emotion label.
    Return None if the prediction is ambiguous and needs the LLM judge.
    """
    import re
    cleaned = re.sub(r"[^a-z]", "", prediction.lower())  # strip all non-alpha chars
    if cleaned in VALID_EMOTIONS:
        return cleaned == label.lower()
    return None


async def judge_one(client: AsyncOpenAI, sem: asyncio.Semaphore, item: dict, idx: int) -> dict:
    prediction = item.get("prediction", "")
    label = item.get("label", "")

    fast = fast_judge(prediction, label)
    if fast is not None:
        result = dict(item)
        result["llm_judge_correct"] = fast
        result["llm_judge_raw"] = "fast_match"
        return result

    question = extract_question(item)

    async with sem:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prediction, label, question)},
            ],
            max_tokens=5,
            temperature=0.0,
        )

    answer = response.choices[0].message.content.strip().lower()
    llm_correct = answer.startswith("yes")

    result = dict(item)
    result["llm_judge_correct"] = llm_correct
    result["llm_judge_raw"] = answer
    return result


async def run(input_path: Path, output_path: Path, concurrency: int, test: bool = False):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load all items
    items = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # Resume: skip already-judged items
    done_indices: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    done_indices.add(d["index"])
        print(f"Resuming: {len(done_indices)} items already judged, skipping them.")

    pending = [item for item in items if item["index"] not in done_indices]
    if test:
        pending = pending[:5]
        print(f"[TEST MODE] Running on first 5 items only.")
    print(f"Total: {len(items)} | Pending: {len(pending)} | Concurrency: {concurrency}")

    sem = asyncio.Semaphore(concurrency)

    tasks = [judge_one(client, sem, item, item["index"]) for item in pending]

    results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 100 == 0 or completed == len(tasks):
            print(f"  Progress: {completed}/{len(tasks)}", flush=True)

    # Append new results to output file
    results.sort(key=lambda x: x["index"])
    mode = "a" if output_path.exists() and done_indices else "w"
    with open(output_path, mode) as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute final accuracy over all judged items
    all_results = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    all_results.sort(key=lambda x: x["index"])
    total = len(all_results)
    llm_correct = sum(1 for r in all_results if r["llm_judge_correct"])
    orig_correct = sum(1 for r in all_results if r.get("correct", False))

    print(f"\n=== Results ({total} items) ===")
    print(f"LLM Judge Accuracy : {llm_correct / total:.4f}  ({llm_correct}/{total})")
    print(f"Original Accuracy  : {orig_correct / total:.4f}  ({orig_correct}/{total})")

    # Per-category breakdown
    cat_stats: dict[str, dict] = defaultdict(lambda: {"llm": 0, "orig": 0, "total": 0})
    for r in all_results:
        cat = r.get("category", "unknown")
        cat_stats[cat]["total"] += 1
        if r["llm_judge_correct"]:
            cat_stats[cat]["llm"] += 1
        if r.get("correct", False):
            cat_stats[cat]["orig"] += 1

    print("\nPer-category breakdown:")
    print(f"{'Category':<35} {'LLM Acc':>10} {'Orig Acc':>10} {'Count':>8}")
    print("-" * 65)
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        n = s["total"]
        print(f"{cat:<35} {s['llm']/n:>10.4f} {s['orig']/n:>10.4f} {n:>8}")

    # TIR_adv: fraction of neutral-correct samples that flipped to wrong under adversarial
    def _print_tir(label: str, stats: dict) -> None:
        print(f"  TIR_adv              : {stats['TIR_adv']:.4f}")
        print(f"  correct→incorrect    : {stats['delta_ci']} / {stats['N_neu_correct']} neutral-correct samples")
        print(f"  neutral accuracy     : {stats['acc_neu']:.4f}")
        print(f"  paired samples (N)   : {stats['N_paired']}")

    tir_llm = compute_tir_adv(all_results, "llm_judge_correct")
    tir_orig = compute_tir_adv(all_results, "correct")

    print("\n=== TIR_adv — LLM Judge ===")
    _print_tir("llm", tir_llm)
    print("\n=== TIR_adv — Original ===")
    _print_tir("orig", tir_orig)


def main():
    parser = argparse.ArgumentParser(description="LLM Judge for emotion classification predictions")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to output JSONL file (default: input dir/llm_judge_results.jsonl)")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent API requests (default: 20)")
    parser.add_argument("--test", action="store_true", help="Run on first 5 items only to verify setup")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "llm_judge_results.jsonl"

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Error: OPENAI_API_KEY environment variable not set.")

    asyncio.run(run(input_path, output_path, args.concurrency, test=args.test))


if __name__ == "__main__":
    main()
