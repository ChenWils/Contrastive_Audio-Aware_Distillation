import json
from collections import defaultdict
from pathlib import Path
import argparse


def parse_condition(category: str) -> str:
    """Extract condition from 'MCR@TASK@CONDITION' format."""
    parts = category.split("@")
    return parts[-1].lower()  # neutral / faithful / adversarial / irrelevant


def compute_TIR_adv(jsonl_path: str) -> dict:
    # Group entries by (task, index) so we can pair neutral vs adversarial
    # key: (task_prefix, index) -> {condition: correct}
    samples = defaultdict(dict)

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            category = entry["category"]          # e.g. MCR@SER@neutral
            condition = parse_condition(category)
            task_prefix = "@".join(category.split("@")[:2])  # e.g. MCR@SER
            audio = entry["audios"][0]["audio_filepath"]
            correct = entry["correct"]
            samples[(task_prefix, audio)][condition] = correct

    # Count correct->incorrect flips between neutral and adversarial
    delta_ci = 0       # neutral correct AND adversarial incorrect
    N = 0              # samples that have BOTH neutral and adversarial entries
    N_neu_correct = 0  # samples that are neutral-correct (true denominator)

    for (task_prefix, idx), conds in samples.items():
        if "neutral" in conds and "adversarial" in conds:
            N += 1
            if conds["neutral"]:
                N_neu_correct += 1
                if not conds["adversarial"]:
                    delta_ci += 1

    TIR_adv = delta_ci / N_neu_correct if N_neu_correct > 0 else 0.0

    return {
        "TIR_adv (normalized by neutral-correct)": TIR_adv,
        "delta_ci (correct->incorrect)": delta_ci,
        "N_neu_correct (denominator)": N_neu_correct,
        "N (paired samples)": N,
        "acc_neu": N_neu_correct / N if N > 0 else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to the prediction JSONL file")
    args = parser.parse_args()

    results = compute_TIR_adv(args.jsonl_path)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
