
from desta.collections.utils.metrics import ConsecutiveWordsAccuracyMetric
import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", "-i", type=str)
    parser.add_argument("--prediction_key", type=str, default="prediction")
    parser.add_argument("--label_key", type=str, default="label")
    return parser.parse_args()

def main(args):
    """
    --prediction_file: path to the prediction file
    --prediction_key: key to the prediction in the prediction file
    --label_key: key to the label in the prediction file

    The report should be easily copy and paste to the paper table. (implement yourself :))
    The report will be saved to the prediction file path with "-report.json" suffix.
    
    """
    
    
    categories_accuracy = defaultdict(list)

    # TODO: Implement your own metrics here
    metrics = ConsecutiveWordsAccuracyMetric()
    # calculate accuracy for each sample
    reported_results = []
    with open(args.prediction_file, "r") as f:
        for i, line in enumerate(f):
            result = json.loads(line)
            
            result["correct"] = metrics(result[args.prediction_key], result[args.label_key]) # save per sample result

            # Calculate per category accuracy
            categories_accuracy[result.get("category", "all")].append(result["correct"])
            reported_results.append(result)


    # write the report
    categories_accuracy = dict([ (k, sum(v) / len(v)) for k, v in categories_accuracy.items()])
    report = {
        "metric": metrics.metric_name,
        "preds_path": args.prediction_file,
        "accuracy_by_sample": sum([reported_results[i]["correct"] for i in range(len(reported_results))]) / len(reported_results),
        "avg_accuracy_by_category": sum(categories_accuracy.values()) / len(categories_accuracy),
        "categories_accuracy": categories_accuracy,
        "results": reported_results, # per sample results
    }

    print(report["categories_accuracy"])

    # save the report
    report_path = args.prediction_file.replace(".jsonl", "-report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to\n{report_path}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
