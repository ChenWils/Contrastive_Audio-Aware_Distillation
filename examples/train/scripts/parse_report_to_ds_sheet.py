import argparse
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Just a parser to parse the \"report\" to the ds google sheet by the ordered task names")
    parser.add_argument("--input_manifest_filepath", "-i", type=str, required=True)
    return parser.parse_args()

def main(args):
    categories_accuracy = json.load(open(args.input_manifest_filepath))["categories_accuracy"]

    task_list = (Path(__file__).resolve().parent / "assets/ds_tasks.txt")
    for task in task_list.open().readlines():
        task = task.strip()
        if task:
            if categories_accuracy.get(f"DS@{task}") is not None:
                print(categories_accuracy[f"DS@{task}"])
            else:
                print("N/A")
        else:
            print()
    print()
    print(args.input_manifest_filepath)

if __name__ == "__main__":
    args = parse_args()
    main(args)
