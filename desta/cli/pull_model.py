import argparse
from huggingface_hub import snapshot_download
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--allow_patterns", "-p", type=str, required=True, help="e.g. 'a100/250324-05@ds-8a100/*'")

    parser.add_argument("--repo_id", type=str, default="kehanlu/my-cool-model")
    parser.add_argument("--revision", "-r", default="my_exps", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_download(
        args.repo_id, local_dir=args.output_dir, allow_patterns=args.allow_patterns, revision=args.revision, local_dir_use_symlinks=False
    )
    


if __name__ == "__main__":
    main()