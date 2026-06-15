import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
import tarfile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--repo_id", type=str)
    parser.add_argument("--revision", type=str, default="desta")
    parser.add_argument("--path_in_repo", '-p', type=str)
    parser.add_argument('--stage', type=str, nargs='+', required=True)

    return parser.parse_args()



def main():
    args = parse_args()

    if "download" in args.stage:
        snapshot_download(
            repo_id=args.repo_id, 
            repo_type="dataset", 
            local_dir=f"{args.data_root}/downloads",
            revision=args.revision, 
            allow_patterns=[f"{args.path_in_repo}*.tar"]
        )
    if "extract" in args.stage:
        for tar_file in tqdm(sorted(Path(f"{args.data_root}/downloads/{args.path_in_repo}").glob("*.tar"))):
            print(f"Extracting {tar_file}")
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(path=f"{args.data_root}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
/home/chenwils/DeSTA3-dev-main/data/desta2