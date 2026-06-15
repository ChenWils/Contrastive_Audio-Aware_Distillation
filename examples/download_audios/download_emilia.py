from huggingface_hub import snapshot_download
import argparse
import tarfile
import os
import json
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./emilia")
    parser.add_argument("--language", "-l", type=str, default="EN")
    parser.add_argument('--stage', '-t', type=str, nargs='+', required=True)

    return parser.parse_args()


def main(args):
    assert os.path.exists(args.data_dir), f"Data directory {args.data_dir} does not exist"
    os.makedirs(f"{args.data_dir}/downloads", exist_ok=True)
    
    if "download" in args.stage:
        print(f"Downloading {args.language} dataset")
        # snapshot_download(repo_id="amphion/Emilia-Dataset", 
                        # allow_patterns=[f"{args.language}/*{i:05d}.tar" for i in range(0, 99)], repo_type="dataset", local_dir=f"{args.data_dir}/downloads", resume_download=True)
        snapshot_download(repo_id="amphion/Emilia-Dataset", repo_type="dataset", local_dir=f"{args.data_dir}/downloads", resume_download=True)
        # downloads/KO/KO-B*.tar

    if "extract" in args.stage:
        # Extract all tar files in the specified directory
        print(f"Extracting tar files in {args.data_dir}/downloads/{args.language}")
        for file_path in tqdm(Path(f"{args.data_dir}/downloads/{args.language}").glob("**/*.tar")): 
            print(file_path)
            with tarfile.open(str(file_path), "r") as tar:
                audio_folder = os.path.join(args.data_dir, args.language, file_path.stem)
                os.makedirs(audio_folder, exist_ok=True)
                tar.extractall(path=audio_folder)
            print(f"Extracted {file_path} to {audio_folder}")

    if "manifest" in args.stage:
        print(f"Creating manifest file for {args.language} dataset")
        count = 0
        for audio_folder in Path(f"{args.data_dir}/{args.language}").iterdir():
            print(audio_folder)
            Path(f"{args.data_dir}/manifest_{args.language}").mkdir(parents=True, exist_ok=True)
            fo = open(f"{args.data_dir}/manifest_{args.language}/{audio_folder.stem}.jsonl", "w")
            for file_path in tqdm(audio_folder.glob("*.json")):
                with file_path.open("r") as f:
                    data = json.load(f)

                    audio_id = data["id"]
                    audio_filepath = f"{args.language}/{audio_folder.stem}/{audio_id}.mp3"
                    
                    new_data = {
                        "id": audio_id,
                        "audio_filepath":  audio_filepath,
                        "text": data["text"],
                        "duration": data["duration"],
                        "speaker": data["speaker"],
                        "language": data["language"],
                        "dnsmos": data["dnsmos"]
                    }
                    count += 1
                    fo.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            fo.close()
        print(f"There are {count} files.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
