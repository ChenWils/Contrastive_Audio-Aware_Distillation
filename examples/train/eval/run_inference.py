import os
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Config, DeSTA3Model
from tqdm import tqdm
import json
from lulutils import resolve_filepath, get_unique_filepath
import argparse
from desta.collections.desta3.data.simple_dataset import _resolve_audio_filepath
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="/lab/DeSTA3-dev/workspace/exps/a100/250326-19@ds-8a100-syn5/hf_models/epoch-10")
    parser.add_argument("--test_manifest", type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main(args):

    print(f"Loading model from {args.model_path}")
    model_path = args.model_path
    model = DeSTA3Model.from_pretrained(model_path)
    model.to(args.device)


    print(f"Loading test manifest from {args.test_manifest}")
    print(f"data root: {args.data_root}")
    test_manifest = resolve_filepath(args.test_manifest)
    test_name = Path(test_manifest).stem
    data_root = args.data_root

    prediction_filepath = Path(model_path).parent.parent / "pred_results" / f"{test_name}" / f"{Path(model_path).stem}.jsonl"
    prediction_filepath.parent.mkdir(parents=True, exist_ok=True)

    fo = open(get_unique_filepath(str(prediction_filepath)), "w")
    with open(test_manifest, 'r') as file:
        for i, line in enumerate(tqdm(file)):
            data = json.loads(line)
            
            messages = data["messages"]

            for message in messages:
                if message.get("audios"):
                    for i, audio in enumerate(message["audios"]):
                        # get absolute path of audio, and patch the audio_filepath
                        audio["audio"] = _resolve_audio_filepath(os.path.join(data_root, audio["audio"]))


            generated_ids = model.generate(
                messages=messages,
                generation_kwargs={"max_new_tokens": 512, "do_sample": False},
            )
        
            data.update({
                "pred": model.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
                "index": i,
            })
            fo.write(json.dumps(data) + "\n")

        fo.close()

        print(f"Results saved to {prediction_filepath}")

if __name__ == "__main__":
    args = parse_args()
    main(args)