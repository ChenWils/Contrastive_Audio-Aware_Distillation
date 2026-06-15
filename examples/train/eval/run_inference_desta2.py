import os
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Config, DeSTA3Model
from transformers import AutoModel
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
    model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, token=os.getenv("HF_TOKEN"))
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
        for i, line in tqdm(enumerate(file)):
            data = json.loads(line)
            
            messages = data["messages"]

            new_messages = []
            for message in messages:
                if message["role"] == "system":
                    # message["content"] = message["content"].replace("Choose one of the options without any explanation.", "Choose one of the answer from the option list.")
                    pass
                    
                    new_messages.append(message)

                if message.get("audios") and message["role"] == "user":
                    for i, audio in enumerate(message["audios"]):
                        # get absolute path of audio, and patch the audio_filepath
                        audio["audio"] = _resolve_audio_filepath(os.path.join(data_root, audio["audio"]))
                        # message["content"] = message["content"] + "\n\nThink step by step before answering."

                        new_messages.append({
                            "role": "audio",
                            "content": audio["audio"],
                        })
                        new_messages.append({
                            "role": "user",
                            "content": message["content"].replace("<|AUDIO|>\n", ""),
                        })
        

            generated_ids = model.chat(
                new_messages, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.6, 
                top_p=0.9
            )

            response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        
            data.update({
                "pred": response,
                "index": i,
            })
            fo.write(json.dumps(data) + "\n")

        fo.close()

        print(f"Results saved to {prediction_filepath}")

if __name__ == "__main__":
    args = parse_args()
    main(args)