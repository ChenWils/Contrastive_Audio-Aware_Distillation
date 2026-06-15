import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model
from pathlib import Path
import logging
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    # output_dir = Path(args.output_dir) / args.model_id.replace("/", "--")
    output_dir = Path(args.model_id).parent.parent / "pred_results" / "Speech-IFeval" / args.model_id.replace("/", "--")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "logs").mkdir(parents=True, exist_ok=True)



    
    manifest_paths = [
        Path(data_dir) / "Speech-IFeval" / "close.jsonl",
        Path(data_dir) / "Speech-IFeval" / "open.jsonl",
        # Path(data_dir) / "Speech-IFeval" / "close-woprompt.jsonl",
        Path(data_dir) / "Speech-IFeval" / "cot.jsonl",
    ]

    # Load model
    model = DeSTA3Model.from_pretrained(
        args.model_id,
        cache_dir=os.getenv("HF_HOME"),
        token=os.getenv("HF_TOKEN"),
    )
    model.to("cuda")
    
    for manifest_path in manifest_paths:
        output_file = output_dir / manifest_path.name

        # logging to a file path that is the same as the manifest file
        logging.basicConfig(filename=output_dir / f"{manifest_path.stem}.log", level=logging.INFO)
        
        logging.info(f"Processing {manifest_path}")
        logging.info(f"Output file: {output_file}")

        with manifest_path.open("r") as fin, output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines()]


            for data in tqdm(datas):

                # TODO: Replace with actual model inference logic
                messages = data["messages"]

                audios = data["messages"][-1]["audios"]
                audios = [{"audio": os.path.join(data_dir, a["audio"])} for a in audios]
                data["messages"][-1]["audios"] = audios

                
                
                generated_ids = model.generate(
                    messages=messages,
                    generation_kwargs={"max_new_tokens": 256, "do_sample": False},
                )

            
                data.update({
                    "pred": model.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
                })
                
                fout.write(json.dumps(data) + "\n")
                logging.info(json.dumps(data))

if __name__ == "__main__":
    args = arg_parser()
    main(args)