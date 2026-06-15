import os
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Config, DeSTA3Model
from tqdm import tqdm
import json
from lulutils import resolve_filepath, get_unique_filepath
import argparse
from desta.collections.desta3.data.simple_dataset import _resolve_audio_filepath

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/lab/DeSTA3-dev/workspace/exps/a100/250326-19@ds-8a100-syn5/hf_models/epoch-10")
    parser.add_argument("--test_manifest", type=str)
    parser.add_argument("--data_root", type=str)
    return parser.parse_args()


def main(args):

    # dict: {audio_filepath: transcription}
    
    model = DeSTA3Model.from_pretrained(args.model_path)
    model.to("cuda")

    cache_path = os.path.join(os.getenv("HF_HOME"), f"transcription_cache-{model.config.encoder_model_id.replace('/', '__')}.json")
    if os.path.exists(cache_path):
        transcription_cache = json.load(open(cache_path))
    else:
        transcription_cache = {}

    test_manifest = resolve_filepath(args.test_manifest)

    results = []
    data_root = args.data_root
    fo = open(get_unique_filepath("MMAU_test.jsonl"), "w")

    for i, line in tqdm(enumerate(open(test_manifest).readlines())):
        data = json.loads(line)
        new_messages = []

        # SUPER important note:
        # 250416: 因為目前是拿舊的格式，所以需要一個轉換，請直接參考 README.md 的輸入格式。或許在明天我會在上傳新的版本。
        for message in data["messages"]:
            if message["role"] == "system":
                new_messages.append({
                    "role": "system",
                    "content": "Imagine you can **hear** the audio clips. Focus on the speech and instruction. Choose one of the options without any explanation."
                })
            if message["role"] == "user":
                new_messages.append(
                    {
                        "role": "user",
                        "content": message["content"].replace("<start_audio><|AUDIO|><end_audio>", "<|AUDIO|>\n"),
                        "audios": [
                            {"audio": 
                             _resolve_audio_filepath(os.path.join(data_root, audio_dict["audio_filepath"])),
                             "transcription": transcription_cache.get(_resolve_audio_filepath(os.path.join(data_root, audio_dict["audio_filepath"])), None)
                             } for audio_dict in data["audios"]]
                    }
                )

        # print(new_messages)
        generated_ids, audios = model.generate(
            messages=new_messages,
            generation_kwargs={"max_new_tokens": 512, "do_sample": False},
            return_audios=True
        )
        # audio is a list of tuple (audio, transcription)
        for a, t in audios:
            print(t)
            transcription_cache[a] = t
        

        result = {
            "messages": new_messages,
            "target": data["target"],
            "pred": model.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
            "metric": "accuracy",
            "category": data["category"],
            "index": i,
        }
        results.append(result)
        fo.write(json.dumps(result) + "\n")

    fo.close()

    with open(cache_path, "w") as f:
        json.dump(transcription_cache, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)