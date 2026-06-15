
# 方便的 inference interface


```
為了方便討論：
evluation: 訓練完後，offline 跑模型 generation，最終產生分數的過程

- inference: 指的是跑模型 generate，產生 prediction file
- 算分數：從 prediction file，用各種 metric 算出分數
```


- 請注意：請使用 `hf_model/` 資料夾底下的 ckpt. `checkpoints/` 底下的是 pytorch lightning 的 ckpt.


## Download a model folder

### `a100/250417-20@250411_llama32_whisperL_qformer6L@250416_llama32-33%`

```bash
python desta/cli/pull_model.py --repo_id Morioh/toilet --revision my_exps --allow_patterns "a100/250326-19@ds-8a100-syn5/hf_models/epoch-10/*" --output_dir /path/to/your/exps_dir
```


### `a100/250417-20@250411_llama32_whisperL_qformer6L@250416_llama32-33%`
https://huggingface.co/Morioh/toilet/tree/my_exps/a100/250417-20%40250411_llama32_whisperL_qformer6L%40250416_llama32-33%25/hf_models/epoch%3D3-step%3D216381

  - 用非常多有的沒的資料訓練的版本。
  - 請優先使用 `epoch=4-step=270478` 再來是 `epoch=3-step=216381` ckpt
```
python desta/cli/pull_model.py --repo_id Morioh/toilet --revision my_exps --allow_patterns "a100/250417-20@250411_llama32_whisperL_qformer6L@250416_llama32-33%/hf_models/epoch=4-step=270478/*" --output_dir /path/to/your/exps_dir
```

## Easy Inference

- messages 中的每個 item，若 content 有 `<|AUDIO|>` 則需要額外提供 `audios` 的資訊，且會檢查 `<|AUDIO|>` 的數量和 `len(audios)` 是否一致

```python
messages = [  # List[Dict]
    {
        "role": "system",  # str
        "content": "Focus on the speech and instruction. Choose one answer from the options without explaination.",  # str
    },
    {
        "role": "user",  # str
        "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe.",  # str, where "<|AUDIO|>" is a special token to mark the audio position
        "audios": [  # List[Dict]
            {
                "audio": "/path/to/filepath",  # str - path to filepath
                "transcription": "Hello world" # str(optional), if none or not provided, the model will do ASR for you.
            }
        ]
    },
]
```

```python
# Load model
model = DeSTA3Model.from_pretrained("/path/to/your/exps_dir/a100/250326-19@ds-8a100-syn5/hf_models/epoch-10/") # this is a checkpoint "folder"
model.to("cuda")

# Generate with "messages"
generated_ids = model.generate(
    messages=messages,
    generation_kwargs={"max_new_tokens": 512, "do_sample": False},
)

print(model.tokenizer.decode(generated_ids[0]))


# ===== Advanced usage =====
generated_ids, audios = model.generate(
    messages=messages,
    generation_kwargs={"max_new_tokens": 512, "do_sample": False},
    return_audios=True # set return_audios
)

print(audios) # this is ASR transcription, you can save this as cache next time you run the same dataset.
```


---

## Benchmark Evaluation

### Download audios

```shell
data_root="/path/to/data_root"

# Dynamic-superb-test
python desta/cli/pull_audios.py --data_root $data_root --repo_id Morioh/livingroom -p dynamic-superb-test --stage download extract --revision eval

# MMAU
python desta/cli/pull_audios.py --data_root $data_root --repo_id Morioh/livingroom -p MMAU-test-mini --stage download extract --revision eval

# Speech-IFeval
python desta/cli/pull_audios.py --data_root $data_root --repo_id Morioh/livingroom -p Speech-IFeval --stage download extract --revision eval
```

### Evaluation

```shell
model_path=/path/to/hf_model/epoch-X
```

#### `Dynamic-superb-test`
```bash
python examples/train/eval/run_inference.py --model_path $model_path \ 
  --data_root $data_root \ 
  --test_manifest https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/hf_eval_manifest/dynamic-superb-test.test.jsonl
```
```bash
python examples/train/eval/eval_accuracy.py -i /path/to/dynamic-superb-test.test/epoch-2.jsonl --prediction_key pred --label_key target
```


#### `MMAU-test-mini`
```bash
python examples/train/eval/run_inference.py --model_path $model_path \ 
  --data_root $data_root \ 
  --test_manifest https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/hf_eval_manifest/MMAU-test-mini.jsonl 
```

```bash
python examples/train/eval/eval_accuracy.py -i /path/to/MMAU-test-mini/epoch-2.jsonl --prediction_key pred --label_key target
```


#### `Speech-IFeval-close`
```bash
python examples/train/eval/run_inference.py \ 
    --model_path $model_path \ 
    --data_root $data_root \ 
    --test_manifest https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/hf_eval_manifest/Speech-IFeval-close.jsonl
```
