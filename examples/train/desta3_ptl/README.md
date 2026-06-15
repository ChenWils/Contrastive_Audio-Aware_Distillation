

```
pip install -e .
pip install git+https://github.com/kehanlu/lulutils.git
pip install whisper_normalizer
pip install transformers==4.49.0
```

```
./examples/train/desta3_ptl/run_desta3.sh
```


Dataset:
- Training:
    - Audios: https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/tree/audios/desta2_audios
        - (only needs tars for audios files, ignore other metadatas)
    - manifest(response from llama3.2): https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/resolve/audios/manifest3/241226_respond2%400811_seed_transcript.1.jsonl

- Dynamic-superb-eval:
    - Audios: https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/tree/audios/dynamic-superb-eval
    - manifest: https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/resolve/audios/manifest3/240720_dynamic-superb-eval.jsonl

- Dynamic-superb1-test
    - Audios: https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/tree/audios/dynamic-superb-test/
    - https://huggingface.co/datasets/kehanlu/df93f08c-c28d-4db9-ad5c-44c6f3d21360/resolve/audios/manifest3/250109_dynamic-superb-test.jsonl


---

### Evaluation:
`2025.03.27`

**Data format**

```
# This will download the tar files and extract them into audios.
# Ignore the jsonl files along with the tar files (they are metadata of the dataset, only for debugging purpose not for our codebase, we only need the audios)
# Then properly set the "manifest" and "data_root" to load actual text/audio data.


desta-pull-audios --data_root /root/lab/DeSTA3-dev/my_data --repo_id Morioh/livingroom -p dynamic-superb-test --revision eval -t download extract 

# or

python examples/download_audios/download_from_hf.py --repo_id Morioh/livingroom --revision desta --path_in_repo $path_in_repo --data_root /root/lab/DeSTA3-dev/my_data --stage download extract

# or change the path in the shell script

./examples/download_audios/download_morioh.sh
```


```
CUDA_VISIBLE_DEVICES=0
./examples/train/desta3_ptl/eval_desta3.sh $CUDA_VISIBLE_DEVICES
```