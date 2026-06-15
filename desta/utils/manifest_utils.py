import json
import os

def load_manifest(file_path):
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]