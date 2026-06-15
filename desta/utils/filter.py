from pathlib import Path
import os

testing_audio_ids = set(line.strip() for line in open(Path(__file__).parent / "assets" / "test_audio_ids.txt"))

def is_testing_data(data):
    if "test" in data.get("split", "").lower():
        return True
    elif Path(data["audio_filepath"]).stem in testing_audio_ids:
        return True
    else:
        return False