from desta.utils.manifest_utils import load_manifest

def get_remaining_data(input_data, output_data, key="audio_filepath"):
    """
    Get remaining data to process.
    Args:
        key: Key to use to identify data.
    Returns:
        List of remaining data.
    """
    output_filepaths = {item[key] for item in output_data}

    remaining_data = [item for item in input_data if item[key] not in output_filepaths]

    return remaining_data
