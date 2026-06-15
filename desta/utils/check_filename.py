
import os

def check_and_modify_filename(file_path):
    """
    Check if the given file path exists and is a file. If it exists, modify the file path.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        str: The modified file path if the file exists, otherwise the original file path.
    """
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_path)
        i = 1
        new_file_path = f"{base}.{i}{ext}"
        
        # Loop to find a non-existing file path
        while os.path.isfile(new_file_path):
            i += 1
            new_file_path = f"{base}.{i}{ext}"
        
        return new_file_path
    return file_path


