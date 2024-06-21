import os
import zipfile

def compress_folder(folder_path):
    """
    Compresses a folder into a zip file in the same location.
    """
    zip_filename = f"{os.path.basename(folder_path)}.zip"
    zip_path = os.path.join(os.path.dirname(folder_path), zip_filename)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))

    print(f"Compressed '{folder_path}' to '{zip_path}'")

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.getcwd()

    # Iterate through the subdirectories in the current directory
    for folder_name in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder_name)
        if os.path.isdir(folder_path):
            compress_folder(folder_path)

