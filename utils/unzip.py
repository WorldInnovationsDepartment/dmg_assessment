import os
import zipfile

def unzip_all_in_folder(folder_path):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Create a directory to unzip the files into
            extract_folder = os.path.join(folder_path, filename[:-4])
            os.makedirs(extract_folder, exist_ok=True)
            # Open the zip file and extract all contents
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f"Extracted {filename} in {extract_folder}")


if __name__ == '__main__':
    folder_path = 'data/annotated'
    unzip_all_in_folder(folder_path)