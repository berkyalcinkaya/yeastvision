import zipfile
import shutil
import os
from models.utils import MODEL_DIR
from os.path import join

def do_install():
    current_directory = os.getcwd()
    folder_path = "yeastvision_weights.zip"
    models = [model for model in os.listdir(MODEL_DIR) if os.path.isdir(join(MODEL_DIR, model))]

    unzip_directory = os.path.join(current_directory, 'unzipped_files')
    os.makedirs(unzip_directory, exist_ok=True)

    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_directory)

    # Move each file into a directory with the same name
    for root, dirs, files in os.walk(unzip_directory):
        for file_name in files:
            # Get the name of the file without the extension
            file_name_without_extension = os.path.splitext(file_name)[0]

            if file_name_without_extension == "bud_Seg":
                destination_directory = join(MODEL_DIR, "artilife/budSeg")

            else:
                destination_name = ""
                for model in models:
                    if model in file_name_without_extension:
                        destination_name = model
                        exit
                destination_directory = join(MODEL_DIR, destination_directory)

            # Move the file into the destination directory
            source_file_path = os.path.join(root, file_name)
            destination_file_path = os.path.join(destination_directory, file_name)
            shutil.move(source_file_path, destination_file_path)

    # Remove the unzipped folder
    shutil.rmtree(unzip_directory)


if __name__ == "__main__":
    do_install()



