import zipfile
import shutil
import os
from yeastvision.models.utils import MODEL_DIR
from os.path import join
from yeastvision.ims.interpolate import RIFE_DIR

def do_install():
    current_directory = os.getcwd()
    folder_path = "yeastvision_weights.zip"
    models = [model for model in os.listdir(MODEL_DIR) if os.path.isdir(join(MODEL_DIR, model)) and model != "__pycache__"]
    print("Extracting weights for the following models", models)

    unzip_directory = os.path.join(current_directory, 'unzipped_files')
    os.makedirs(unzip_directory, exist_ok=True)

    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_directory)

    # Move each file into a directory with the same name
    for root, dirs, files in os.walk(unzip_directory):
        for file_name in files:
            print("found", file_name)

            if "._" not in file_name:
                # Get the name of the file without the extension
                file_name_without_extension = os.path.splitext(file_name)[0]

                if file_name_without_extension == "Bud_Seg":
                    destination_directory = join(MODEL_DIR, "artilife/budSeg")
                    print("\tlocated BudSeg weights -- moving to", destination_directory)
                
                elif file_name_without_extension == "flownet":
                    destination_directory = RIFE_DIR
                    print("\tlocated weights for movie interpolation -- moving to", destination_directory)

                else:
                    destination_name = ""
                    for model in models:
                        if model in file_name_without_extension:
                            destination_name = model
                            print("\tfound", file_name, "belong to", model)
                            exit
                    destination_directory = join(MODEL_DIR, destination_name)

                # Move the file into the destination directory
                source_file_path = os.path.join(root, file_name)
                destination_file_path = os.path.join(destination_directory, file_name)
                print("\t", destination_file_path)
                shutil.move(source_file_path, destination_file_path)

    # Remove the unzipped folder
    shutil.rmtree(unzip_directory)
    print("FINISHED - removing", unzip_directory)


if __name__ == "__main__":
    do_install()



