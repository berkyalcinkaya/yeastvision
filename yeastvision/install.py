import yeastvision
from yeastvision.models.utils import MODEL_DIR
from yeastvision.ims.interpolate import RIFE_DIR, RIFE_WEIGHTS_PATH
from tqdm import tqdm
import os
import requests

RIFE_FILENAME = "flownet.pkl"
RIFE_URL = f"https://github.com/berkyalcinkaya/yeastvision/blob/main/weights/{RIFE_FILENAME}?raw=True"
def MODEL_URL(model_name): return f"https://github.com/berkyalcinkaya/yeastvision/blob/main/weights/{model_name}?raw=True"
TEST_MOVIE_URL = "https://github.com/berkyalcinkaya/yeastvision/blob/main/data/sample_movie_1"
TEST_MOVIE_NUM_IMS = 10
TEST_MOVIE_ITEMS = ["cdc", "phase", "mask1"]
TEST_MOVIE_IM_FORMAT = "im00x"
TEST_MOVIE_DIR = os.path.join(os.path.dirname(yeastvision.__path__[0]), "data/sample_movie_1")


def install_weight(model_name):
    # Construct the URL for the model weights
    
    # Make the request to download the file
    response = requests.get(MODEL_URL(model_name))
    response.raise_for_status()  # This will raise an exception if the request failed
    
    # Define the file path where the weights will be saved
    file_path = os.path.join(MODEL_DIR, model_name, model_name)
    
    # Write the downloaded data to the file
    with open(file_path, 'wb') as file:
        file.write(response.content)

def install_rife():
    response = requests.get(RIFE_URL)
    response.raise_for_status()  # This will raise an exception if the request failed
    # Define the file path where the weights will be saved
    file_path = os.path.join(RIFE_DIR, RIFE_FILENAME)
    
    # Write the downloaded data to the file
    with open(file_path, 'wb') as file:
        file.write(response.content)

def install_test_ims():
    src = TEST_MOVIE_DIR

    if not os.path.exists(TEST_MOVIE_DIR):
        os.mkdir(src)

    for i in tqdm(range(TEST_MOVIE_NUM_IMS)):
        for extension in TEST_MOVIE_ITEMS:
            im_name = f"{TEST_MOVIE_IM_FORMAT.replace('x', str(i))}_{extension}.tif"
            im_url = f"{TEST_MOVIE_URL}/{im_name}?raw=True"

            response = requests.get(im_url)
            response.raise_for_status()

            outpath = os.path.join(src, im_name)
            with open(outpath, "wb") as file:
                file.write(response.content)

# def do_install():
#     current_directory = os.getcwd()
#     folder_path = "yeastvision_weights.zip"
#     models = [model for model in os.listdir(MODEL_DIR) if os.path.isdir(join(MODEL_DIR, model)) and model != "__pycache__"]
#     print("Extracting weights for the following models", models)

#     unzip_directory = os.path.join(current_directory, 'unzipped_files')
#     os.makedirs(unzip_directory, exist_ok=True)

#     with zipfile.ZipFile(folder_path, 'r') as zip_ref:
#         zip_ref.extractall(unzip_directory)

#     # Move each file into a directory with the same name
#     for root, dirs, files in os.walk(unzip_directory):
#         for file_name in files:
#             print("found", file_name)

#             if "._" not in file_name:
#                 # Get the name of the file without the extension
#                 file_name_without_extension = os.path.splitext(file_name)[0]
                
#                 if file_name_without_extension == "flownet":
#                     destination_directory = RIFE_DIR
#                     print("\tlocated weights for movie interpolation -- moving to", destination_directory)

#                 else:
#                     destination_name = ""
#                     for model in models:
#                         if model in file_name_without_extension:
#                             destination_name = model
#                             print("\tfound", file_name, "belong to", model)
#                             exit
#                     destination_directory = join(MODEL_DIR, destination_name)

#                 # Move the file into the destination directory
#                 source_file_path = os.path.join(root, file_name)
#                 destination_file_path = os.path.join(destination_directory, file_name)
#                 print("\t", destination_file_path)
#                 shutil.move(source_file_path, destination_file_path)

#     # Remove the unzipped folder
#     shutil.rmtree(unzip_directory)
#     print("FINISHED - removing", unzip_directory)


# if __name__ == "__main__":
#     do_install()



