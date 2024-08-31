import numpy as np
from skimage.morphology import disk, dilation, opening
from scipy.ndimage import zoom
from yeastvision.models.proSeg.model import ProSeg
from yeastvision.models.budSeg.model import BudSeg
from yeastvision.models.matSeg.model import MatSeg
from yeastvision.models.spoSeg.model import SpoSeg
from yeastvision.models.utils import produce_weight_path


def resize_image(image, target_shape):
    zoom_factors = [n / float(o) for n, o in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=0)

def extend_seg_output(to_segment, output, start, stop):
    correct_len = []
    for i in range(len(output)):
        template = np.zeros_like(to_segment, dtype=np.uint16)
        template[start:stop] = output[i]
        correct_len.append(template)
    return correct_len


def _get_proSeg(proSeg_params, proSeg_weights)->ProSeg: 
    if not proSeg_params:
        proSeg_params = ProSeg.hyperparams
    if not proSeg_weights:
        proSeg_weights = produce_weight_path("proSeg", "proSeg")
    return ProSeg(proSeg_params, proSeg_weights), proSeg_params, proSeg_weights

def _get_matSeg(matSeg_params, matSeg_weights)->MatSeg: 
    if not matSeg_params:
        budSeg_params = MatSeg.hyperparams
    if not matSeg_weights:
        matSeg_weights = produce_weight_path("matSeg", "matSeg")
    return MatSeg(budSeg_params, matSeg_weights), matSeg_params, matSeg_weights

def _get_spoSeg(spoSeg_params, spoSeg_weights)->SpoSeg: 
    if not spoSeg_params:
        spoSeg_params = SpoSeg.hyperparams
    if not spoSeg_weights:
        spoSeg_weights = produce_weight_path("spoSeg", "spoSeg")
    return SpoSeg(spoSeg_params, spoSeg_weights), spoSeg_params, spoSeg_weights

def _get_budSeg(budSeg_params, budSeg_weights)->BudSeg: 
    if not budSeg_params:
        budSeg_params = BudSeg.hyperparams
    if not budSeg_weights:
        budSeg_weights = produce_weight_path("budSeg", "budSeg")
    return BudSeg(budSeg_params, budSeg_weights), budSeg_params, budSeg_weights

def remove_artif(I2A,disk_size): # I2A = IS2 % disk radius is 3 for ~500x~1000, 6 for larger images
# we need a function to define the disk size base in the average cell size
    I2AA=np.copy(I2A) #   plt.imshow(IS2)
    # Applying logical operation and morphological opening
    I2A1 = binar(I2A);#binar(I2A) plt.imshow(I2A1)     plt.imshow(I2A)
 

    # Create a disk-shaped structuring element with radius 3
    selem = disk(disk_size)
    # Perform morphological opening
    I2B = opening(I2A1, selem)

       
    # Morphological dilation   plt.imshow(I2B)
    I2C = dilation(I2B, disk(disk_size))  # Adjust the disk size as needed


    I3 = I2AA * I2C # plt.imshow(I3)

    # Extract unique objects
    objs = np.unique(I3)
    objs = objs[1:len(objs)]
    
    # Initialize an image of zeros with the same size as I2A
    I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
    # Mapping the original image values where they match the unique objects
    AZ=1
    for obj in objs:
        I4[I2A == obj] = AZ
        AZ=AZ+1
    
    return I4

#Helper Functions

def OAM_23121_tp3(M, cel, no_obj1, A):
    tp3 = M.copy()
    tp3[tp3 == cel] = no_obj1 + A
    return tp3

def resize_image(image, target_shape):
    zoom_factors = [n / float(o) for n, o in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=0)

def binar(IS1):
    # Copy IS1 to IS1B
    IS1B = np.copy(IS1)
    
    # Convert non-zero elements to 1
    IS1B[IS1 != 0] = 1

    
    return IS1B

def cal_celldata(all_obj, ccel):
    cell_data = np.zeros((ccel, 5))

    for iv in range(ccel):
        first_occurrence = np.argmax(all_obj[iv, :] > 0)
        last_occurrence = len(all_obj[iv, :]) - np.argmax((all_obj[iv, :][::-1] > 0)) - 1
        
        cell_data[iv, 0] = first_occurrence  # 1st occurrence
        cell_data[iv, 1] = last_occurrence   # Last occurrence
        
    for iv in range(ccel):
        cell_data[iv, 2] = cell_data[iv, 1] - cell_data[iv, 0] + 1  # Times the cell appears
        aa1 = all_obj[iv, :]
        aa2 = aa1[int(cell_data[iv, 0]):int(cell_data[iv, 1]) + 1]
        aa3 = np.where(aa2 == 0)[0]
        cell_data[iv, 3] = len(aa3)  # Number of times it disappears between 1st and last occurrence
        cell_data[iv, 4] = (cell_data[iv, 3] * 100) / cell_data[iv, 2]  # Percentage of times the cell disappears

    return cell_data


def cal_allob(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC[0])))

    for iv in range(0, ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[0][its] is not None:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[0][its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

# Removing artifacts - cells that appear once and cells that disappear thresh % of the time or more
def cal_allob1(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC[1])))

    for iv in range(ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[0][its] is not None: #and np.sum(TETC[0][its]) > 0:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[0][its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

def cal_allob2(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC)))

    for iv in range(ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[its] is not None: #and np.sum(TETC[its]) > 0:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

def replace_none_with_empty_array(data):
    if isinstance(data, list):
        return [replace_none_with_empty_array(item) for item in data]
    elif data is None:
        return np.array([])
    else:
        return data
    
    