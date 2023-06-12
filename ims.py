from skimage.exposure import equalize_adapthist
from skimage.filters import median, gaussian
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte, img_as_uint, img_as_float
import numpy as np
import warnings

def do_adapt_hist(ims):
    return [equalize_adapthist(im) for im in ims]

def do_median(ims):
    return [median(im,np.ones((3,3))) for im in ims]

def do_gaussian(ims):
    return [gaussian(im, np.ones((3,3))) for im in ims]


def rolling_mean(data, window):
    ''' compute the rolling mean of the data over the given window '''

    result = np.full_like(data, np.nan)

    conv = np.convolve(data, np.ones(window)/window, mode='valid')
    result[(len(data) - len(conv))//2: (len(conv) - len(data))//2] = conv

    return result

def calc_brightness(images, sigma=2.5):
    brightness = []
    for image in images:
        mask = np.ones_like(image, dtype=bool)
        if sigma is not None:
            mean = np.mean(image)
            std = np.std(image)
            dist = np.abs(image - mean) / std
            mask[dist > sigma] = False
        brightness.append(np.mean(image[mask]))
    return np.array(brightness)

def scale_image_brightness(image, scale):
    ''' scale image brightness by a factor '''
    adjusted_image = scale * img_as_float(image)
    # handle overflow:
    adjusted_image[adjusted_image >= 1.0] = 1.0

    # catch warning for loosing some accuracy by converting back to int types
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        if image.dtype == np.dtype('uint8'):
            adjusted_image = img_as_ubyte(adjusted_image)
        elif image.dtype == np.dtype('uint16'):
            adjusted_image = img_as_uint(adjusted_image)

    return adjusted_image


def deflicker(images, window = 10, sigma=2.5):
    ''' Deflicker images
    Image brightness is scaled to match a rolling mean to avoid flickering
    Parameters
    ----------
    images: list of strings
        Filenames of the images that should be processed
    window: int
        The width of the window for the rolling mean
    outdir: string
        The directory where the adjusted images are saved
    fmt: string
        Output format. One of png, tiff, jpg
    '''

    brightness = calc_brightness(images, sigma=sigma)
    target_brightness = rolling_mean(brightness, window)
    adjusted_ims = []
    for image, b, tb in zip(images, brightness, target_brightness):
        if np.isnan(tb):
            adjusted_image = image.copy()
        else:
            adjusted_image = scale_image_brightness(image, tb / b)
        adjusted_ims.append(adjusted_image)
    return adjusted_ims




