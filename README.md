

# yeastvision 

<img src="yeastvision/docs/figs/logo.png" height="200" align="right" vspace=10 hspace = 0>

### A GUI-based framework for deep-learning enabled segmentation, tracking, time-series analysis of the full Saccharomyces cerevisiae lifecycle 

[![PyPI version](https://badge.fury.io/py/yeastvision.svg)](https://badge.fury.io/py/yeastvision)
[![Downloads](https://pepy.tech/badge/yeastvision)](https://pepy.tech/project/yeastvision)
[![Downloads](https://pepy.tech/badge/yeastvision/month)](https://pepy.tech/project/yeastvision)
[![Python version](https://img.shields.io/pypi/pyversions/yeastvision)](https://pypistats.org/packages/yeastvision)
[![License: GPL v3](https://img.shields.io/github/license/berkyalcinkaya/yeastvision)](https://github.com/berkyalcinkaya/yeastvision/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/berkyalcinkaya/yeastvision)](https://github.com/berkyalcinkaya/yeastvision/graphs/contributors)
[![repo size](https://img.shields.io/github/repo-size/berkyalcinkaya/yeastvision)](https://github.com/berkyalcinkaya/yeastvision/)
[![GitHub stars](https://img.shields.io/github/stars/berkyalcinkaya/yeastvision?style=social)](https://github.com/berkyalcinkaya/yeastvision/)
[![GitHub forks](https://img.shields.io/github/forks/berkyalcinkaya/yeastvision?style=social)](https://github.com/berkyalcinkaya/yeastvision/)

<img src="yeastvision/docs/figs/lifecycle_general.png" title="Saccharomyces cerevisiae full lifecycle">
<em>Yeastvision can identify and track a single cell throughout all life cycle stages</em>

<br/>

<img src="yeastvision/docs/figs/gui.png" height = 350 title="yeastvision GUI window" align=right>

### Key Features

- Enhance time-series resolution up to 16x using a generative video interpolation model
- Load, analyze, and segment multiple experiments containing numerous phase/flourescent channels
- Segment cytoplasm, vacuoles, buds, mating, and sporulating yeast cells
- Track and reconstruct lineages of large cell colonies
- Extract and plot time-series data in the GUI


# Installation

## Local installation (< 2 minutes)

### System requirements

This package supports Linux, Windows and Mac OS (versions later than Yosemite). GPU support is available for NVIDIA GPU's. A GPU is recommended, but not required, to run `yeastvision`
 
### Instructions 

`yeastvision` is ready to go for cpu-usage as soon as it downloaded. GPU-usage requires some additional steps after download. To download:

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt/command prompt
3. If you have an older `yeastvision` environment you should remove it with `conda env remove -n yeastvision` before creating a new one. 
4. Create a new environment with `conda create --name yeastvision python=3.10.0`. 
5. Activate this new environment by running `conda activate yeastvision`
6. Run `python -m pip install yeastvision` to download our package plus all dependencies
7. Download the weights [online](https://drive.google.com/file/d/1PuI6UIwKyuAUBoRnzjlZWkuT5p6_PX_C/view?usp=sharing). 
8. Run `install-weights` in the same directory as the *yeastvision_weights.zip* file


You should upgrade the [yeastvision package](https://pypi.org/project/yeastvision/) periodically as it is still in development. To do so, run the following in the environment:

~~~sh
python -m pip install yeastvision --upgrade
~~~

### Using yeastvision with Nvidia GPU: PyTorch Configurations

To use your NVIDIA GPU with python, you will first need to install a [NVIDIA driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) for
your GPU. Once downloaded, ensure that your
GPU is detected by running `nvidia-smi` in the terminal.

Yeastvision relies on `pytorch` for implementation of the deep-learning models, which we will need to configure for gpu usage. Ensure your yeastvision conda environment is active for the following commands.

First, we need to remove the CPU version of torch:
~~~
pip uninstall torch
~~~
And the cpu version of torchvision:
~~~
pip uninstall torchvision
~~~

Now install `torch` and `torchvision` for CUDA version 11.3 (Ensure that your nvidia drivers are up to date for version 11.3 by running `nvidia-smi` and check that a version >=11.3 is displayed in the top right corner of the output table).
~~~
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
~~~~

After install you can check `conda list` for `pytorch`, and its version info should have `cuXX.X`, not `cpu`.

# Running yeastvision

## Quickstart
Activate the correct conda environment. In the command prompt, run either
- `yeastvision -test` to open the GUI with our sample 10-frame movie.
- `yeastvision` to open a blank GUI window. Drag and drop <ins>directory</ins> containing 2D-image files into the GUI to begin segmenting, labeling, or processing

Yeastvision accepts directories of image files, loaded through drag-and-drop or the file menu. Each file in the directory contains only a single 2D image, named with a standard image file extension.

**Note:** As you segment, interpolate, or process images, GUI-generated labels and images are stored in the loaded directory as `.npz` files. Deleting these files results in loss of the data. **Yeastvision expects directories to contain only files with valid image extensions or `.npz` files created in previous yeastvision sessions.**

## To load directories containing additional image channels and pre-generated masks, name your files properly:
For experiments containing multiple channels and pre-generated labels, our GUI loading capabilities make it easy to:
- Track pre-generated labels (a nucleus label, for example)
- extract and plot fluorescence intensities
- interpolate additional fluorescence channels.  

__When additional channels/masks are present in the directory, files should be named accordingly to their channel and mask type__:
1. Each data type in the directory should have a standard id that directly precedes the file extension (ex: *_phase.tif, _channel2.png, _mask1.tif*)
3. Any image that acts as a label should have `_mask` in the id (ex: *_mask_nucleus.tif, _mask_cytoplasm.jpg*).
4. Ensure that each id has the same number of time points: Having 5 phase images but only 4 fluorescence images will raise an error.

Here is an example of a directory with two time points, two channels, and two pre-generated labels, sorted by name:

*im001_channel1.tif, im001_channel2.tif, <br>
im001_mask1.tif, im001_mask1.tif, <br>
im002_channel1.tif, im002_channel2.tif, <br>
im002_mask1.tif, im002_mask2.tif* <br>





# GUI Features

## Segmentation
Yeastvision contains models to accurately segment yeast in all stages of their lifecycle. Simply choose one of the following models from the model dropdown and click run. </em>

**Pixel flow-based models**

| Model  | Segments |
| ------ | -------- |
| proSeg | proliferating cells (general cytoplasm segmentation) |
| spoSeg | sporulating cells |
| matSeg | mating cells |
| budSeg | bud-necks |

<img src="yeastvision/docs/figs/lifecycle_segmentation.png" title="Saccharomyces cerevisiae full lifecycle">
<em>Yeastvision contains models and tracking algorithms to analyze all stages of the yeast lifecycle</em>

## Model Retraining
1. Ensure that training data is loaded into the current experiment
2. Select the model to be retrained from the mainscreen model dropdown 
3. Click Menu->Models->Retrain
4. Ensure training data is correct and choose model suffix (default is date-time)
5. Select hyperparameters (default should work for most use cases)
6. Train the model, using terminal to gauge progress. 
7. The custom model will auto-run on the next available image in the training set, if there is not a mask already on this image. 
8. If you are happy with the new model, go to Menu->Models->Load Custom Models, and the model will be added to the model dropdown. Otherwise, retrain with new data

#### Retraining Tips
- Even though it possible to retrain using only CPU, training takes very long without a GPU 
- When you are initially producing a training set, leave some blank masks towards the end of the movie so that the training procedure has room to auto-run
- The path to the new weights will be printed on the terminal. 
- Ensure that the fullname of the retrained model is present in the weights filename upon trying to load it via the models menu. This ensures that GUI can associate the weights with the correct model architecture 

## Timeseries analysis: interpolation, tracking, and plotting
1) Optional: Interpolate images to increase resolution, generating intermediate frames that improve tracking accuracy
2) Segment and track the interpolated frame. Tracking automatically generates a cell data table that includes various morphological and image properties of each cell over each frame in the movie. 
3) Optional: reconstruct proliferating cell lineages
4) Remove interpolated frames from the mask, so that only frames present in the original movie exist in the final tracked movie
5) Produce some initial plots of the data using the 'show plot window' button. This will allow you to view single cell and population averages over time.




## Keyboard Shortcuts

| Command     | Function |
| ----------- | ----------|
| up/down     | scroll through channnels|
| cntrl + up/down|  Scroll through labels | 
| right/left arrows | scroll through timeseries |
| cntrl + right/left | scroll through timeseries by 3 |
| O | outline Drawing |
| B | brush Drawing |
| E | eraser |
| .| increment brush size |
| , | decrecement brush size
| Delete/Backspace | Delete Selected Cell |
| c | show current label contours |
| f | toggle probability (if present) |
| space bar | toggle mask display |
| p | show plot window |

## Troubleshooting: Common Problems

| Problem     | Solution |
| ----------- | ----------- |
| Cannot scroll through images/masks on the display | Click on the display to bring focus back to this widget|
| Loaded images without masks but cannot draw | An existing label must be present to draw: Add a blank label with File -> Add Blank Label |




