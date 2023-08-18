# <p>  <b>YeastVision </b> </p>


# Installation

## Local installation (< 2 minutes)

### System requirements

This package supports Linux, Windows and Mac OS. Mac Os should be later than Yosemite. This system has been heavily tested on Linux and Mac OS machines, and less thoroughly on Windows. 
 
### Instructions 

If you have an older `yeastvision` environment you should remove it with `conda env remove -n yeastvision` before creating a new one. 

Yeastvision is ready to go for cpu-usage as soon as it downloaded. GPU-usage requires some additional steps after download. To download:

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt/command prompt
3. Create a new environment with `conda create --name yeastvision python=3.10.0`. 
4. Activate this new environment by running `conda activate yeastvision`
5. Run `python -m pip install yeastvision` to download our package plus all dependencies
6. Download the weights [online](https://drive.google.com/file/d/12FM_DaEQVNGsnDrK_YVX9huxqyVO8Nyb/view?usp=share_link). 
7. Run `install-weights` in the same directory as the *yeastvision_weights.zip* file


You should upgrade yeastvision (package [here](https://pypi.org/project/yeastvision/)) periodically as it is still in development. To do so, run the following in the environment:

~~~sh
python -m pip install yeastvision --upgrade
~~~

### Using YeastVision with Nvidia GPU

Again, enusre your yeastvision conda environment is active for the following commands.

To use your NVIDIA GPU with python, you will first need to install the NVIDIA driver for your GPU, check out this [website](https://www.nvidia.com/Download/index.aspx?lang=en-us) to download it. Ensure it is downloaded and your GPU is detected by running `nvidia-smi` in the terminal.

Yeastvision relies on two machine-learning frameworks: `tensorflow` and `pytorch`. We will need to configure both of these packages for gpu usage

#### PyTorch

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

#### Tensorflow

All we need to do here is install the cuDNN package for tensorflow gpu usage
~~~
conda install cudnn=8.1.0
~~~

## Common Installation Problems

You may receive the following error upon upgrading `torch` and `torchvision`:
~~~
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
~~~
This is solved by upgrading the charselt_normalizer package with the following command: `pip install --force-reinstall charset-normalizer==3.1.0`

Report any other installation errors.

# Run yeastvision locally

The quickest way to start is to open the GUI from a command line terminal. Activate the correct conda environment, then run:
~~~~
yeastvision
~~~~

To get started, drop an image or directory of images into the GUI. 

**Masks can be loaded by dropping them into the top half of the screen.**



