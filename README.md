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

Next we need to remove the CPU version of torch:
~~~
pip uninstall torch
~~~
And the cpu version of torchvision:
~~~
pip uninstall torchvision
~~~

To install the GPU version of torch and torchvision, first ensure you have downloaded the proper nvidia drivers for your GPU. Then for pytorch and torchvision, follow the instructions [here](https://pytorch.org/get-started/locally/). The conda install is strongly recommended, and then choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 10.2). You can check the highest version of CUDA that your nvidia driver supports by running: 
~~~
nvidia-smi
~~~
For instance this command will install the 11.6 version on Linux and Windows (note the `torchaudio` commands are removed because yeastvision doesn't require them):
~~~
conda install pytorch==1.12.0 torchvision==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
~~~
The 11.6 configuration is recommended as this system was thoroughly tested with this system.  However, for some GPUs which do not support CUDA 11.6 or later, the above command will timeout. In that case, you can quickly try an older version like cuda 11.3:
~~~
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
~~~~
Info on how to install several older versions is available [here](https://pytorch.org/get-started/previous-versions/). 

After install you can check `conda list` for `pytorch`, and its version info should have `cuXX.X`, not `cpu`.

## Common Installation Problems

You may receive the following error upon upgrading `torch` and `torchvision`:
~~~
AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)
~~~
This is solved by upgrading the charselt_normalizer package with the following command: `AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)`

Report any other installation errors.

# Run yeastvision locally

The quickest way to start is to open the GUI from a command line terminal. Activate the correct conda environment, then run:
~~~~
yeastvision
~~~~

To get started, drop an image or directory of images into the GUI. 

**Masks can be loaded by dropping them into the top half of the screen.**



