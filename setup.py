from setuptools import setup

requires = [
"numpy==1.24",
"cellpose==2.1.0",
"matplotlib",
"munkres",
"opencv_python_headless",
"pandas==1.5.0",
"patchify==0.2.3",
"PyQt5==5.15.9",
"pyqtgraph==0.13.0",
"scikit_image",
"scikit_learn",
"scipy",
"tensorflow==2.10.0",
"torch==1.12.0",
"tqdm",
"trackpy",
"torchvision==0.13.0",
"memory-profiler",
"QSwitchControl"]

try:
    import torch
    a = torch.ones(2, 3)
    major_version, minor_version, _ = torch.__version__.split(".")
    if major_version == "2" or int(minor_version) >= 12:
        requires.remove("torch==1.12.0")
        requires.remove("torchvision==0.13.0")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = ["yeastvision","yeastvision.data",  "yeastvision.plot", "yeastvision.track", 
"yeastvision.models", "yeastvision.ims", "yeastvision.ims.rife_model",
            "yeastvision.ims.rife_model.pytorch_msssim", "yeastvision.parts", "yeastvision.flou", "yeastvision.disk", 
            "yeastvision.models.proSeg", "yeastvision.models.budSeg", "yeastvision.models.budSeg",
            "yeastvision.models.matSeg", "yeastvision.models.tetradSeg", "yeastvision.models.budNET", 
            "yeastvision.models.vacNET", "yeastvision.models.YeaZ"]


setup(
    name = "yeastvision",
    version = "0.1.26",
    description = "Deep learning-enabled image analysis of the full yeast life cycle",
    author = "Berk Yalcinkaya",
    url = "https://github.com/berkyalcinkaya/yeastvision",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email="berkyalcinkaya55@gmail.com",
    license = "BSD",
    packages = packages,
    install_requires = requires,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': [
          'yeastvision = yeastvision.__main__:main',
          'install-weights = yeastvision.install_weights:do_install']
       }
)
