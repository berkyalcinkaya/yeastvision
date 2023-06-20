from setuptools import setup, find_packages

requires = ["cellpose==2.1.0",
"matplotlib==3.6.0",
"munkres==1.1.4",
"numpy==1.23.3",
"opencv_python_headless==4.6.0.66",
"pandas==1.5.0",
"patchify==0.2.3",
"Pillow==9.5.0",
"PyQt5==5.15.9",
"pyqtgraph==0.13.0",
"scikit_image==0.19.3",
"scikit_learn==1.2.2",
"scipy==1.9.1",
"tensorflow==2.10.0",
"torch==1.12.1",
"tqdm==4.65.0",
"trackpy==0.5.0"]

packages = ["yeastvision", "yeastvision.plot", "yeastvision.track", "yeastvision.models", 
            "yeastvision.parts", "yeastvision.flou", "yeastvision.disk", 
            "yeastvision.models.artilife", "yeastvision.models.artilife.budSeg",
            "yeastvision.models.matSeg", "yeastvision.models.tetradSeg", "yeastvision.models.budNET", 
            "yeastvision.models.vacNET", "yeastvision.models.YeaZ"]

setup(
    name = "yeastvision",
    version = "0.1.1",
    description = "Deep learning-enabled image analysis of the yeast full life cycle",
    author = "Berk Yalcinkaya",
    url = "https://github.com/berkyalcinkaya/budNET_gui",
    author_email="berkyalcinkaya55@gmail.com",
    license = "BSD",
    packages = packages,
    install_requires = requires,
    entry_points = {
        'console_scripts': [
          'yeastvision = yeastvision.__main__:main',
          'install-weights = yeastvision.install_weights:do_install']
       }
)
