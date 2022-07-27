# TorchRDM
This package contains some basic code for you to get started with obtaining and analyzing RDM (Representation Dissimilarity Matrix) with PyTorch as the backend. Each of the functionalities is tested to some degree, but bugs are still possible. Please use with caution.

## Before using
### Development install
TODO: This is not complete and may screw with torch version. Maybe use poetry to package and manage dev usage?
Please install the requirements by running `pip install -r requirements.txt`.

## Basic usage
### Preprocessing the data
First contact Chris or anyone from OLab/HeLab to obtain the data. You can then use the `process_images` function from `src.utils.process_imgs` to preprocess the images.

### Finding the RDM of a `nn.Module` model
If you have a model and the preprocessed data, you can use the `RDMFinder` class from `src.RDMFinder` to easily find the RDM for some specified region/layer of interest in your network.

### Second Order RDM
If you have an RDM and the human RDM, you can use the `SecOrdRDM` class from `src.SecOrdRDM` to calculate the 2nd order RDM.

### Analyzing the RDM
Some simple analysis functions are provided in `src.utils.analysis` to analyze the RDM. These can be applied to the RDM tensors directly.

### Visualization
WIP
