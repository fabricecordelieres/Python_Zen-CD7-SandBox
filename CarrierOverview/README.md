# CarrierOverview

This folder contains a class file that eases the process of extracting metadata from a CarrierOverview.czi file.
It also provides functions to get, display and save the file into regular image formats.

This class has two dependencies:
* __OpenCV Python__ that should be installed using [pip install opencv-python](https://pypi.org/project/opencv-python/). It allows displaying/resaving the image into a different format.
* __czifile__, a library allowing to parse the czi file's metadata and extracting the image: install it using the following instruction: [pip install czifile](https://pypi.org/project/czifile/)

## About the example files
They have been used in particular to decipher how the *_ImageCenterPosition_* are computed. Here are the exepected values:
|Dataset|ImageCenterPosition_X_mm|ImageCenterPosition_Y_mm|
|---|:---:|:---:|
|CarrierOverview.czi|13.68|38|
|CarrierOverview2.czi|13.73|38|
|CarrierOverview3.czi|57.18|38|
|CarrierOverview4.czi|57.23|38|
|CarrierOverview5.czi|55.93|40|
The positions' extraction still needs to be a bit retuned: works for the first 3 datasets, has a 1 digit discrepency on the second last ones.
