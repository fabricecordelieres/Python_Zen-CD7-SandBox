# CarrierOverview

This folder contains a class file that eases the process of extracting metadata from a CarrierOverview.czi file.
It also provides functions to get, display and save the file into regular image formats.

This class has two dependencies:
* __OpenCV Python__ that should be installed using [pip install opencv-python](https://pypi.org/project/opencv-python/). It allows displaying/resaving the image into a different format.
* __czifile__, a library allowing to parse the czi file's metadata and extracting the image: install it using the following instruction: [pip install czifile](https://pypi.org/project/czifile/)
