# Image-Compression

This repository contains two independent compression algorithms LZW and JPEG.

## LZW 
LZW (Lempel–Ziv–Welch) is a very interesting comrpession technique as it is a dictionary based image compression technique but dictionary is not send from one end to other rather it is build at each end at both compressing and decomressing end. It is a lossless compression algorithm.

Enter the filename of the image to be comrpessed in the code and the code would generate a file named lzw_encodecode.txt and display the entropy of the image and compression achieved. Currently this code is limited only for B&W photos.

## JPEG
JPEG (Joint Photographic Experts Group) is currently most commonly used image compression algorithm. It uses DCT (Discrete Cosine Transform). It is a lossy compression scheme. It eliminates data of those frequencies which are not easily perceived by human eyes.

The code is not perfectly copy of the actual JPEG compression algorithm but focuses on only the DCT and quantization part. It produces 3 separate files for colored images each for YUV components. 

## Dependencies
- [python](https://www.python.org/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [numpy](https://numpy.org/)

## Installation
```sh
pip install numpy
pip install opencv-python
```
If you have 2 versions of python installed on your system then replace ```pip``` by ```pip3```

## Running
For lzw compression
```sh
python LZW_Code.py
```

For jpeg compression
```sh
python JPEG_Code.py
```

If you have 2 versions of python installed on your system then replace ```python``` by ```python3```
