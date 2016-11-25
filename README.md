Learn Computer Vision and OpenCV Through Examples
=================================================

This a sequence of chapters in book form that I have been working on.  Each chapter contains an IPython notebook with code for easy re-use, and markdown for easy reading.  Each chapter is self-contained and all sample code is included in that chapter.

Learn Computer Vision and OpenCV Through Examples
This a sequence of chapters in book form that I have been working on. Each chapter contains an IPython notebook with code for easy re-use, and markdown for easy reading. Each chapter is self-contained and all sample code is included in that chapter.

Installing OpenCV 3.1 on Ubuntu

How to install OpenCV 3.1 on Ubuntu
In [ ]:
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
$ sudo apt-get install build-essential cmake git pkg-config
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libatlas-base-dev gfortran	
$ sudo apt-get install libgtk2.0-dev
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python get-pip.py
$ pip install numpy	
$ sudo apt-get install python2.7-dev
$ cd ~
$ git clone https://github.com/Itseez/opencv.git
$ cd opencv
$ git checkout 3.0.0
$ cd ~/opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=OFF ..
    
$ sudo make -j4 
$ sudo make install
$ sudo ldconfig
