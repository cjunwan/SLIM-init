# Development image for SLIM-init.
#
# Ubuntu 22.04 provides Eigen 3.4, Boost 1.74, Ceres 2.0, TBB, glog and gflags
# from apt. OpenCV is built from source because the line tracker needs the
# contrib modules line_descriptor and ximgproc (EdgeDrawing), which require
# OpenCV >= 4.5.2.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        pkg-config \
        ca-certificates \
        libeigen3-dev \
        libboost-dev \
        libtbb-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libceres-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCV + opencv_contrib (only the modules this project needs).
ARG OPENCV_VERSION=4.12.0
WORKDIR /tmp
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip \
    && wget -q -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip \
    && unzip -q opencv.zip && unzip -q opencv_contrib.zip \
    && cmake -S opencv-${OPENCV_VERSION} -B opencv-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,features2d,flann,video,line_descriptor,ximgproc \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_JAVA=OFF \
        -DWITH_GTK=ON \
        -DWITH_TBB=OFF \
    && cmake --build opencv-build -j"$(nproc)" \
    && cmake --install opencv-build \
    && ldconfig \
    && rm -rf /tmp/opencv*

WORKDIR /workspace
CMD ["/bin/bash"]
