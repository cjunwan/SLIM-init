cmake_minimum_required(VERSION 3.10)
project(slim_init CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DEIGEN_NO_DEBUG")

add_definitions(-DSOPHUS_USE_BASIC_LOGGING)
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Ceres REQUIRED)
find_package(Threads REQUIRED)
# line_descriptor and ximgproc (EdgeDrawing) come from opencv_contrib, OpenCV >= 4.5.2.
find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  calib3d
  highgui
  line_descriptor
  ximgproc)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

# Camera models (from camodocal).
add_library(camera_model SHARED
  src/camera_models/camera_models/Camera.cc
  src/camera_models/camera_models/CameraFactory.cc
  src/camera_models/camera_models/PinholeCamera.cc
  src/camera_models/camera_models/CataCamera.cc
  src/camera_models/camera_models/EquidistantCamera.cc
  src/camera_models/camera_models/ScaramuzzaCamera.cc
  src/camera_models/gpl/gpl.cc)
target_link_libraries(camera_model
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES})

# Feature tracking, IMU preintegration and the initialization method.
add_library(slim_init SHARED
  src/featureTracker/featureTracker.cpp
  src/featureTracker/lineFeatureTracker.cpp
  src/featureTracker/parameters.cpp
  src/IMU/basicTypes.cpp
  src/IMU/imuPreintegrated.cpp
  src/initMethod/drtVioInit.cpp
  src/initMethod/slimInit.cpp)
target_include_directories(slim_init PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/featureTracker
  ${PROJECT_SOURCE_DIR}/include/initMethod)
target_link_libraries(slim_init
  camera_model
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  stdc++fs)

# Evaluation driver for EuRoC-format datasets.
add_executable(run_euroc app/main.cpp)
target_link_libraries(run_euroc
  slim_init
  Threads::Threads)
