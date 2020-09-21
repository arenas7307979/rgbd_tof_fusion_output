cmake_minimum_required(VERSION 2.8.3)
project(camodocal)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
include
${EIGEN3_INCLUDE_DIR}
)

cs_add_library(${PROJECT_NAME} SHARED
src/camera_models/Camera.cc
src/camera_models/CameraFactory.cc
src/camera_models/CataCamera.cc
src/camera_models/CostFunctionFactory.cc
src/camera_models/EquidistantCamera.cc
src/camera_models/PinholeCamera.cc
src/camera_models/ScaramuzzaCamera.cc
src/camera_models/gpl.cc
)

cs_install()
cs_export()
