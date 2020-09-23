cmake_minimum_required(VERSION 2.8.3)
project(cam_calibration)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)


# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     std_msgs
#     geometry_msgs
#     nav_msgs
#     tf
#     cv_bridge
#     image_transport)

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
    include
    )

find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)



catkin_package()

SET(CSM_SRC 
  src/csm/laser_data.cpp
  src/csm/math_utils.cpp
  src/csm/math_utils_gsl.cpp
  src/csm/utils.cpp
  src/csm/logging.cpp
  #csm/sm_options.cpp
  src/csm/orientation.cpp
  src/csm/clustering.cpp)

SET(EGSL_SRC
    src/egsl/egsl.cpp
    src/egsl/egsl_conversions.cpp
    src/egsl/egsl_misc.cpp
    src/egsl/egsl_ops.cpp)

ADD_LIBRARY(csm_eigen
    ${CSM_SRC}
    ${EGSL_SRC} 
    )

add_library(calc_lib
    src/camera_models/src/Utils.cc
    src/camera_models/src/Camera.cc
    src/camera_models/src/CameraFactory.cc
    src/camera_models/src/PinholeCamera.cc
    src/camera_models/src/PinholeFullCamera.cc
    src/camera_models/src/CataCamera.cc
    src/camera_models/src/EquidistantCamera.cc
    src/camera_models/src/ScaramuzzaCamera.cc
    src/calc_cam_pose/calcCamPose.cpp
    src/calc_cam_pose/ethz_apriltag/Edge.cc
    src/calc_cam_pose/ethz_apriltag/FloatImage.cc
    src/calc_cam_pose/ethz_apriltag/Gaussian.cc
    src/calc_cam_pose/ethz_apriltag/GLine2D.cc
    src/calc_cam_pose/ethz_apriltag/GLineSegment2D.cc
    src/calc_cam_pose/ethz_apriltag/GrayModel.cc
    src/calc_cam_pose/ethz_apriltag/Homography33.cc
    src/calc_cam_pose/ethz_apriltag/MathUtil.cc
    src/calc_cam_pose/ethz_apriltag/Quad.cc
    src/calc_cam_pose/ethz_apriltag/Segment.cc
    src/calc_cam_pose/ethz_apriltag/TagDetection.cc
    src/calc_cam_pose/ethz_apriltag/TagDetector.cc
    src/calc_cam_pose/ethz_apriltag/TagFamily.cc
    src/calc_cam_pose/ethz_apriltag/UnionFindSimple.cc
    src/data_selection/data_selection.cpp)

add_library(calib_lib
    src/rgbd_calibration.cpp
    # src/plane_extraction.cpp
    src/solver.cpp
    src/utils.cpp
    src/solveQyx.cpp
 )

target_link_libraries(calc_lib ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(cam_odo_cal src/main_node.cpp )
target_link_libraries(cam_odo_cal calib_lib calc_lib csm_eigen ${CERES_LIBRARIES}) 


add_executable(rgbd_calibration_node src/main_rgbd_calibration_node.cpp )
target_link_libraries(rgbd_calibration_node calib_lib calc_lib csm_eigen ${CERES_LIBRARIES}) 

#### simulation  ###
# add_executable(simulate simulateData/main.cpp simulateData/simulateOdoCam.cpp)
# target_link_libraries(simulate ${OpenCV_LIBS})
