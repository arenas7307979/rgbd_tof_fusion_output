RGB AND Detph Calibration
Input:
	RGB(gray) image
	Depth PCL (X_depthcam / Y_depthcam / Z_depthcam)
-----------------------------------------------
0. catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo or Release
   catkin config --merge-devel

1. catkin build
2. change sensor(rgb and depth) parameter in 
   docker_slam/datasets/CamOdomCalibraTool/src/camOdoCalib/config
3. change checkboard parameters:
   docker_slam/datasets/CamOdomCalibraTool/src/camOdoCalib/src/calc_cam_pose/calcCamPose.cpp

4. roslaunch cam_calibration rgbd_calibration.launch

