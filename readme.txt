1. catkin build
2. change sensor(rgb and depth) parameter in 
   docker_slam/datasets/CamOdomCalibraTool/src/camOdoCalib/config
3. change checkboard parameters:
   docker_slam/datasets/CamOdomCalibraTool/src/camOdoCalib/src/calc_cam_pose/calcCamPose.cpp

4. roslaunch cam_calibration rgbd_calibration.launch

