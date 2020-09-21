#ifndef RGBD_CALIBRATION_H
#define RGBD_CALIBRATION_H
#endif

#include <stdio.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <fstream> //zdf
#include <math.h>
#include <chrono>

//eigen
#include "Eigen/Dense"

#include <fstream>
#include <Eigen/Dense>
#include "sophus/se3.hpp"
#include <opencv2/opencv.hpp>

//camera model
#include "../src/camera_models/include/Camera.h"
#include "../src/camera_models/include/CameraFactory.h"
#include "../src/calc_cam_pose/calcCamPose.h"


//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/common/io.h>
#include <sensor_msgs/PointCloud2.h>

// PCL
typedef pcl::PointCloud<pcl::PointXYZ>  PCLCloud3;       ///< 3D pcl PointCloud.


//Note!!!
//校正板參數在calcCamPose.cpp修改
//Calibration board parameters are modified in calcCamPose.cpp.
class RGBD_CALIBRATION
{

public:
    struct RGBFrame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RGBFrame() {}
        virtual ~RGBFrame() {}
        double timestamp;
        std::vector<cv::Point3f> x3Dw;
        Sophus::SE3d Twc;
        std::vector<cv::Point2f> uv_distorted;
        std::vector<int> id_landmark;
    };
    using RGBFramePtr = std::shared_ptr<RGBFrame>;
    using RGBFrameConstPtr = std::shared_ptr<const RGBFrame>;

    struct DEPTHFrame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        DEPTHFrame();
        virtual ~DEPTHFrame(){}
    };
    using DEPTHFramePtr = std::shared_ptr<DEPTHFrame>;
    using DEPTHFrameConstPtr = std::shared_ptr<const DEPTHFrame>;
    //Note!!!
    //校正板參數在calcCamPose.cpp修改
    //Calibration board parameters are modified in calcCamPose.cpp.
    RGBD_CALIBRATION(CameraPtr depth_cam, CameraPtr rgb_cam, Sophus::SE3d& init_Tdepth_rgb);
    ~RGBD_CALIBRATION(){};
    //Note!!!
    //校正板參數在calcCamPose.cpp修改
    //Calibration board parameters are modified in calcCamPose.cpp.
    void Optimize(std::vector<std::tuple<cv::Mat, std::shared_ptr<PCLCloud3>, double>>& rgb_depth_time);
    void EstimateLocalModel(std::map<double, RGBFramePtr>& rgb_frame_vec, std::map<double, std::shared_ptr<PCLCloud3>>& depth_frame_vec);
private:
    CameraPtr model_depth_cam;
    CameraPtr model_rgb_cam;
    Sophus::SE3d Tdepth_rgb;
};
    using RGBD_CALIBRATIONPtr = std::shared_ptr<RGBD_CALIBRATION>;
    using RGBD_CALIBRATIONConstPtr = std::shared_ptr<const RGBD_CALIBRATION>;