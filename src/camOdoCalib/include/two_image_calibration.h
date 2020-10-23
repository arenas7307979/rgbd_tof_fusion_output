#pragma onece
#include <ros/ros.h>
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
#include "sophus/se3.hpp"

//camera model
#include "../src/calc_cam_pose/calcCamPose.h"

class TWO_CAM_EXTRINSIC
{
    struct GrayRGBFrame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GrayRGBFrame() {}
        virtual ~GrayRGBFrame() {}
        double timestamp;
        std::vector<cv::Point3f> x3Dw;
        Sophus::SE3d Twc;
        std::vector<cv::Point2f> uv_distorted;
        std::vector<int> id_landmark;
    };
    using GrayRGBFramePtr = std::shared_ptr<GrayRGBFrame>;
    using GrayRGBFrameConstPtr = std::shared_ptr<const GrayRGBFrame>;

    struct GrayDepthFrame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GrayDepthFrame() {}
        virtual ~GrayDepthFrame() {}
        double timestamp;
        std::vector<cv::Point3f> x3Dw;
        Sophus::SE3d Twc;
        std::vector<cv::Point2f> uv_distorted;
        std::vector<int> id_landmark;
    };
    using GrayDepthFramePtr = std::shared_ptr<GrayDepthFrame>;
    using GrayDepthFrameConstPtr = std::shared_ptr<const GrayDepthFrame>;

    //rgb_cam_ and depth_cam_ are gray image
    TWO_CAM_EXTRINSIC(CameraPtr cam0, CameraPtr cam1, Sophus::SE3d &init_T01);
    ~TWO_CAM_EXTRINSIC(){};

    void Optimize(std::vector<GrayRGBFrame> cam0_msg_vec, std::vector<GrayDepthFrame> cam1_msg_vec);
};

using TWO_CAM_ExtrinsicPtr = std::shared_ptr<TWO_CAM_EXTRINSIC>;
using TWO_CAM_ExtrinsicConstPtr = std::shared_ptr<const TWO_CAM_EXTRINSIC>;