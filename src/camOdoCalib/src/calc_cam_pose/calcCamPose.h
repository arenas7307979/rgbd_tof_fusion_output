#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "camera_models/Camera.h"
#include "ethz_apriltag/Tag36h11.h"
#include "ethz_apriltag/TagDetector.h"
namespace rgbd_calibration
{

//chessboard info
static const int col = 6;
static const int row = 5;
static const float square_size = 0.108; // unit:  m

//threshold for remove larget distance of center of pcl and chessboard.
static const float diff_pcl_and_chessboard_center = 0.08;

// static const double chessboard_obs_distance_max = 0.07;

//remove same view (move>0.04 --> obs)
static const double chessboard_obs_distance_min = 0.04;

enum PatternType
{
    APRIL,
    CHESS,
    CIRCLE
};

void FindTargetCorner(cv::Mat &img_raw, const PatternType &pt,
                      std::vector<cv::Point3f> &p3ds,
                      std::vector<cv::Point2f> &p2ds,
                      std::vector<int> &id_landmark);
bool EstimatePose(const std::vector<cv::Point3f> &p3ds,
                  const std::vector<cv::Point2f> &p2ds, const double &fx,
                  const double &cx, const double &fy, const double &cy,
                  Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, cv::Mat &img_raw, const CameraPtr &cam);
bool calcCamPose(const double &timestamps, const cv::Mat &image,
                 const CameraPtr &cam, Eigen::Matrix4d &Twc);

bool calcCamPoseRGBD(const double &timestamps, const cv::Mat &image,
                     const CameraPtr &cam, Eigen::Matrix4d &Twc, std::vector<cv::Point3f> &x3Dw,
                     std::vector<cv::Point2f> &uv_2d_distorted, std::vector<int> &id_landmark);
} // namespace rgbd_calibration
