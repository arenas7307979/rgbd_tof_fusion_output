#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <fstream> //zdf
#include <math.h>
#include <chrono>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "rgbd_calibration.h"
#define opencv_viewer

template <class T>
void GetParam(const std::string &param_name, T &param)
{
  if (!ros::param::get(param_name, param))
  {
    ROS_ERROR_STREAM("Can not find \"" << param_name << "\" in ROS Parameter Server");
    ros::shutdown();
  }
}

class RGBDCheckerNode
{
public:
  RGBDCheckerNode(ros::NodeHandle &pr_nh)
  {
    depth_pub_ = pr_nh.advertise<sensor_msgs::PointCloud2>("/calibration/depth_cloud", 10);

    std::string config_file_depth, config_file_rgb;
    GetParam("/calibration/config_file_depth", config_file_depth);
    GetParam("/calibration/config_file_rgb", config_file_rgb);

    //camera parameter in config folder
    camera_depth = CameraFactory::instance()->generateCameraFromYamlFile(config_file_depth);
    camera_rgb = CameraFactory::instance()->generateCameraFromYamlFile(config_file_rgb);
    std::vector<double> depthparam, rgb_param;
    //get intrinsic parameter
    depthparam = camera_depth->getK();
    rgb_param = camera_rgb->getK();
    for (size_t i = 0; i < rgb_param.size(); i++)
    {
      std::cout << "depthparam[i] : " << depthparam[i] << std::endl;
      std::cout << "rgb_param[i] : " << rgb_param[i] << std::endl;
    }

    //Get initial Tdepth_rgb extrinsic param
    double qw, qx, qy, qz;
    double x, y, z;
    GetParam("/Tdepth_rgb/tx", x);
    GetParam("/Tdepth_rgb/ty", y);
    GetParam("/Tdepth_rgb/tz", z);
    GetParam("/Tdepth_rgb/qx", qx);
    GetParam("/Tdepth_rgb/qy", qy);
    GetParam("/Tdepth_rgb/qz", qz);
    GetParam("/Tdepth_rgb/qw", qw);
    Tdepth_rgb.translation().x() = x;
    Tdepth_rgb.translation().y() = y;
    Tdepth_rgb.translation().z() = z;
    Tdepth_rgb.setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    // std::cout << "Tdepth_rgb translation: " << Tdepth_rgb.translation() << std::endl;
    // std::cout << "Tdepth_unit_quaternion: " << Tdepth_rgb.so3().unit_quaternion().coeffs() << std::endl;
    // Tdepth_rgb = Tdepth_rgb.inverse();
    // std::cout << "Tdepth_rgb= " << Tdepth_rgb.translation() <<std::endl;
    // std::cout << "Tdepth_rgb= " << Tdepth_rgb.unit_quaternion().coeffs() <<std::endl;
    //Note!!!
    //校正板參數在calcCamPose.cpp修改
    //Calibration board parameters are modified in calcCamPose.cpp.
    rgbd_calibr = std::make_shared<RGBD_CALIBRATION>(camera_depth, camera_rgb, Tdepth_rgb);
  }
  ~RGBDCheckerNode() {}

  void ImageDepthImgCallback(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &depth_msg)
  {

    if(img_msg == nullptr || depth_msg == nullptr)
      return;

    cv::Mat mono_img, depth_img;
    double timestamp = img_msg->header.stamp.toSec();
    if (img_msg->encoding == sensor_msgs::image_encodings::BGR8 || 
    img_msg->encoding == sensor_msgs::image_encodings::RGB8 || 
    img_msg->encoding == sensor_msgs::image_encodings::BGRA8)
    {
      //rgb image
      cv::Mat rgb_img;
      rgb_img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
      cv::cvtColor(rgb_img, mono_img, CV_BGR2GRAY);
    }
    else if (img_msg->encoding == sensor_msgs::image_encodings::MONO8 || img_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      mono_img = cv_bridge::toCvShare(img_msg, "mono8")->image;
    }
    if (depth_msg->encoding == sensor_msgs::image_encodings::BGR8 || depth_msg->encoding == sensor_msgs::image_encodings::RGB8)
    {

      //rgb image
      cv::Mat rgb_img;
      rgb_img = cv_bridge::toCvShare(depth_msg, "bgr8")->image;
      cv::cvtColor(rgb_img, depth_img, CV_BGR2GRAY);
    }
    else if (depth_msg->encoding == sensor_msgs::image_encodings::MONO8 || depth_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      depth_img = cv_bridge::toCvShare(depth_msg, "mono8")->image;
    }

#if 1
    Eigen::Matrix4d Twc_mono;
    std::vector<cv::Point3f> x3Dw_mono;
    std::vector<cv::Point2f> uv_2d_distorted_mono;
    std::vector<int> id_landmark_mono;
    std::vector<cv::Point2f> reproj_mono;
    bool isCalOk;
    isCalOk = rgbd_calibration::calcCamPoseRGBD(frame_index, mono_img, camera_rgb, Twc_mono, x3Dw_mono, uv_2d_distorted_mono, id_landmark_mono);
    if(isCalOk == false)
     return;

    cv::Mat debug_img = mono_img.clone();
    cv::cvtColor(debug_img, debug_img, CV_GRAY2BGR);

    Sophus::SE3d Twc_mono_soph;
    if (x3Dw_mono.size() == rgbd_calibration::row * rgbd_calibration::col)
    {
        Eigen::Matrix3d roatation_matrix = Twc_mono.block(0, 0, 3, 3).matrix();
        Eigen::Quaterniond qwc(roatation_matrix);
        Twc_mono_soph.translation()= Twc_mono.block(0, 3, 3, 1);
        Twc_mono_soph.setQuaternion(qwc);

      //Project Xw to "stereo Left image" to check instrinsic value
      for (int k = 0; k < x3Dw_mono.size(); k++)
      {
        Eigen::Vector3d x3c_mono = Twc_mono_soph.inverse() * Eigen::Vector3d(x3Dw_mono[k].x, x3Dw_mono[k].y, x3Dw_mono[k].z);
        Eigen::Vector2d tmp_reproj;
        camera_rgb->spaceToPlane(x3c_mono, tmp_reproj);
        reproj_mono.push_back(cv::Point2f(tmp_reproj.x(), tmp_reproj.y()));
        cv::circle(debug_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(255,0,0), 2);
        cv::circle(debug_img, uv_2d_distorted_mono[k], 2, cv::Scalar(0,0,255), 0);
      }
    }
    cv::imshow("rgb_instrinsic", debug_img);
#endif

#if 1
   //Project Xw to "Confidence image" to check instrinsic value
    Eigen::Matrix4d Twc_depth;
    std::vector<cv::Point3f> x3Dw_depth;
    std::vector<cv::Point2f> uv_2d_distorted_depth;
    std::vector<int> id_landmark_depth;
    std::vector<cv::Point2f> reproj_depth;
    isCalOk = rgbd_calibration::calcCamPoseRGBD(frame_index, depth_img, camera_depth, Twc_depth, x3Dw_depth, uv_2d_distorted_depth, id_landmark_depth);
    if (isCalOk == false)
      return;
      
    cv::Mat debug_depth_img = depth_img.clone();
    cv::cvtColor(debug_depth_img, debug_depth_img, CV_GRAY2BGR);
    Sophus::SE3d Twc_depth_soph;
    if (x3Dw_depth.size() == rgbd_calibration::row * rgbd_calibration::col)
    {
      Eigen::Matrix3d roatation_matrix = Twc_depth.block(0, 0, 3, 3).matrix();
      Eigen::Quaterniond qwc(roatation_matrix);
      Twc_depth_soph.translation() = Twc_depth.block(0, 3, 3, 1);
      Twc_depth_soph.setQuaternion(qwc);

      //Project Xw to "stereo Left image" to check instrinsic value
      for (int k = 0; k < x3Dw_depth.size(); k++)
      {
        Eigen::Vector3d x3c_depth = Twc_depth_soph.inverse() * Eigen::Vector3d(x3Dw_depth[k].x, x3Dw_depth[k].y, x3Dw_depth[k].z);
        Eigen::Vector2d tmp_reproj;
        camera_depth->spaceToPlane(x3c_depth, tmp_reproj);
        reproj_depth.push_back(cv::Point2f(tmp_reproj.x(), tmp_reproj.y()));
        cv::circle(debug_depth_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(255, 0, 0), 2);
        cv::circle(debug_depth_img, uv_2d_distorted_depth[k], 4, cv::Scalar(0,0,255), 0);
      }
    }
    cv::imshow("depth_instrinsic", debug_depth_img);

    //Project Xw from "Confidence image" to "stereo Left image" to check extrinsic
    cv::Mat ext_debug_img = mono_img.clone();
    cv::cvtColor(ext_debug_img, ext_debug_img, CV_GRAY2BGR);
    if (x3Dw_depth.size() == rgbd_calibration::row * rgbd_calibration::col)
    {
      for (int k = 0; k < x3Dw_depth.size(); k++)
      {
        Eigen::Vector3d x3c_img = Tdepth_rgb.inverse() *  Twc_depth_soph.inverse() * Eigen::Vector3d(x3Dw_depth[k].x, x3Dw_depth[k].y, x3Dw_depth[k].z);
        Eigen::Vector2d tmp_reproj;
        camera_rgb->spaceToPlane(x3c_img, tmp_reproj);
        // reproj_depth.push_back(cv::Point2f(tmp_reproj.x(), tmp_reproj.y()));
        cv::circle(ext_debug_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(0,255,0), 0);
      }
    }
    cv::imshow("prej_ext_debug_img", ext_debug_img);
#endif


    cv::waitKey(2);
    frame_index++;
  }

private:
  //threading init
  // std::unique_ptr<ThreadPool> threading_pool_opt;
  ros::Publisher depth_pub_;
  CameraPtr camera_rgb, camera_depth;
  Sophus::SE3d Tdepth_rgb;
  RGBD_CALIBRATIONPtr rgbd_calibr;
  int frame_index = 0;
  int tmp_data_size = 0;
  double obs_frame_count = 0;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgbd_chessboard_checker");
  ros::NodeHandle n;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  RGBDCheckerNode rgbd_node(n);
  message_filters::Subscriber<sensor_msgs::Image> sub_img(n, "cam0/image_raw", 100),
      sub_depth_img(n, "camera/confindence/image", 100);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
  message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10000), sub_img, sub_depth_img);
  sync.registerCallback(boost::bind(&RGBDCheckerNode::ImageDepthImgCallback, &rgbd_node, _1, _2));

  //the following three rows are to run the calibrating project through playing bag package
  // ros::Subscriber sub_imu = n.subscribe(WHEEL_TOPIC, 500, wheel_callback, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber sub_img = n.subscribe(IMAGE_RGB_TOPIC, 200, image_callback);
  // std::thread calc_thread = std::thread(&RGBDCheckerNode::calc_process, &rgbd_node);

  ros::spin();
  return 0;
}
