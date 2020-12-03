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
#define opencv_viewer 1

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
    depth_pub_ = pr_nh.advertise<sensor_msgs::Image>("/project/depth_to_stereo", 10);

    std::string config_file_rgb_l, config_file_rgb_r;
    GetParam("/calibration/config_file_rgb_l", config_file_rgb_l);
    GetParam("/calibration/config_file_rgb_r", config_file_rgb_r);

    //camera parameter in config folder
    // camera_depth = CameraFactory::instance()->generateCameraFromYamlFile(config_file_depth);
    camera_rgb_l = CameraFactory::instance()->generateCameraFromYamlFile(config_file_rgb_l);
    camera_rgb_r = CameraFactory::instance()->generateCameraFromYamlFile(config_file_rgb_r);

    //get intrinsic parameter
    rgb_l_param = camera_rgb_l->getK();
    rgb_r_param = camera_rgb_r->getK();
    for (size_t i = 0; i < rgb_l_param.size(); i++)
    {
      std::cout << "rgb_l_param[i] : " << rgb_l_param[i] << std::endl;
      std::cout << "rgb_r_param[i] : " << rgb_r_param[i] << std::endl;
    }

    //Get initial Tdepth_rgb extrinsic param
    // double qw, qx, qy, qz;
    // double x, y, z;
    // GetParam("/Tdepth_rgb/tx", x);
    // GetParam("/Tdepth_rgb/ty", y);
    // GetParam("/Tdepth_rgb/tz", z);
    // GetParam("/Tdepth_rgb/qx", qx);
    // GetParam("/Tdepth_rgb/qy", qy);
    // GetParam("/Tdepth_rgb/qz", qz);
    // GetParam("/Tdepth_rgb/qw", qw);
    // Tdepth_rgb.translation().x() = x;
    // Tdepth_rgb.translation().y() = y;
    // Tdepth_rgb.translation().z() = z;
    // Tdepth_rgb.setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    // std::cout << "Tdepth_rgb translation: " << Tdepth_rgb.translation() << std::endl;
    // std::cout << "Tdepth_unit_quaternion: " << Tdepth_rgb.so3().unit_quaternion().coeffs() << std::endl;
    // Tdepth_rgb = Tdepth_rgb.inverse();
    // std::cout << "Tdepth_rgb= " << Tdepth_rgb.translation() <<std::endl;
    // std::cout << "Tdepth_rgb= " << Tdepth_rgb.unit_quaternion().coeffs() <<std::endl;

    //Note!!!
    //校正板參數在calcCamPose.cpp修改
    //Calibration board parameters are modified in calcCamPose.cpp.
    // rgbd_calibr = std::make_shared<RGBD_CALIBRATION>(camera_depth, camera_rgb, Tdepth_rgb);
  }
  ~RGBDCheckerNode() {}

  void StereoImgCallback(const sensor_msgs::ImageConstPtr &img_l_msg,
                             const sensor_msgs::ImageConstPtr &img_r_msg, 
                             const sensor_msgs::ImageConstPtr &depth_msg)
  {
    if (img_l_msg == nullptr || img_r_msg == nullptr || depth_msg == nullptr) {
      std::cout << "one of msg is null" << std::endl;
      return;
    }

    cv::Mat mono_l_img, mono_r_img, depth_img;
    // double timestamp = img_l_msg->header.stamp.toSec();
    if (img_l_msg->encoding == sensor_msgs::image_encodings::BGR8 || 
    img_l_msg->encoding == sensor_msgs::image_encodings::RGB8 || 
    img_l_msg->encoding == sensor_msgs::image_encodings::BGRA8)
    {
      //rgb image
      cv::Mat rgb_img;
      rgb_img = cv_bridge::toCvShare(img_l_msg, "bgr8")->image;
      cv::cvtColor(rgb_img, mono_l_img, CV_BGR2GRAY);
    }
    else if (img_l_msg->encoding == sensor_msgs::image_encodings::MONO8 || img_l_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      mono_l_img = cv_bridge::toCvShare(img_l_msg, "mono8")->image;
    }
    
    if (img_r_msg->encoding == sensor_msgs::image_encodings::BGR8 || 
    img_r_msg->encoding == sensor_msgs::image_encodings::RGB8 || 
    img_r_msg->encoding == sensor_msgs::image_encodings::BGRA8)
    {
      //rgb image
      cv::Mat rgb_img;
      rgb_img = cv_bridge::toCvShare(img_r_msg, "bgr8")->image;
      cv::cvtColor(rgb_img, mono_r_img, CV_BGR2GRAY);
    }
    else if (img_r_msg->encoding == sensor_msgs::image_encodings::MONO8 || img_r_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      mono_r_img = cv_bridge::toCvShare(img_r_msg, "mono8")->image;
    }
    
    //convert depth inforamation
    cv::Mat depth_range;
    double depth_scalin_factor = 1000.0;
    double inv_depth_scalin_factor = 1.0 / depth_scalin_factor;
    cv_bridge::CvImagePtr depth_range_Ptr;
    depth_range_Ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    if(depth_range_Ptr->encoding == sensor_msgs::image_encodings::TYPE_32FC1){
      (depth_range_Ptr->image).convertTo(depth_range_Ptr->image, CV_16UC1, depth_scalin_factor);
    }
    depth_range_Ptr->image.copyTo(depth_range);
    
    //project depth image to stereo-left cam
    cv::Mat cvDepthProjectLeftCam(cv::Size(camera_rgb_l->imageWidth(), camera_rgb_l->imageHeight()), CV_16UC1, cv::Scalar(0));

    for (size_t i = 0; i < camera_rgb_l->imageHeight(); i++) //row rmap[0][0].size[0]
    {
      for (size_t j = 0; j < camera_rgb_l->imageWidth(); j++) //colum rmap[0][0].size[1]
      {
        double depth = double(depth_range.at<uint16_t>(i, j) * inv_depth_scalin_factor);
        double Xc = ((j - depthparam[2]) / depthparam[0]) * depth;
        double Yc = ((i - depthparam[3]) / depthparam[1]) * depth;
        Eigen::Vector3d x3c_left = Tdepth_rgb.inverse() * Eigen::Vector3d(Xc, Yc, depth);

        //reproject to stereo left-cam
        Eigen::Vector2d depth2left_uv;
        camera_rgb_l->spaceToPlane(x3c_left, depth2left_uv);
        int u_leftcam = int(depth2left_uv.x());
        int v_leftcam = int(depth2left_uv.y());
        if (u_leftcam < 0 || v_leftcam < 0 || v_leftcam > camera_rgb_l->imageHeight()|| u_leftcam > camera_rgb_l->imageWidth())
          continue;

        cvDepthProjectLeftCam.at<uint16_t>(v_leftcam, u_leftcam) = depth_range.at<uint16_t>(i, j);
      }
    }
#if 0
    cv_bridge::CvImage cv_ros;
    cv_ros.header = img_l_msg->header;
    cv_ros.header.frame_id = "map";
    cv_ros.header.seq = 0;
    cv_ros.encoding = "mono16";
    cv_ros.image = cvDepthProjectLeftCam;
    depth_pub_.publish(cv_ros);
#endif
    /* left image */
    Eigen::Matrix4d Twc_mono;
    std::vector<cv::Point3f> x3Dw_mono;
    std::vector<cv::Point2f> uv_2d_distorted_mono;
    std::vector<int> id_landmark_mono;
    std::vector<cv::Point2f> reproj_mono;
    bool isCalOk;
    // isCalOk = rgbd_calibration::calcCamPoseRGBD(frame_index, mono_img, camera_rgb, Twc_mono, x3Dw_mono, uv_2d_distorted_mono, id_landmark_mono);
    isCalOk = rgbd_calibration::calcCamPoseApril(frame_index, mono_l_img, camera_rgb_l, Twc_mono, x3Dw_mono, uv_2d_distorted_mono, id_landmark_mono);
    if (isCalOk == false) {
      std::cout << "Left rgb cal failed" << std::endl;
      return;
    }

    cv::Mat debug_img = mono_l_img.clone();
    cv::cvtColor(debug_img, debug_img, CV_GRAY2BGR);

    Sophus::SE3d Twc_mono_soph;    
    if (uv_2d_distorted_mono.size() > 20) {
        Eigen::Matrix3d roatation_matrix = Twc_mono.block(0, 0, 3, 3).matrix();
        Eigen::Quaterniond qwc(roatation_matrix);
        Twc_mono_soph.translation()= Twc_mono.block(0, 3, 3, 1);
        Twc_mono_soph.setQuaternion(qwc);
        for (int k = 0; k < x3Dw_mono.size(); k++)
        {
          Eigen::Vector3d x3c_mono = Twc_mono_soph.inverse() * Eigen::Vector3d(x3Dw_mono[k].x, x3Dw_mono[k].y, x3Dw_mono[k].z);
          Eigen::Vector2d tmp_reproj;
          camera_rgb_l->spaceToPlane(x3c_mono, tmp_reproj);
          reproj_mono.push_back(cv::Point2f(tmp_reproj.x(), tmp_reproj.y()));
          cv::circle(debug_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(255,0,0), 2);
          cv::circle(debug_img, uv_2d_distorted_mono[k], 2, cv::Scalar(0,0,255), 0);
        }
    }
    else {
      std::cout << "Left rgb 2d distorted not enough" << std::endl;
    }

#if opencv_viewer
    cv::imshow("rgb_instrinsic_l", debug_img);
    cv::waitKey(1);
#endif

    frame_index++;
  }

private:
  //threading init
  // std::unique_ptr<ThreadPool> threading_pool_opt;
  ros::Publisher depth_pub_;
  CameraPtr camera_rgb_l, camera_rgb_r;//, camera_depth;
  Sophus::SE3d Tdepth_rgb;
  RGBD_CALIBRATIONPtr rgbd_calibr;
  int frame_index = 0;
  int tmp_data_size = 0;
  double obs_frame_count = 0;
  std::vector<double> depthparam, rgb_l_param, rgb_r_param;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "stereo_april_check");
  ros::NodeHandle nh;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  RGBDCheckerNode rgbd_node(nh);

  message_filters::Subscriber<sensor_msgs::Image> sub_img_l(nh, "/cam0/image_raw", 100),
      sub_img_r(nh, "/cam1/image_raw", 100),
      sub_depth_img(nh, "/camera/depth/image_raw", 100);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
  message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10000), sub_img_l, sub_img_r, sub_depth_img);
  sync.registerCallback(boost::bind(&RGBDCheckerNode::StereoImgCallback, &rgbd_node, _1, _2, _3));
  
  //the following three rows are to run the calibrating project through playing bag package
  // ros::Subscriber sub_imu = nh.subscribe(WHEEL_TOPIC, 500, wheel_callback, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber sub_img0 = nh.subscribe("/cam0/image_raw", 200, &image_callback0);
  // ros::Subscriber sub_img1 = nh.subscribe("/cam1/image_raw", 200, &image_callback1);
  // ros::Subscriber sub_img2 = nh.subscribe("/camera/depth/image_raw", 200, &image_callback2);
  // std::thread calc_thread = std::thread(&RGBDCheckerNode::calc_process, &rgbd_node);

  ros::spin();
  return 0;
}
