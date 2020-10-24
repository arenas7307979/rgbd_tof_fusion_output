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
// #include "two_image_calibration.h"
#define opencv_viewer 1
#define NotOny_Publish_Color 1

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
    depth_pub_ = pr_nh.advertise<sensor_msgs::Image>("/project/tof_to_left", 1);
    reproj_pub_ = pr_nh.advertise<sensor_msgs::Image>("/project/reproj_debug", 1);
    sync_stereoL_pub_ = pr_nh.advertise<sensor_msgs::Image>("/project/sync_left_gray_img", 1);
    sync_tof_mono_pub = pr_nh.advertise<sensor_msgs::Image>("/project/sync_tof_gray_img", 1);

    std::string config_file_depth, config_file_rgb;
    GetParam("/calibration/config_file_depth", config_file_depth);
    GetParam("/calibration/config_file_rgb", config_file_rgb);

    //camera parameter in config folder
    camera_depth = CameraFactory::instance()->generateCameraFromYamlFile(config_file_depth);
    camera_rgb = CameraFactory::instance()->generateCameraFromYamlFile(config_file_rgb);
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
    // Tdepth_rgb = Tdepth_rgb.inverse();
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

  //img_msg color image(stereo left)
  // depth_msg is confidence image
  // depth_range_msg is depth inforamtion.
  void ImageDepthImgCallback(const sensor_msgs::ImageConstPtr &img_msg,
                             const sensor_msgs::ImageConstPtr &depth_msg,
                             const sensor_msgs::ImageConstPtr &depth_range_msg)
  {
    if (img_msg == nullptr || depth_msg == nullptr || depth_range_msg == nullptr)
      return;

    cv::Mat mono_img, depth_img;
    double timestamp = img_msg->header.stamp.toSec();
    if (img_msg->encoding == sensor_msgs::image_encodings::BGR8 ||
        img_msg->encoding == sensor_msgs::image_encodings::RGB8 ||
        img_msg->encoding == sensor_msgs::image_encodings::BGRA8)
    {
      //rgb image
      cv::Mat rgb_img;
      if (img_msg->encoding == sensor_msgs::image_encodings::BGRA8)
      {
        rgb_img = cv_bridge::toCvShare(img_msg, "bgra8")->image;
        cv::cvtColor(rgb_img, mono_img, CV_BGRA2GRAY);
      }
      else
      {
        rgb_img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
        cv::cvtColor(rgb_img, mono_img, CV_BGR2GRAY);
      }
    }
    else if (img_msg->encoding == sensor_msgs::image_encodings::MONO8 || img_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      mono_img = cv_bridge::toCvShare(img_msg, "mono8")->image;
    }

    if (img_msg->encoding == sensor_msgs::image_encodings::BGR8 ||
        img_msg->encoding == sensor_msgs::image_encodings::RGB8 ||
        img_msg->encoding == sensor_msgs::image_encodings::BGRA8)
    {
      //rgb image
      cv::Mat rgb_img;
      if (img_msg->encoding == sensor_msgs::image_encodings::BGRA8)
      {
        rgb_img = cv_bridge::toCvShare(depth_msg, "bgra8")->image;
        cv::cvtColor(rgb_img, depth_img, CV_BGRA2GRAY);
      }
      else
      {
        rgb_img = cv_bridge::toCvShare(depth_msg, "bgr8")->image;
        cv::cvtColor(rgb_img, depth_img, CV_BGR2GRAY);
      }
    }
    else if (depth_msg->encoding == sensor_msgs::image_encodings::MONO8 || depth_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      depth_img = cv_bridge::toCvShare(depth_msg, "mono8")->image;
    }

    cv_bridge::CvImage cv_ros_stereo_gray;
    cv_ros_stereo_gray.header = img_msg->header;
    cv_ros_stereo_gray.header.frame_id = "map";
    cv_ros_stereo_gray.header.seq = 0;
    //cv_ros.header.stamp = img_msg->header; //microseconds->seconds
    cv_ros_stereo_gray.encoding = "mono8";
    cv_ros_stereo_gray.image = mono_img;
    sync_stereoL_pub_.publish(cv_ros_stereo_gray);

    cv_bridge::CvImage cv_ros_tof_gray;
    cv_ros_tof_gray.header = img_msg->header;
    cv_ros_tof_gray.header.frame_id = "map";
    cv_ros_tof_gray.header.seq = 0;
    //cv_ros.header.stamp = img_msg->header; //microseconds->seconds
    cv_ros_tof_gray.encoding = "mono8";
    cv_ros_tof_gray.image = depth_img;
    sync_tof_mono_pub.publish(cv_ros_tof_gray);


#if NotOny_Publish_Color
    //convert depth_range inforamation
    cv::Mat depth_range;
    double depth_scalin_factor = 1000.0;
    double inv_depth_scalin_factor = 1.0 / depth_scalin_factor;
    cv_bridge::CvImagePtr depth_range_Ptr;
    depth_range_Ptr = cv_bridge::toCvCopy(depth_range_msg, depth_range_msg->encoding);
    if (depth_range_Ptr->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
      (depth_range_Ptr->image).convertTo(depth_range_Ptr->image, CV_16UC1, depth_scalin_factor);
    }
    depth_range_Ptr->image.copyTo(depth_range);

    //project depth image to stereo-left cam
    cv::Mat cvDepthProjectLeftCam(cv::Size(camera_rgb->imageWidth(), camera_rgb->imageHeight()), CV_16UC1, cv::Scalar(0));

    for (size_t i = 0; i < camera_depth->imageHeight(); i++) //row rmap[0][0].size[0]
    {
      for (size_t j = 0; j < camera_depth->imageWidth(); j++) //colum rmap[0][0].size[1]
      {
        double depth = double(depth_range.at<uint16_t>(i, j) * inv_depth_scalin_factor);
        // double Xc = ((j - depthparam[2]) / depthparam[0]) * depth;
        // double Yc = ((i - depthparam[3]) / depthparam[1]) * depth;
        Eigen::Vector3d x3c_left;
        Eigen::Vector2d depth_uv(j, i);
        camera_depth->liftProjective(depth_uv, x3c_left);
        x3c_left = x3c_left * depth;
        x3c_left = Tdepth_rgb.inverse() * x3c_left;

        //reproject to stereo left-cam
        Eigen::Vector2d depth2left_uv;
        camera_rgb->spaceToPlane(x3c_left, depth2left_uv);
        int u_leftcam = int(depth2left_uv.x());
        int v_leftcam = int(depth2left_uv.y());
        if (u_leftcam < 0 || v_leftcam < 0 || v_leftcam > camera_rgb->imageHeight() || u_leftcam > camera_rgb->imageWidth())
          continue;

        cvDepthProjectLeftCam.at<uint16_t>(v_leftcam, u_leftcam) = depth_range.at<uint16_t>(i, j);
      }
    }

    cv_bridge::CvImage cv_ros;
    cv_ros.header = img_msg->header;
    cv_ros.header.frame_id = "map";
    cv_ros.header.seq = 0;
    //cv_ros.header.stamp = img_msg->header; //microseconds->seconds
    cv_ros.encoding = "mono16";
    cv_ros.image = cvDepthProjectLeftCam;
    depth_pub_.publish(cv_ros);

#if 0
    Eigen::Matrix4d Twc_mono;
    std::vector<cv::Point3f> x3Dw_mono;
    std::vector<cv::Point2f> uv_2d_distorted_mono;
    std::vector<int> id_landmark_mono;
    std::vector<cv::Point2f> reproj_mono;
    bool isCalOk;
    isCalOk = rgbd_calibration::calcCamPoseRGBD(frame_index, mono_img, camera_rgb, Twc_mono, x3Dw_mono, uv_2d_distorted_mono, id_landmark_mono);
    if (isCalOk == false)
      return;

    cv::Mat debug_img = mono_img.clone();
    cv::cvtColor(debug_img, debug_img, CV_GRAY2BGR);

    Sophus::SE3d Twc_mono_soph;
    if (x3Dw_mono.size() == rgbd_calibration::row * rgbd_calibration::col)
    {
      Eigen::Matrix3d roatation_matrix = Twc_mono.block(0, 0, 3, 3).matrix();
      Eigen::Quaterniond qwc(roatation_matrix);
      Twc_mono_soph.translation() = Twc_mono.block(0, 3, 3, 1);
      Twc_mono_soph.setQuaternion(qwc);

      //Project Xw to "stereo Left image" to check instrinsic value
      for (int k = 0; k < x3Dw_mono.size(); k++)
      {
        Eigen::Vector3d x3c_mono = Twc_mono_soph.inverse() * Eigen::Vector3d(x3Dw_mono[k].x, x3Dw_mono[k].y, x3Dw_mono[k].z);
        Eigen::Vector2d tmp_reproj;
        camera_rgb->spaceToPlane(x3c_mono, tmp_reproj);
        reproj_mono.push_back(cv::Point2f(tmp_reproj.x(), tmp_reproj.y()));
        cv::circle(debug_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(255, 0, 0), 2);
        cv::circle(debug_img, uv_2d_distorted_mono[k], 2, cv::Scalar(0, 0, 255), 0);
      }
    }
    // cv::imshow("rgb_instrinsic", debug_img);
#endif

#if opencv_viewer
    //Project Xw to "Confidence image" to check instrinsic value
    Eigen::Matrix4d Twc_depth;
    std::vector<cv::Point3f> x3Dw_depth;
    std::vector<cv::Point2f> uv_2d_distorted_depth;
    std::vector<int> id_landmark_depth;
    std::vector<cv::Point2f> reproj_depth;
    bool isCalOk = rgbd_calibration::calcCamPoseRGBD(frame_index, depth_img, camera_depth, Twc_depth, x3Dw_depth, uv_2d_distorted_depth, id_landmark_depth);
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
        if (tmp_reproj.x() < 0 || tmp_reproj.y() < 0 || tmp_reproj.x() > debug_depth_img.cols || tmp_reproj.y() > debug_depth_img.rows)
          continue;

        cv::circle(debug_depth_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(255, 0, 0), 2);
        cv::circle(debug_depth_img, uv_2d_distorted_depth[k], 4, cv::Scalar(0, 0, 255), 0);
      }
    }
    // cv::imshow("depth_instrinsic", debug_depth_img);

   double rgb_fx =  269.3954642460685;
   double rgb_fy =  269.80331670442996;
   double rgb_cx =  339.7385948862712;
   double rgb_cy =  195.15237910597025;

    //Project Xw from "Confidence image" to "stereo Left image" to check extrinsic
    cv::Mat ext_debug_img = mono_img.clone();
    cv::cvtColor(ext_debug_img, ext_debug_img, CV_GRAY2BGR);
    if (x3Dw_depth.size() == rgbd_calibration::row * rgbd_calibration::col)
    {
      for (int k = 0; k < x3Dw_depth.size(); k++)
      {
        Eigen::Vector3d x3c_img = Tdepth_rgb.inverse() * Twc_depth_soph.inverse() * Eigen::Vector3d(x3Dw_depth[k].x, x3Dw_depth[k].y, x3Dw_depth[k].z);
        Eigen::Vector2d tmp_reproj;
        tmp_reproj.x() = rgb_fx  * (x3c_img.x() / x3c_img.z()) + rgb_cx;
        tmp_reproj.y() = rgb_fy  * (x3c_img.y() / x3c_img.z()) + rgb_cy;
        // camera_rgb->spaceToPlane(x3c_img, tmp_reproj);
        if (tmp_reproj.x() < 0 || tmp_reproj.y() < 0 || tmp_reproj.x() > ext_debug_img.cols || tmp_reproj.y() > ext_debug_img.rows)
          continue;
        cv::circle(ext_debug_img, cv::Point2f(tmp_reproj.x(), tmp_reproj.y()), 2, cv::Scalar(0, 255, 0), 0);
      }
    }
    cv_bridge::CvImage cv_ros_reproj;
    cv_ros_reproj.header = img_msg->header;
    cv_ros_reproj.header.frame_id = "map";
    cv_ros_reproj.header.seq = 0;
    //cv_ros.header.stamp = img_msg->header; //microseconds->seconds
    cv_ros_reproj.encoding = "bgr8";
    cv_ros_reproj.image = ext_debug_img;
    reproj_pub_.publish(cv_ros_reproj);

    // cv::imshow("prej_ext_debug_img", ext_debug_img);
    // cv::waitKey(1);
#endif

#endif
    frame_index++;
  }

private:
  //threading init
  // std::unique_ptr<ThreadPool> threading_pool_opt;
  ros::Publisher depth_pub_, reproj_pub_, sync_stereoL_pub_, sync_tof_mono_pub;
  CameraPtr camera_rgb, camera_depth;
  Sophus::SE3d Tdepth_rgb;
  RGBD_CALIBRATIONPtr rgbd_calibr;
  int frame_index = 0;
  int tmp_data_size = 0;
  double obs_frame_count = 0;
  std::vector<double> depthparam, rgb_param;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgbd_chessboard_checker");
  ros::NodeHandle n;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  RGBDCheckerNode rgbd_node(n);
  message_filters::Subscriber<sensor_msgs::Image> sub_img(n, "cam0/image_raw", 100),
      sub_confidence_img(n, "camera/confindence/image", 100),
      sub_depth_img(n, "camera/depth/image", 100);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
  message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10000), sub_img, sub_confidence_img, sub_depth_img);
  sync.registerCallback(boost::bind(&RGBDCheckerNode::ImageDepthImgCallback, &rgbd_node, _1, _2, _3));

  //the following three rows are to run the calibrating project through playing bag package
  // ros::Subscriber sub_imu = n.subscribe(WHEEL_TOPIC, 500, wheel_callback, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber sub_img = n.subscribe(IMAGE_RGB_TOPIC, 200, image_callback);
  // std::thread calc_thread = std::thread(&RGBDCheckerNode::calc_process, &rgbd_node);

  ros::spin();
  return 0;
}
