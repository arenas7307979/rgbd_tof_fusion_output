/*******************************************************
 * Copyright (C) 2019, SLAM Group, Megvii-R
 *******************************************************/

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


// PCL
typedef pcl::PointCloud<pcl::PointXYZ>  PCLCloud3;       ///< 3D pcl PointCloud.

std::vector<std::tuple<cv::Mat, std::shared_ptr<PCLCloud3>, double>> rgb_depth_time;
std::mutex m_buf;
bool hasImg = false; //zdf


//record the first frame calculated successfully
bool fisrt_frame = true;
Eigen::Matrix3d Rwc0;
Eigen::Vector3d twc0;
//decide if the frequent is decreased
bool halfFreq = false;
int frame_index = 0;

template <class T>
void GetParam(const std::string &param_name, T &param)
{
  if (!ros::param::get(param_name, param))
  {
    ROS_ERROR_STREAM("Can not find \"" << param_name << "\" in ROS Parameter Server");
    ros::shutdown();
  }
}

class RGBDCalibrationNode
{
public:
  RGBDCalibrationNode(ros::NodeHandle &pr_nh)
  {
    depth_pub_ = pr_nh.advertise<sensor_msgs::PointCloud2>("/calibration/depth_cloud", 10);

    std::string config_file_depth, config_file_rgb;
    GetParam("/calibration/config_file_depth", config_file_depth);
    GetParam("/calibration/config_file_rgb", config_file_rgb);

    //camera parameter in config folder
    camera_depth = CameraFactory::instance()->generateCameraFromYamlFile(config_file_depth);
    camera_rgb =  CameraFactory::instance()->generateCameraFromYamlFile(config_file_rgb);
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
    Sophus::SE3d Tdepth_rgb;
    Tdepth_rgb.translation().x() = x;
    Tdepth_rgb.translation().y() = y;
    Tdepth_rgb.translation().z() = z;
    Tdepth_rgb.setQuaternion(Eigen::Quaterniond(qw, qx, qy, qz));
    std::cout << "Tdepth_rgb translation: " << Tdepth_rgb.translation() <<std::endl;
    std::cout << "Tdepth_unit_quaternion: " << Tdepth_rgb.so3().unit_quaternion().coeffs() <<std::endl;

    //Note!!!
    //校正板參數在calcCamPose.cpp修改
    //Calibration board parameters are modified in calcCamPose.cpp.
    rgbd_calibr = std::make_shared<RGBD_CALIBRATION>(camera_depth, camera_rgb, Tdepth_rgb);

  }
  ~RGBDCalibrationNode() {}

  void ImageDepthImgCallback(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &depth_msg)
  {
    frame_index++;
    m_buf.lock();

    bool color_image = false;

    //decode rgb or gray image
    cv::Mat mono_img, depth_img, rgb_img;
    double timestamp = img_msg->header.stamp.toSec();
    if (img_msg->encoding == sensor_msgs::image_encodings::BGR8 || img_msg->encoding == sensor_msgs::image_encodings::RGB8)
    {
      color_image = true;
      //rgb image
      rgb_img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
      cv::cvtColor(rgb_img, mono_img, CV_BGR2GRAY);
    }
    else if(img_msg->encoding == sensor_msgs::image_encodings::MONO8 || img_msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
    {
      //gray image
      cv::Mat mono_img = cv_bridge::toCvShare(img_msg, "mono")->image.clone();
    }
    else
    {
      std::cout << "not supporting image type" << std::endl;
      exit(-1);
    }


    cv::Mat bgr[3]; //destination array
    if (color_image == true)
    {
      cv::split(rgb_img, bgr); //split source
    }
    else
    {
      bgr[0] = mono_img;
      bgr[1] = mono_img;
      bgr[2] = mono_img;
    }
    //decode depth images
    //convert to image to gray and depth image to TYPE_16UC1
    double depth_scaling_factor_ = 1000.0;
    double inv_depth_scaling_factor =  0.001;
    /* get depth image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
      (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, depth_scaling_factor_);
    }
    cv_ptr->image.copyTo(depth_img);
    std::shared_ptr<PCLCloud3> cloud = std::make_shared<PCLCloud3>(camera_depth->imageWidth(), camera_depth->imageHeight());

    pcl::PointCloud<pcl::PointXYZRGB> cloud_calibration;


    int pcl_num = 0;
    //convert depth img to pcl type
    for (int j = 0; j < camera_depth->imageHeight(); ++j)
    {
      for (int k = 0; k < camera_depth->imageWidth(); ++k)
      {
        float depth = (depth_img.at<uint16_t>(j, k)) * inv_depth_scaling_factor; //(y,x) convert to meter
        
        if (depth < 0.05)
          depth = 0;
        else
          pcl_num++;

        Eigen::Vector3d Pc;
        camera_depth->liftProjective(Eigen::Vector2d(k, j), Pc);
        Pc = Pc * depth;

        cloud->at(k, j).x = Pc.x(); //Xc in depth cam
        cloud->at(k, j).y = Pc.y(); //Yc in depth cam
        cloud->at(k, j).z = Pc.z(); //Zc in depth cam

        pcl::PointXYZRGB pt;
        pt.x = Pc.x();
        pt.y = Pc.y();
        pt.z = Pc.z();
        pt.b = bgr[0].at<uchar>(j, k);
        pt.g = bgr[1].at<uchar>(j, k);
        pt.r = bgr[2].at<uchar>(j, k);
        cloud_calibration.push_back(pt);
      }
    }

    cloud_calibration.width = cloud_calibration.points.size();
    cloud_calibration.height = 1;
    cloud_calibration.is_dense = true;
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud_calibration, cloud_msg);
    cloud_msg.header.frame_id = "map";
    depth_pub_.publish(cloud_msg);

    if (pcl_num < 50)
    {
      m_buf.unlock();
      return;
    }

    std::cout << "cloud->at(k, j).z =" << cloud->at(camera_depth->imageWidth()*0.5, camera_depth->imageHeight()*0.5).z  <<std::endl;
    rgb_depth_time.push_back(std::tuple<cv::Mat,  std::shared_ptr<PCLCloud3>, double>(mono_img, cloud, obs_frame_count));
    obs_frame_count++;
    m_buf.unlock();
  }

  // extract images with same timestamp from two topics
  void calc_process()
  {
#if 1
    double t_last = 0.0; //time of last image
    bool first = true;   //judge if last frame was calculated successfully

    while (1)
    {
      std::chrono::milliseconds dura(4000);
      std::this_thread::sleep_for(dura);

      m_buf.lock();
      int cur_data_size = rgb_depth_time.size();
      m_buf.unlock();

      // std::cout << "cur_data_size:" << cur_data_size <<std::endl;

      if (cur_data_size > 30 && first == true)
      {
        tmp_data_size = cur_data_size;
        first = false;
        continue;
      }
      if(first == true)
        continue;

      if (first == false && cur_data_size > tmp_data_size)
      {  
        tmp_data_size = cur_data_size;
        continue;
      }
      else if (first == false && cur_data_size == tmp_data_size)
      {
        //run optimize
        std::cout << "====optimize====" << std::endl;
        std::cout << "rgb_depth_time.size()=" << cur_data_size << std::endl;

        m_buf.lock();
        //implemnet optimize
        rgbd_calibr->Optimize(rgb_depth_time);
        m_buf.unlock();

        return;
      }

      // cv::Mat image;
      // bool isCalOk = calcCamPose(time, image, camera_rgb, Twc);
#endif
    }
  };

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
  ros::init(argc, argv, "rgbd_calibration");
  ros::NodeHandle n;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  RGBDCalibrationNode rgbd_node(n);
  message_filters::Subscriber<sensor_msgs::Image> sub_img(n, "/cam0/image_raw", 100), 
  sub_depth_img(n, "camera/depth/image_raw", 100);
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
  message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10000), sub_img, sub_depth_img);
  sync.registerCallback(boost::bind(&RGBDCalibrationNode::ImageDepthImgCallback, &rgbd_node, _1, _2));

  //the following three rows are to run the calibrating project through playing bag package
  // ros::Subscriber sub_imu = n.subscribe(WHEEL_TOPIC, 500, wheel_callback, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber sub_img = n.subscribe(IMAGE_RGB_TOPIC, 200, image_callback);
  std::thread calc_thread = std::thread(&RGBDCalibrationNode::calc_process, &rgbd_node);

  ros::spin();
  return 0;
}
