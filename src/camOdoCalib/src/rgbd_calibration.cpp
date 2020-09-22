#include "rgbd_calibration.h"

//Note!!!
//校正板參數在calcCamPose.cpp修改
//Calibration board parameters are modified in calcCamPose.cpp.
RGBD_CALIBRATION::RGBD_CALIBRATION(CameraPtr depth_cam_, CameraPtr rgb_cam_, Sophus::SE3d &init_Tdepth_rgb)
{
    model_depth_cam = depth_cam_;
    model_rgb_cam = rgb_cam_;
    Tdepth_rgb = init_Tdepth_rgb;
}

//Note!!!
//校正板參數在calcCamPose.cpp修改
//Calibration board parameters are modified in calcCamPose.cpp.
void RGBD_CALIBRATION::Optimize(std::vector<std::tuple<cv::Mat, std::shared_ptr<PCLCloud3>, double>> &rgb_depth_time)
{
    //step1. convert to calibration board image to pose Tcw / uv / x3Dc
    for (int i = 0; i < rgb_depth_time.size(); i++)
    {
        RGBFramePtr cur_rgb_info = std::make_shared<RGBFrame>();
        Eigen::Matrix4d Twc;
        std::vector<cv::Point3f> x3Dw;
        std::vector<cv::Point2f> uv_2d_distorted;
        std::vector<int> id_landmark;
        bool isCalOk = calcCamPoseRGBD(std::get<2>(rgb_depth_time[i]), std::get<0>(rgb_depth_time[i]), model_rgb_cam, Twc, x3Dw, uv_2d_distorted, id_landmark);

        int center_count = 0; //4 represent find center (20, 21, 14, 15)
        for (int k = 0; k < id_landmark.size(); k++)
        {
            if (id_landmark[k] == 20 || id_landmark[k] == 14 || id_landmark[k] == 15 || id_landmark[k] == 21)
                center_count++;
        }

        if (isCalOk == false || center_count != 4)
        {
            //not find 4 block of checkboard center or estimate pose failure
            //center of x3Dc(0.794, 0.794)
            bad_idx.push_back(i);
            continue;
        }

        //check id has 
        cur_rgb_info->timestamp = std::get<2>(rgb_depth_time[i]);
        cur_rgb_info->uv_distorted = uv_2d_distorted;
        cur_rgb_info->x3Dw = x3Dw;
        cur_rgb_info->id_landmark = id_landmark; // 1 landmark_id is consists of 4 uv_point; 22 id_landmark : 88 uv_distorted point 

        //std::cout << "cur_rgb_info->id_landmark size=" << cur_rgb_info->id_landmark.size() << std::endl;
        //std::cout << "cur_rgb_info->x3Dw size=" << cur_rgb_info->x3Dw.size() << std::endl;
        //std::cout << "cur_rgb_info->uv_distorted size=" << cur_rgb_info->uv_distorted.size() << std::endl;
        //exit(-1);

        Eigen::Matrix3d roatation_matrix = Twc.block(0, 0, 3, 3).matrix();
        Eigen::Quaterniond qwc(roatation_matrix);
        Eigen::Vector3d twc = Twc.block(0,3,3,1);
        cur_rgb_info->Twc.translation() = twc;
        cur_rgb_info->Twc.setQuaternion(qwc);
        rgb_frame_vec[cur_rgb_info->timestamp] = (cur_rgb_info);
        pcl_frame_vec[cur_rgb_info->timestamp] = std::get<1>(rgb_depth_time[i]);
        close_to_far.push_back(std::pair<double, double>(twc.norm(), cur_rgb_info->timestamp));
    }
    std::sort(close_to_far.begin(), close_to_far.end(), OrderByDistance());

    //step2.
    //("Estimating undistortion map...");
    EstimateLocalModel();
    std::cout << "rgb_frame_vec size:" << rgb_frame_vec.size() <<std::endl;
    std::cout << "pcl_frame_vec size:" << pcl_frame_vec.size() <<std::endl;
    //step2 undistort pcl
}

//("Estimating undistortion map...");
void RGBD_CALIBRATION::EstimateLocalModel()
{
    std::cout << "close_to_far.size() size:" << close_to_far.size() <<std::endl;
    Eigen::Vector3d chessboard_center(0.794, 0.794, 0.794); //x3Dcolor
    for (size_t i = 0; i < close_to_far.size(); i++)
    {
        //close to far
        double timestamp = close_to_far[i].second;

        //project rgb_cam center to depth cam ( Tdepth_ro_color * x3Dcolor )
        Eigen::Vector3d proj_center_chess_to_depthCam = Tdepth_rgb * chessboard_center;

        // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
        calibration::PlaneInfo plane_info;
        // if (extractPlane(gt_cb, und_cloud, und_color_cb_center, plane_info))
        // {
        // }
    }
    


}
