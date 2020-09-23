#include "rgbd_calibration.h"
#include "systrace/tracer.h"

//Note!!!
//校正板參數在calcCamPose.cpp修改
//Calibration board parameters are modified in calcCamPose.cpp.
RGBD_CALIBRATION::RGBD_CALIBRATION(CameraPtr depth_cam_, CameraPtr rgb_cam_, Sophus::SE3d &init_Tdepth_rgb)
{
    model_depth_cam = depth_cam_;
    model_rgb_cam = rgb_cam_;
    Tdepth_rgb = init_Tdepth_rgb;

    std::cout << "RGBD_CALIBRATION 1" << std::endl;
    calibration::Size2 images_size_depth, undistortion_matrix_cell_size_;
    std::cout << "RGBD_CALIBRATION 2" << std::endl;
    images_size_depth.x() = depth_cam_->imageWidth();
    images_size_depth.y() = depth_cam_->imageHeight();
    std::cout << "RGBD_CALIBRATION 3" << std::endl;
    undistortion_matrix_cell_size_.x() = 2;
    undistortion_matrix_cell_size_.y() = 2;

    local_model_ = boost::make_shared<calibration::LocalModel>(images_size_depth);
    calibration::LocalModel::Data::Ptr local_matrix = local_model_->createMatrix(undistortion_matrix_cell_size_, calibration::LocalPolynomial::IdentityCoefficients());
    std::cout << "RGBD_CALIBRATION 4" << std::endl;
    local_model_->setMatrix(local_matrix);
    std::cout << "RGBD_CALIBRATION 5" << std::endl;
    local_fit_ = boost::make_shared<calibration::LocalMatrixFitPCL>(local_model_);
}

//Note!!!
//校正板參數在calcCamPose.cpp修改
//Calibration board parameters are modified in calcCamPose.cpp.
void RGBD_CALIBRATION::Optimize(std::vector<std::tuple<cv::Mat, std::shared_ptr<PCLCloud3>, double>> &rgb_depth_time)
{
    int obs_count = 0;
    Sophus::SE3d Twc_last;
    //step1. convert to calibration board image to pose Tcw / uv / x3Dc
    for (int i = 0; i < rgb_depth_time.size(); i++)
    {
        RGBFramePtr cur_rgb_info = std::make_shared<RGBFrame>();
        Eigen::Matrix4d Twc;
        std::vector<cv::Point3f> x3Dw;
        std::vector<cv::Point2f> uv_2d_distorted;
        std::vector<int> id_landmark;
        bool isCalOk = calcCamPoseRGBD(std::get<2>(rgb_depth_time[i]), std::get<0>(rgb_depth_time[i]), model_rgb_cam, Twc, x3Dw, uv_2d_distorted, id_landmark);

        Eigen::Matrix3d roatation_matrix = Twc.block(0, 0, 3, 3).matrix();
        Eigen::Quaterniond qwc(roatation_matrix);
        Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);

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

        if(obs_count == 0){
            Twc_last.translation() = twc;
            Twc_last.setQuaternion(Eigen::Quaterniond(roatation_matrix));
            obs_count++;
        }
        else
        {
            Sophus::SE3d Twc_cur;
            Twc_cur.translation() = twc;
            Twc_cur.setQuaternion(Eigen::Quaterniond(roatation_matrix));
            Sophus::SE3d Tw1_Tw2 = Twc_last.inverse() * Twc_cur;
            Twc_last = Twc_cur;
            if (Tw1_Tw2.translation().norm() < 0.1)
            {
                bad_idx.push_back(i);
                continue;
            }
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

        cur_rgb_info->Twc.translation() = twc;
        cur_rgb_info->Twc.setQuaternion(qwc);
        rgb_frame_vec[cur_rgb_info->timestamp] = (cur_rgb_info);
        pcl_frame_vec[cur_rgb_info->timestamp] = std::get<1>(rgb_depth_time[i]);
        close_to_far.push_back(std::pair<double, double>(twc.norm(), cur_rgb_info->timestamp));
        obs_count++;
        if (obs_count > 20)
            break;
    }
    //Sorting by
    std::cout << "close_to_far size:" << close_to_far.size() <<std::endl;
    std::sort(close_to_far.begin(), close_to_far.end(), OrderByDistance());

    //step2.
    //("Estimating undistortion map...");
    EstimateLocalModel();
    //step2 undistort pcl
}

//("Estimating undistortion map...");
void RGBD_CALIBRATION::EstimateLocalModel()
{
    int max_threads_ = 2;
    Eigen::Vector3d chessboard_center(0.794, 0.794, 0.794); //x3Dcolor

    //project rgb_cam center to depth cam ( Tdepth_ro_color * x3Dcolor )
    Eigen::Vector3d proj_center_chess_to_depthCam = Tdepth_rgb * chessboard_center;

    for (size_t i = 0; i < close_to_far.size(); i += max_threads_)
    {
        slam_ros::Tracer::TraceBegin("EstimateLocalModel");
#pragma omp parallel for schedule(static, 1)
        for (int th = 0; th < max_threads_; ++th)
        {
            if (i + th >= close_to_far.size())
                continue;

            //close to far
            double timestamp = close_to_far[i + th].second;

            // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
            calibration::PlaneInfo plane_info;
            boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ>> cloud;
            cloud = make_shared_ptr(pcl_frame_vec[timestamp]);

            if (ExtractPlane(model_rgb_cam, cloud, proj_center_chess_to_depthCam, plane_info))
            {
                plane_info_map_[timestamp] = plane_info;
                std::cout << "plane_info indices=" << plane_info.indices_->size() << std::endl;
                std::vector<int> indices; // = *plane_info.indices_;
                indices.reserve(plane_info.indices_->size());
                int w = model_depth_cam->imageWidth();
                int h = model_depth_cam->imageHeight();
                for (size_t j = 0; j < plane_info.indices_->size(); ++j)
                {
                    int r = (*plane_info.indices_)[j] / w;
                    int c = (*plane_info.indices_)[j] % w;
                    if ((r - h / 2) * (r - h / 2) + (c - w / 2) * (c - w / 2) < (h / 2) * (h / 2))
                        indices.push_back((*plane_info.indices_)[j]);
                }
                calibration::Plane fitted_plane = calibration::PlaneFit<double>::fit(calibration::PCLConversion<double>::toPointMatrix(*cloud, indices));
                plane_info_map_[timestamp].plane_ = fitted_plane;

                #pragma omp critical
                {
                // local_fit_->accumulateCloud(cloud, *plane_info.indices_);
                // local_fit_->addAccumulatedPoints(fitted_plane);
                // for (Size1 c = 0; c < gt_cb.corners().elements(); ++c)
                // {
                //     const Point3 &corner = gt_cb.corners()[c];
                //     inverse_global_fit_->addPoint(0, 0, corner, fitted_plane);
                // }
                // if (i + th > 20)
                //     inverse_global_fit_->update();
                }
            }
        }
        slam_ros::Tracer::TraceEnd();
    }
}

bool RGBD_CALIBRATION::ExtractPlane(CameraPtr color_cam_model,
                                    const PCLCloud3::ConstPtr &cloud,
                                    const Eigen::Vector3d &center,
                                    calibration::PlaneInfo &plane_info)
{
    slam_ros::ScopedTrace tracer("ExtractPlane");
    //color_cb.width() = rgb_image num of rows / cols
    double radius = std::min(color_cam_model->imageWidth(), color_cam_model->imageHeight()) / 1.5; // TODO Add parameter
    calibration::PointPlaneExtraction<calibration::PCLPoint3> plane_extractor;
    plane_extractor.setInputCloud(cloud);
    plane_extractor.setRadius(radius);
    bool plane_extracted = false;

    int r[] = {0, 1, 2};         // TODO Add parameter
    int k[] = {0, 1, -1, 2, -2}; // TODO Add parameter
    for (int i = 0; i < 5 && !plane_extracted; ++i)
    {
        for (int j = 0; j < 3 && !plane_extracted; ++j)
        {
            plane_extractor.setRadius((1 + r[j]) * radius);
            plane_extractor.setPoint(calibration::PCLPoint3(center.x(), center.y(), center.z() + (1 + r[j]) * radius * k[i]));
            plane_extracted = plane_extractor.extract(plane_info);
        }
    }

    return plane_extracted;
}
