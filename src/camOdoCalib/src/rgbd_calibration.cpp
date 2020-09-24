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

    calibration::Size2 images_size_depth, undistortion_matrix_cell_size_;
    images_size_depth.x() = depth_cam_->imageWidth();
    images_size_depth.y() = depth_cam_->imageHeight();
    undistortion_matrix_cell_size_.x() = 2;
    undistortion_matrix_cell_size_.y() = 2;

    depth_error_function_ = calibration::Polynomial<double, 2>(Eigen::Vector3d(1, 0, 0));

    //TODO:: introduce
    local_model_ = boost::make_shared<calibration::LocalModel>(images_size_depth);
    calibration::LocalModel::Data::Ptr local_matrix = local_model_->createMatrix(undistortion_matrix_cell_size_, calibration::LocalPolynomial::IdentityCoefficients());
    local_model_->setMatrix(local_matrix);
    local_matrix_ = boost::make_shared<calibration::LocalMatrixPCL>(local_model_);

    //TODO:: introduce
    global_model_ = boost::make_shared<calibration::GlobalModel>(images_size_depth);
    calibration::GlobalModel::Data::Ptr global_data = boost::make_shared<calibration::GlobalModel::Data>(calibration::Size2(2, 2), calibration::GlobalPolynomial::IdentityCoefficients());
    global_model_->setMatrix(global_data);
    global_matrix_ = boost::make_shared<calibration::GlobalMatrixPCL>(global_model_);

    //TODO:: introduce
    local_fit_ = boost::make_shared<calibration::LocalMatrixFitPCL>(local_model_);
    local_fit_ ->setDepthErrorFunction(depth_error_function_);
    
    global_fit_ = boost::make_shared<calibration::GlobalMatrixFitPCL>(global_model_);
    global_fit_->setDepthErrorFunction(depth_error_function_);
    calibration::InverseGlobalModel::Data::Ptr matrix = boost::make_shared<calibration::InverseGlobalModel::Data>( calibration::Size2(1, 1), calibration::InverseGlobalPolynomial::IdentityCoefficients() );
    inverse_global_model_ = boost::make_shared<calibration::InverseGlobalModel>(global_model_->imageSize());
    inverse_global_model_->setMatrix(matrix);
    inverse_global_fit_ = boost::make_shared<calibration::InverseGlobalMatrixFitEigen>(inverse_global_model_);
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
        if (obs_count > 30)
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
    int max_threads_ = 4;
    Eigen::Vector3d chessboard_center(0.794, 0.794, 0.794); //x3Dw_color
    bool global_update = false;
    bool local_update = false;

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

            //project rgb_cam center to depth cam ( Tdepth_ro_color * Tcw * x3Dw_color )
            Eigen::Vector3d proj_center_chess_to_depthCam = Tdepth_rgb * rgb_frame_vec[timestamp]->Twc.inverse() * chessboard_center;

            // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
            calibration::PlaneInfo plane_info;
            boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> und_cloud;
            und_cloud = make_shared_ptr(pcl_frame_vec[timestamp]);

#pragma omp critical
            if (global_update)
            {
                calibration::Point3 und_color_cb_center_tmp;
                und_color_cb_center_tmp.x() = proj_center_chess_to_depthCam.x();
                und_color_cb_center_tmp.y() = proj_center_chess_to_depthCam.y();
                und_color_cb_center_tmp.z() = proj_center_chess_to_depthCam.z();
                calibration::InverseGlobalMatrixEigen inverse_global(inverse_global_fit_->model());
                inverse_global.undistort(0, 0, und_color_cb_center_tmp);
                proj_center_chess_to_depthCam.x() = und_color_cb_center_tmp.x();
                proj_center_chess_to_depthCam.y() = und_color_cb_center_tmp.y();
                proj_center_chess_to_depthCam.z() = und_color_cb_center_tmp.z();
            }

#pragma omp critical
            if (local_update)
            {
                calibration::LocalMatrixPCL local(local_fit_->model());
                local.undistort(*und_cloud);
            }

            if (ExtractPlane(model_rgb_cam, und_cloud, proj_center_chess_to_depthCam, plane_info))
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
                calibration::Plane fitted_plane = calibration::PlaneFit<double>::fit(calibration::PCLConversion<double>::toPointMatrix(*pcl_frame_vec[timestamp], indices));
                plane_info_map_[timestamp].plane_ = fitted_plane;
#pragma omp critical
                {
                    local_fit_->accumulateCloud(*pcl_frame_vec[timestamp], *plane_info.indices_);
                    local_fit_->addAccumulatedPoints(fitted_plane);

                    //Due to Td_c extrinsic parameters error, not every points from calibration board projected onto 
                    //depth camera coordinates is not on the pcl fitting plane, 
                    //so we create a polynomial to correct this problem 
                    //next time to correct the color camera center when it projected to depth camera coordinat 
                    //inverse_global_fit_ is second order az^2 + bz+ c.
                    for (int k = 0; k < rgb_frame_vec[timestamp]->x3Dw.size(); k++)
                    {
                        auto &point_Xw = rgb_frame_vec[timestamp]->x3Dw[k];
                        Eigen::Vector3d Xw(point_Xw.x, point_Xw.y, point_Xw.z);
                        Eigen::Vector3d obs_proj_check_to_depth = Tdepth_rgb * rgb_frame_vec[timestamp]->Twc.inverse() * Xw;
                        const calibration::Point3 &corner = calibration::Point3(obs_proj_check_to_depth.x(),
                                                                                obs_proj_check_to_depth.y(),
                                                                                obs_proj_check_to_depth.z());
                        inverse_global_fit_->addPoint(0, 0, corner, fitted_plane);
                    }
                    if (i + th > 20){
                        inverse_global_fit_->update();
                        // std::cout << "inverse_global_fit_ params:"<< inverse_global_fit_->
                    }
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

#if 0
    int r[] = {0, 1, 2};         // TODO Add parameter
    int k[] = {0, 1, -1, 2, -2}; // TODO Add parameter
    for (int i = 0; i < 5 ; ++i)
    {
        for (int j = 0; j < 3 ; ++j)
        {
            plane_extractor.setRadius((1 + r[j]) * radius);
            plane_extractor.setPoint(calibration::PCLPoint3(center.x(), center.y(), center.z() + (1 + r[j]) * radius * k[i]));
            plane_extracted = plane_extractor.extract(plane_info);
        }
    }
#else
    plane_extractor.setPoint(calibration::PCLPoint3(center.x(), center.y(), center.z()));
    plane_extracted = plane_extractor.extract(plane_info);
#endif

    return plane_extracted;
}
