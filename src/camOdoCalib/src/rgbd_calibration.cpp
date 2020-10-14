#include "rgbd_calibration.h"
#include "systrace/tracer.h"


//Note!!!
//校正板參數在calcCamPose.h修改
//Calibration board parameters are modified in calcCamPose.h.
RGBD_CALIBRATION::RGBD_CALIBRATION(CameraPtr depth_cam_, CameraPtr rgb_cam_, Sophus::SE3d &init_Tdepth_rgb)
{
    model_depth_cam = depth_cam_;
    model_rgb_cam = rgb_cam_;
    Tdepth_rgb = init_Tdepth_rgb;

    calibration::Size2 images_size_depth;
    images_size_depth.x() = depth_cam_->imageWidth();
    images_size_depth.y() = depth_cam_->imageHeight();
    
    std::cout << "depth_cam_->imageWidth()=" << depth_cam_->imageWidth() <<std::endl;
    std::cout << "depth_cam_->imageHeight()=" << depth_cam_->imageHeight() <<std::endl;
    //TODO:: initial (1,0,0) or (0,1,0) -> [0] + [1]*z + [2]*z^2
    depth_error_function_ = calibration::Polynomial<double, 2>(Eigen::Vector3d(0.0,1.0,0.0));
    //TODO:: introduce
    local_model_ = boost::make_shared<calibration::LocalModel>(images_size_depth);
    calibration::LocalModel::Data::Ptr local_matrix = local_model_->createMatrix(calibration::Size2(2, 2), calibration::LocalPolynomial::IdentityCoefficients());
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
//校正板參數在calcCamPose.h修改
//Calibration board parameters are modified in calcCamPose.h.
void RGBD_CALIBRATION::Optimize(std::vector<std::tuple<cv::Mat, std::shared_ptr<PCLCloud3>, double>> &rgb_depth_time)
{
    int obs_count = 0;
    int init_state = true;
    Sophus::SE3d Twc_last;
    calibration::PointPlaneExtraction<calibration::PCLPoint3>::Ptr plane_extractor =
        boost::make_shared<calibration::PointPlaneExtraction<calibration::PCLPoint3>>();

    //step1. convert to calibration board image to pose Tcw / uv / x3Dc
    for (int i = 0; i < rgb_depth_time.size(); i++)
    {
        RGBFramePtr cur_rgb_info = std::make_shared<RGBFrame>();
        Eigen::Matrix4d Twc;
        std::vector<cv::Point3f> x3Dw;
        std::vector<cv::Point2f> uv_2d_distorted;
        std::vector<int> id_landmark;
        bool isCalOk = rgbd_calibration::calcCamPoseRGBD(std::get<2>(rgb_depth_time[i]), std::get<0>(rgb_depth_time[i]), model_rgb_cam, Twc, x3Dw, uv_2d_distorted, id_landmark);

        Eigen::Matrix3d roatation_matrix = Twc.block(0, 0, 3, 3).matrix();
        Eigen::Quaterniond qwc(roatation_matrix);
        Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);

        if (isCalOk == false)
        {
            bad_idx.push_back(i);
            std::cout << "isCalOk false=" <<std::endl;
            continue;
        }
        
        if(init_state){
            Twc_last.translation() = twc;
            Twc_last.setQuaternion(Eigen::Quaterniond(roatation_matrix));
            init_state = false;
        }
        else
        {
            Sophus::SE3d Twc_cur;
            Twc_cur.translation() = twc;
            Twc_cur.setQuaternion(Eigen::Quaterniond(roatation_matrix));
            Sophus::SE3d Tw1_Tw2 = Twc_last.inverse() * Twc_cur;
            Twc_last = Twc_cur;
            if (Tw1_Tw2.translation().norm() < 0.05)
            {
                bad_idx.push_back(i);
                std::cout << "no enough translation=" << Tw1_Tw2.translation().norm() <<std::endl;
                continue;
            }
        }
    

        //check id has
        cur_rgb_info->timestamp = std::get<2>(rgb_depth_time[i]);
        cur_rgb_info->uv_distorted = uv_2d_distorted;
        cur_rgb_info->x3Dw = x3Dw;


        cur_rgb_info->id_landmark = id_landmark; // 1 landmark_id is consists of 4 uv_point; 22 id_landmark : 88 uv_distorted point
        cur_rgb_info->Twc.translation() = twc;
        cur_rgb_info->Twc.setQuaternion(qwc);
        boost::shared_ptr<PCLCloud3> pcl_current = make_shared_ptr(std::get<1>(rgb_depth_time[i]));
        calibration::Pose plane_pose;
        DepthFramePtr cur_depth_info = std::make_shared<DepthFrame>();

        cur_depth_info->cloud_ = pcl_current;
        // cur_depth_info_->pl
        //Transform initial

        rgb_frame_vec[cur_rgb_info->timestamp] = (cur_rgb_info);
        pcl_frame_vec[cur_rgb_info->timestamp] = cur_depth_info;

        close_to_far.push_back(std::pair<double, double>(twc.norm(), cur_rgb_info->timestamp));
        obs_count++;

        if (obs_count > 70)
        {
            rgb_depth_time.erase(rgb_depth_time.begin() + obs_count + 1, rgb_depth_time.end());
            break;
        }
    }

    //Sorting by
    std::cout << "close_to_far size:" << close_to_far.size() << std::endl;
    std::sort(close_to_far.begin(), close_to_far.end(), OrderByDistance());

    //step2.
    // ("Estimating undistortion map...");
    std::cout << "EstimateLocalModel" <<std::endl;
    EstimateLocalModel();

    //step3.
    // ("Recomputing undistortion map..."); according local_fit and inverse_global_fit to undistort center and pcl for new input
    std::cout << "EstimateLocalModelReverse" <<std::endl;
    EstimateLocalModelReverse();

    //step4. ("Estimating global error correction map...");
    std::cout << "EstimateGlobalModel" << std::endl;
    EstimateGlobalModel();

    //step5. estimate inital Tdc
    std::cout << "EstimateTransform Start" << std::endl;
    int obs_num = EstimateTransform(); //estimate pose between planes
    std::cout << "EstimateTransform ENDing" << std::endl;


    //step6. optimize BA, "obs_count" --> need change to 70
    OptimizeAll(obs_num);
    std::cout << "OptimizeAll End" << std::endl;
}

void RGBD_CALIBRATION::OptimizeAll(int obs_num)
{
    std::cout << "OptimizeAll Start" << std::endl;
    ceres::Problem problem;
    Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::DontAlign | Eigen::RowMajor> data(obs_num, 7); //Twc
    Eigen::Matrix<double, 1, 7, Eigen::DontAlign | Eigen::RowMajor> transform;                     //Tdc
    transform[0] = Tdepth_rgb.unit_quaternion().w();
    transform[1] = Tdepth_rgb.unit_quaternion().x();
    transform[2] = Tdepth_rgb.unit_quaternion().y();
    transform[3] = Tdepth_rgb.unit_quaternion().z();
    transform.tail<3>() = Tdepth_rgb.translation();
    double delta[4] = {1.0, 1.0, 0.0, 0.0}; //depth cam instrinsic weight
                                            //fx*delta[0], fy*delta[1], cx+delta[2], cx+delta[3]

    calibration::Size2 images_size_;
    images_size_.x() = model_depth_cam->imageWidth();
    images_size_.y() = model_depth_cam->imageHeight();

    for (int i = 0; i < close_to_far.size(); i++)
    {
        double timestamp = close_to_far[i].second;
        if (!pcl_frame_vec[timestamp]->plane_extracted_ || pcl_frame_vec.find(timestamp) == pcl_frame_vec.end())
            continue;

        rgb_frame_vec[timestamp]->Tcw = rgb_frame_vec[timestamp]->Twc.inverse();
        data.row(i)[0] = rgb_frame_vec[timestamp]->Tcw.unit_quaternion().w();
        data.row(i)[1] = rgb_frame_vec[timestamp]->Tcw.unit_quaternion().x();
        data.row(i)[2] = rgb_frame_vec[timestamp]->Tcw.unit_quaternion().y();
        data.row(i)[3] = rgb_frame_vec[timestamp]->Tcw.unit_quaternion().z();
        data.row(i).tail<3>() = rgb_frame_vec[timestamp]->Tcw.translation();

        TransformDistortionError *error;
        ceres::CostFunction *cost_function;

        error = new TransformDistortionError(model_depth_cam,
                                             rgbd_calibration::col,
                                             rgbd_calibration::row,
                                             rgb_frame_vec[timestamp]->x3Dw,
                                             calibration::PCLConversion<double>::toPointMatrix(*pcl_frame_vec[timestamp]->undistorted_cloud_),
                                             *pcl_frame_vec[timestamp]->estimated_plane_.indices_,
                                             depth_error_function_,
                                             images_size_);

        cost_function = new TransformDistortionCostFunction(error,
                                                            ceres::DO_NOT_TAKE_OWNERSHIP,
                                                            3 * pcl_frame_vec[timestamp]->cloud_->size());

        problem.AddResidualBlock(cost_function,
                                 NULL, //new ceres::CauchyLoss(1.0),
                                 transform.data(),
                                 transform.data() + 4,
                                 global_matrix_->model()->dataPtr(),
                                 data.row(i).data(),
                                 data.row(i).data() + 4,
                                 delta);

        ReprojectionError *repr_error = new ReprojectionError(model_rgb_cam,
                                                              rgb_frame_vec[timestamp]->x3Dw,
                                                              rgbd_calibration::col, rgbd_calibration::row,
                                                              rgb_frame_vec[timestamp]->uv_distorted);


        //number of observation uv factor : rgb_frame_vec[timestamp]->x3Dw.size()
        //number of uv-residual: 2(u and v) * rgb_frame_vec[timestamp]->x3Dw.size()
        ceres::CostFunction *repr_cost_function = new ReprojectionCostFunction(repr_error,
                                                                               ceres::DO_NOT_TAKE_OWNERSHIP,
                                                                               2 * rgb_frame_vec[timestamp]->x3Dw.size());

        problem.AddResidualBlock(repr_cost_function,
                                 NULL, //new ceres::CauchyLoss(1.0),
                                 data.row(i).data(),
                                 data.row(i).data() + 4);

        problem.SetParameterization(data.row(i).data(), new ceres::QuaternionParameterization());
    }
    std::cout << "OptimizeAll error5" << std::endl;
    problem.SetParameterization(transform.data(), new ceres::QuaternionParameterization());
    std::cout << "OptimizeAll error6" << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;

    //  problem.SetParameterBlockConstant(delta);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //output-------------------------
    Eigen::Quaterniond Qdc(transform[0], transform[1], transform[2], transform[3]); //W X Y Z
    // rotation = Quaternion(transform[0], transform[1], transform[2], transform[3]);
    Eigen::Vector3d tdc = transform.tail<3>();
    std::cout << "Qdc xyzw=" << Qdc.coeffs() << std::endl;
    std::cout << "tdc=" << tdc << std::endl;

    std::cout << "delta[0]=" << delta[0] << " delta[1]=" << delta[1]
              << " delta[2]=" << delta[2] << " delta[3]=" << delta[3] << std::endl;

    const int DEGREE = calibration::MathTraits<calibration::GlobalPolynomial>::Degree;
    const int MIN_DEGREE = calibration::MathTraits<calibration::GlobalPolynomial>::MinDegree;
    const int SIZE = DEGREE - MIN_DEGREE + 1;
    typedef calibration::MathTraits<calibration::GlobalPolynomial>::Coefficients Coefficients;

#if 1
    calibration::GlobalPolynomial p1(global_matrix_->model()->polynomial(0, 0));
    calibration::GlobalPolynomial p2(global_matrix_->model()->polynomial(0, 1));
    calibration::GlobalPolynomial p3(global_matrix_->model()->polynomial(1, 0));

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(SIZE, SIZE);
    Eigen::Matrix<double, Eigen::Dynamic, 1> b(SIZE, 1);
    for (int i = 0; i < SIZE; ++i)
    {
        double x(i + 1);
        double y = p2.evaluate(x) + p3.evaluate(x) - p1.evaluate(x);
        double tmp(1.0);
        for (int j = 0; j < MIN_DEGREE; ++j)
            tmp *= x;
        for (int j = 0; j < SIZE; ++j)
        {
            A(i, j) = tmp;
            tmp *= x;
        }
        b[i] = y;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> x = A.colPivHouseholderQr().solve(b);
    global_matrix_->model()->polynomial(1, 1) = x;
    calibration::GlobalPolynomial p4(global_matrix_->model()->polynomial(1, 1));

    std::cout << "p1 poly = " << p1.coefficients() << std::endl;
    std::cout << "p2 poly = " << p2.coefficients() << std::endl;
    std::cout << "p3 poly = " << p3.coefficients() << std::endl;
    std::cout << "p4 poly = " << p4.coefficients() << std::endl;

#endif
}

int RGBD_CALIBRATION::EstimateTransform()
{
    int obs_num = 0;
    //Initial extrinsic parameters are calculated based on the plane of common observation
    calibration::PlaneToPlaneCalibration calib_map;
    for (int i = 0; i < close_to_far.size(); i++)
    {
        /* code */
        double timestamp = close_to_far[i].second;
        if (!pcl_frame_vec[timestamp]->plane_extracted_ || pcl_frame_vec.find(timestamp) == pcl_frame_vec.end())
            continue;

        //Sophus::SE3d cur_world_curdepth = rgb_frame_vec[timestamp]->Twc * Tdepth_rgb.inverse();
        //Sophus::SE3d Tdepthinit_curdepth = world_pose.inverse() * cur_world_curdepth;
        //convert to sophus to transfrom

        //calibration::Pose project_curplane_to_init;
        //project_curplane_to_init.matrix().setIdentity();
        //project_curplane_to_init.matrix().block(0, 0, 3, 3) = (Tdepthinit_curdepth.rotationMatrix());
        //project_curplane_to_init.matrix().col(3).head<3>() = Tdepthinit_curdepth.translation();
        //calibration::Plane transfor_cur_to_world_plane = pcl_frame_vec[timestamp]->estimated_plane_.plane_.transform(project_curplane_to_init);
        //std::cout << "Tdepthinit_curdepth.rotationMatrix()" << Tdepthinit_curdepth.rotationMatrix() << std::endl;
        //std::cout << "Tdepthinit_curdepth.translation()" << Tdepthinit_curdepth.translation() << std::endl;
        //std::cout << "project_curplane_to_init" << project_curplane_to_init.matrix() << std::endl;


        //convert curr plane to inital plane
        calib_map.addPair(pcl_frame_vec[timestamp]->estimated_plane_.plane_,   pcl_frame_vec[timestamp]->checkboard_plane_);
        obs_num++;
        // std::cout << "plane eq:" << pcl_frame_vec[timestamp]->estimated_plane_.plane_.coeffs() << std::endl;
    }

    if (calib_map.getPairNumber() > 5)
    {
        //Change inital extrinsic param
        calibration::Pose Tdc_tmp = calib_map.estimateTransform();
        calibration::AngleAxis rotation(Tdc_tmp.linear());
        calibration::Translation3 translation(Tdc_tmp.translation());
        Eigen::Matrix3d rot_matrix = rotation.toRotationMatrix();
        Eigen::Vector3d translation_vec = translation.vector();
        Tdepth_rgb.setQuaternion(Eigen::Quaterniond(rot_matrix));
        Tdepth_rgb.translation() = translation_vec;
        std::cout << "Tdepth_rgb = " << Tdepth_rgb.matrix3x4() <<std::endl;
    }

    return obs_num;
}

void RGBD_CALIBRATION::EstimateGlobalModel()
{

#if 1
    for (size_t i = 0; i < close_to_far.size(); i += max_threads_)
    {
        int th = 0 ;
        // for (int th = 0; th < max_threads_; ++th)
        {
            if (i + th >= close_to_far.size())
                continue;

            //estimate center x3Dw in depth coordinate.
            double timestamp = close_to_far[i + th].second;

            Eigen::Vector3d chessboard_ori_to_depthcam(rgb_frame_vec[timestamp]->x3Dw[0].x,
                                                       rgb_frame_vec[timestamp]->x3Dw[0].y,
                                                       rgb_frame_vec[timestamp]->x3Dw[0].z);
            Eigen::Vector3d chessboard_ori_to_depthcam1(rgb_frame_vec[timestamp]->x3Dw[1].x,
                                                        rgb_frame_vec[timestamp]->x3Dw[1].y,
                                                        rgb_frame_vec[timestamp]->x3Dw[1].z);
            Eigen::Vector3d chessboard_ori_to_depthcam2(rgb_frame_vec[timestamp]->x3Dw[rgbd_calibration::col].x,
                                                        rgb_frame_vec[timestamp]->x3Dw[rgbd_calibration::col].y,
                                                        rgb_frame_vec[timestamp]->x3Dw[rgbd_calibration::col].z);

            chessboard_ori_to_depthcam = rgb_frame_vec[timestamp]->Twc.inverse() * chessboard_ori_to_depthcam;
            chessboard_ori_to_depthcam1 = rgb_frame_vec[timestamp]->Twc.inverse() * chessboard_ori_to_depthcam1;
            chessboard_ori_to_depthcam2 = rgb_frame_vec[timestamp]->Twc.inverse() * chessboard_ori_to_depthcam2;

            Eigen::Vector3d chessboard_ori_mid_to_depthcam(rgb_frame_vec[timestamp]->x3Dw.back().x,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().y,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().z);
            Eigen::Vector3d proj_center_chess_to_depthCam = (chessboard_ori_mid_to_depthcam / 2);
            proj_center_chess_to_depthCam = Tdepth_rgb * rgb_frame_vec[timestamp]->Twc.inverse() * proj_center_chess_to_depthCam;

            calibration::Point3 und_color_cb_center_tmp;
            und_color_cb_center_tmp.x() = proj_center_chess_to_depthCam.x();
            und_color_cb_center_tmp.y() = proj_center_chess_to_depthCam.y();
            und_color_cb_center_tmp.z() = proj_center_chess_to_depthCam.z();

            calibration::InverseGlobalMatrixEigen inverse_global(inverse_global_fit_->model());
            inverse_global.undistort(0, 0, und_color_cb_center_tmp);
            proj_center_chess_to_depthCam.x() = und_color_cb_center_tmp.x();
            proj_center_chess_to_depthCam.y() = und_color_cb_center_tmp.y();
            proj_center_chess_to_depthCam.z() = und_color_cb_center_tmp.z();

            // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
            boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> und_cloud;
            und_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*pcl_frame_vec[timestamp]->cloud_);
            calibration::LocalMatrixPCL local(local_fit_->model());
            local.undistort(*und_cloud);

            calibration::PlaneInfo plane_info;
            if (ExtractPlane(model_rgb_cam, und_cloud, proj_center_chess_to_depthCam, plane_info))
            {
                std::cout << "EstimateGlobalModel ExtractPlaneInliers=" << plane_info.indices_->size() << std::endl;
                if (plane_info.indices_->size() > 200)
                {
                    pcl_frame_vec[timestamp]->estimated_plane_ = plane_info;
                    pcl_frame_vec[timestamp]->undistorted_cloud_ = und_cloud;
                    pcl_frame_vec[timestamp]->plane_extracted_ = true;
                }

                {
                    calibration::Indices reduced = *plane_info.indices_;
                    std::random_shuffle(reduced.begin(), reduced.end());
                    //reduced.resize(reduced.size() / 5) is sample inliners
                    global_fit_->accumulateCloud(*und_cloud, reduced);
                    //TODO:: target plane : pcl_frame_vec[timestamp]->checkboard_plane_.plane_
                    //input:: *und_cloud, reduced which are changed by local_fit
                    //ans1:
                    // global_fit_->addAccumulatedPoints(pcl_frame_vec[timestamp]->checkboard_plane_); //XY-PLANE (0,0,1) normal, 0 distance

                    //ans2:
                    calibration::Plane fitted_plane =
                    calibration::PlaneFit<double>::fit(calibration::PCLConversion<double>::toPointMatrix(*und_cloud, reduced));
                    global_fit_->addAccumulatedPoints(fitted_plane);

                    //ans3:
                    calibration::Plane plane_from_chessboard = calibration::Plane::Through(chessboard_ori_to_depthcam,
                                                                                         chessboard_ori_to_depthcam1,
                                                                                         chessboard_ori_to_depthcam2);
                    pcl_frame_vec[timestamp]->checkboard_plane_ = plane_from_chessboard;
                    
                    //global_fit_->addAccumulatedPoints(plane_from_chessboard);
                }
            }
        }
    }
    global_fit_->update();
#endif
}

void RGBD_CALIBRATION::EstimateLocalModelReverse()
{

    //only reset obs, retain calibration matrix
    local_fit_->reset();
    bool global_update = true;
    bool local_update = true;
    int w = model_depth_cam->imageWidth();
    int h = model_depth_cam->imageHeight();
    for (size_t i = 0; i < close_to_far.size(); i += max_threads_)
    {
        // if (i > 30)
        //     break;
        // for (int th = 0; th < max_threads_; ++th)
        int th = 0;
        {
            if (i + th >= close_to_far.size())
                continue;

            //close to far
            double timestamp = close_to_far[i + th].second;

            Eigen::Vector3d chessboard_ori_mid_to_depthcam(rgb_frame_vec[timestamp]->x3Dw.back().x,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().y,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().z);
            Eigen::Vector3d proj_center_chess_to_depthCam = (chessboard_ori_mid_to_depthcam / 2);
            proj_center_chess_to_depthCam = Tdepth_rgb * rgb_frame_vec[timestamp]->Twc.inverse() * proj_center_chess_to_depthCam;
            
            // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
            boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> und_cloud;
            und_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*pcl_frame_vec[timestamp]->cloud_);

//correct center x3Ddepth from checkboard project to depth cam
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

//correct pcl undistoration
            if (local_update)
            {
                calibration::LocalMatrixPCL local(local_fit_->model());
                local.undistort(*und_cloud);
            }

            calibration::PlaneInfo plane_info;
            if (ExtractPlane(model_rgb_cam, und_cloud, proj_center_chess_to_depthCam, plane_info))
            {
                std::cout << "EstimateLocalModelReverse ExtractPlaneInliers=" << plane_info.indices_->size() << std::endl;
                //indices info from current extraction plane of inliers --> plane_info
                //indices only for current fitted_plane
                boost::shared_ptr<std::vector<int>> indices = boost::make_shared<std::vector<int>>(); // = *plane_info.indices_;
                indices->reserve(plane_info.indices_->size());
                for (size_t j = 0; j < plane_info.indices_->size(); ++j)
                {
                    int r = (*plane_info.indices_)[j] / w;
                    int c = (*plane_info.indices_)[j] % w;
                    if ((r - h / 2) * (r - h / 2) + (c - w / 2) * (c - w / 2) < (h / 2) * (h / 2))
                        indices->push_back((*plane_info.indices_)[j]);
                }

                //fitted_plane and plane_info are different
                calibration::Plane fitted_plane = 
                calibration::PlaneFit<double>::fit(calibration::PCLConversion<double>::toPointMatrix(*pcl_frame_vec[timestamp]->cloud_, *indices));

                //old_indices infor from idx of inliner for EstimateLocalModelPlane result(old)
                boost::shared_ptr<std::vector<int>> old_indices;
                {
                    old_indices = plane_info_map_[timestamp].indices_;
                }
                indices->clear();
                //std::set_union, input [old_indices, plane_info.indices_]  output *indices
                std::set_union(old_indices->begin(), old_indices->end(), plane_info.indices_->begin(), plane_info.indices_->end(), 
                std::back_inserter(*indices));

                {
                    local_fit_->accumulateCloud(*pcl_frame_vec[timestamp]->cloud_, *indices);
                    local_fit_->addAccumulatedPoints(fitted_plane);
                    plane_info_map_[timestamp].indices_ = indices;
                }
            }
        }
        //There is a table of the same size as the image in local_fit_
        //and each coordinate in the table corresponds to a polynomial(second order) for correction function
        std::cout << "===EstimateLocalModelReverse local_fit_===" <<std::endl;
        local_fit_->update();
    }
}

//("Estimating undistortion map...");
void RGBD_CALIBRATION::EstimateLocalModel()
{
    int w = model_depth_cam->imageWidth();
    int h = model_depth_cam->imageHeight();
    bool global_update = false;
    bool local_update = false;
    std::cout << "EstimateLocalModel" << std::endl;
    for (size_t i = 0; i < close_to_far.size(); i += max_threads_)
    {
        bool extract_plane_sucess = false;
        int th = 0;
        {
            if (i + th >= close_to_far.size())
                continue;

            //close to far
            double timestamp = close_to_far[i + th].second;

            Eigen::Vector3d chessboard_ori_mid_to_depthcam(rgb_frame_vec[timestamp]->x3Dw.back().x,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().y,
                                                           rgb_frame_vec[timestamp]->x3Dw.back().z);
            Eigen::Vector3d proj_center_chess_to_depthCam = (chessboard_ori_mid_to_depthcam / 2);

            proj_center_chess_to_depthCam = Tdepth_rgb * rgb_frame_vec[timestamp]->Twc.inverse() * proj_center_chess_to_depthCam;
            // Extract plane from undistorted cloud(pcl_frame_vec[timestamp])
            boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> und_cloud;
            und_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*pcl_frame_vec[timestamp]->cloud_);
            //correct center x3Ddepth from checkboard project to depth cam
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

            //correct pcl undistoration
            if (local_update)
            {
                calibration::LocalMatrixPCL local(local_fit_->model());
                local.undistort(*und_cloud);
            }
            calibration::PlaneInfo plane_info;
            if (ExtractPlane(model_rgb_cam, und_cloud, proj_center_chess_to_depthCam, plane_info))
            // if (ExtractPlane(model_rgb_cam, und_cloud, proj_center_chess_to_depthCam, plane_info))
            {
                std::cout << "EstimateLocalModel ExtractPlaneInliers=" << plane_info.indices_->size() << std::endl;
                extract_plane_sucess = true;
                plane_info_map_[timestamp] = plane_info;
                std::vector<int> indices; // = *plane_info.indices_;
                indices.reserve(plane_info.indices_->size());
                for (size_t j = 0; j < plane_info.indices_->size(); ++j)
                {
                    int r = (*plane_info.indices_)[j] / w;
                    int c = (*plane_info.indices_)[j] % w;
                    if ((r - h / 2) * (r - h / 2) + (c - w / 2) * (c - w / 2) < (h / 2) * (h / 2))
                        indices.push_back((*plane_info.indices_)[j]);
                }
                //fitted_plane and plane_info are different

                calibration::Plane fitted_plane =
                    calibration::PlaneFit<double>::fit(calibration::PCLConversion<double>::toPointMatrix(*pcl_frame_vec[timestamp]->cloud_, indices));
                plane_info_map_[timestamp].plane_ = fitted_plane;


                // std:;cout << "params fitted_plane.offset=" << fitted_plane.offset <<std::endl;

                {
                    local_fit_->accumulateCloud(*pcl_frame_vec[timestamp]->cloud_, *plane_info.indices_);
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
                        const calibration::Point3 corner = calibration::Point3(obs_proj_check_to_depth.x(),
                                                                               obs_proj_check_to_depth.y(),
                                                                               obs_proj_check_to_depth.z());
                        inverse_global_fit_->addPoint(0, 0, corner, fitted_plane);
                    }
                    if (i + th > 20)
                    {
                        inverse_global_fit_->update();
                        global_update = true;
                        std::cout << "global correcting0 = " << inverse_global_fit_->model()->polynomial(0, 0).data()[0] << std::endl;
                        std::cout << "global correcting1 = " << inverse_global_fit_->model()->polynomial(0, 0).data()[1] << std::endl;
                        std::cout << "global correcting2 = " << inverse_global_fit_->model()->polynomial(0, 0).data()[2] << std::endl;
                    }
                }
            }
        }
        //There is a table of the same size as the image in local_fit_
        //and each coordinate in the table corresponds to a polynomial(second order) for correction function

        if (extract_plane_sucess)
        {
            local_fit_->update();
            local_update = true;
        }
    }
}

bool RGBD_CALIBRATION::ExtractPlane(CameraPtr color_cam_model,
                                    const PCLCloud3::ConstPtr &cloud,
                                    const Eigen::Vector3d &center,
                                    calibration::PlaneInfo &plane_info)
{
    slam_ros::ScopedTrace tracer("ExtractPlane");
    //color_cb.width() = rgb_image num of rows / cols
    double radius =  2 * (std::min(color_cam_model->imageWidth(), color_cam_model->imageHeight()) / 1.5); // TODO Add parameter
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
    plane_extractor.setRadius(2 * radius);
    plane_extractor.setPoint(calibration::PCLPoint3(center.x(), center.y(), center.z()));
    plane_extracted = plane_extractor.extract(plane_info);
#endif
    return plane_extracted;
}
