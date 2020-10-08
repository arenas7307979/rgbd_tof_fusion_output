#include "vector"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

//camera model
#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"


//calibration_common
#include "globals.h"
#include "calibration_common/algorithms/plane_extraction.h"
#include "calibration_common/ceres/math.h"
#include "calibration_common/base/math.h"
#include "calibration_common/algorithms/plane_to_plane_calibration.h"
#include "calibration_common/algorithms/point_on_plane_calibration.h"
#include "calibration_common/base/pcl_conversion.h"
#include "calibration_common/objects/planar_object.h"

class ReprojectionError
{
public:
// TODO::add chessboard row_corner * col_corner

    ReprojectionError(CameraPtr camera_model,
                      std::vector<cv::Point3f> &Xw_vec,
                      int corner_cols_,
                      int corner_rows_,
                      std::vector<cv::Point2f> &uv_distorted) : camera_model_(camera_model), checkerboard_(Xw_vec)
    {
        chessboard_corner_colrow = calibration::Size2(corner_cols_, corner_rows_);
        image_corners_ = calibration::Cloud2(chessboard_corner_colrow);
        for (int i = 0; i < image_corners_.elements(); i++)
        {
            image_corners_[i][0] = uv_distorted[i].x;
            image_corners_[i][1] = uv_distorted[i].y;
        }
    }

    template <typename T>
    typename calibration::Types<T>::Pose toEigen(const T *const q, const T *const t) const
    {
        typename calibration::Types<T>::Quaternion rotation(q[0], q[1], q[2], q[3]);
        typename calibration::Types<T>::Translation3 translation(t[0], t[1], t[2]);
        return translation * rotation;
    }

    template <typename T>
    bool
    operator()(const T *const checkerboard_pose_q,
               const T *const checkerboard_pose_t,
               T *residuals) const
    {
        typename calibration::Types<T>::Pose checkerboard_pose_eigen = toEigen<T>(checkerboard_pose_q, checkerboard_pose_t);

        typename calibration::Types<T>::Cloud3 cb_corners(chessboard_corner_colrow);
        for (calibration::Size1 i = 0; i < cb_corners.elements(); ++i)
        {
            cb_corners[i][0] = checkerboard_[i].x;
            cb_corners[i][1] = checkerboard_[i].y;
            cb_corners[i][2] = checkerboard_[i].z;
        }
        cb_corners.container() = checkerboard_pose_eigen * cb_corners.container(); //Tcw * Xw

        typename calibration::Types<T>::Cloud2 reprojected_corners(chessboard_corner_colrow);
        for (calibration::Size1 i = 0; i < cb_corners.elements(); ++i)
        {
            Eigen::Vector3d Xci(cb_corners[i][0], cb_corners[i][1], cb_corners[i][2]);
            Eigen::Vector2d uv_disorted; //Xc to uv_distorted
            camera_model_->spaceToPlane(Xci, uv_disorted);
            reprojected_corners[i][0] = uv_disorted.x();
            reprojected_corners[i][1] = uv_disorted.y();
        }

        Eigen::Map<Eigen::Matrix<T, 2, Eigen::Dynamic>> residual_map(residuals, 2, cb_corners.elements());
        residual_map = (reprojected_corners.container() - image_corners_.container().cast<T>()) / (0.5 * std::sqrt(T(cb_corners.elements())));
        return true;
  }

private:
  calibration::Size2 chessboard_corner_colrow;
  CameraPtr camera_model_;
  std::vector<cv::Point3f> checkerboard_;
  calibration::Cloud2 image_corners_; //distorted points
};


typedef ceres::NumericDiffCostFunction<ReprojectionError, ceres::CENTRAL, ceres::DYNAMIC, 4, 3> ReprojectionCostFunction;



  // const Checkerboard::ConstPtr &checkerboard_;

  // const Cloud3 depth_points_;
  // const Indices plane_indices_;

  // const Polynomial<Scalar, 2> depth_error_function_;
  // const Size2 images_size_;

class TransformDistortionError
{
public:
  TransformDistortionError(const CameraPtr depth_model,
                           int corner_cols_,
                           int corner_rows_,
                           std::vector<cv::Point3f> Xw_vec_,
                           const calibration::Cloud3 &depth_points,
                           const calibration::Indices &plane_indices,
                           const calibration::Polynomial<double, 2> &depth_error_function,
                           const calibration::Size2 &images_size)
      : depth_cam_model(depth_model),
        Xw_vec(Xw_vec_),
        depth_points_(depth_points),
        plane_indices_(plane_indices),
        depth_error_function_(depth_error_function),
        images_size_(images_size)
  {
    chessboard_corner_colrow = calibration::Size2(corner_cols_, corner_rows_);
    chessboard_Xw = calibration::Cloud3(chessboard_corner_colrow);

    for (int i = 0; i < chessboard_Xw.elements(); i++)
    {
      chessboard_Xw[i][0] = Xw_vec_[i].x;
      chessboard_Xw[i][1] = Xw_vec_[i].y;
      chessboard_Xw[i][2] = Xw_vec_[i].z;
    }
  }

  template <typename T>
  typename calibration::Types<T>::Pose toEigen(const T *const q, const T *const t) const
  {
    typename calibration::Types<T>::Quaternion rotation(q[0], q[1], q[2], q[3]);
    typename calibration::Types<T>::Translation3 translation(t[0], t[1], t[2]);
    return translation * rotation;
  }
  
  template <typename T>
  bool operator()(const T *const color_sensor_pose_q,
                  const T *const color_sensor_pose_t,
                  const T *const global_undistortion,
                  const T *const checkerboard_pose_q,
                  const T *const checkerboard_pose_t,
                  const T *const delta,
                  T *residuals) const
  {
    typename calibration::Types<T>::Pose color_sensor_pose_eigen = toEigen<T>(color_sensor_pose_q, color_sensor_pose_t);
    typename calibration::Types<T>::Pose checkerboard_pose_eigen = toEigen<T>(checkerboard_pose_q, checkerboard_pose_t);
    const int DEGREE = calibration::MathTraits<calibration::GlobalPolynomial>::Degree;
    const int MIN_DEGREE = calibration::MathTraits<calibration::GlobalPolynomial>::MinDegree;
    const int SIZE = calibration::MathTraits<calibration::GlobalPolynomial>::Size; //DEGREE - MIN_DEGREE + 1;
    typedef calibration::MathTraits<calibration::GlobalPolynomial>::Coefficients Coefficients;

    calibration::Size1 index = 0;

    Coefficients c1, c2, c3;
    for (calibration::Size1 i = 0; i < DEGREE - MIN_DEGREE + 1; ++i)
      c1[i] = global_undistortion[index++];
    for (calibration::Size1 i = 0; i < DEGREE - MIN_DEGREE + 1; ++i)
      c2[i] = global_undistortion[index++];
    for (calibration::Size1 i = 0; i < DEGREE - MIN_DEGREE + 1; ++i)
      c3[i] = global_undistortion[index++];

    calibration::Polynomial<T, DEGREE, MIN_DEGREE> p1(c1);
    calibration::Polynomial<T, DEGREE, MIN_DEGREE> p2(c2);
    calibration::Polynomial<T, DEGREE, MIN_DEGREE> p3(c3);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A(SIZE, SIZE);
    Eigen::Matrix<T, Eigen::Dynamic, 1> b(SIZE, 1);
    for (int i = 0; i < SIZE; ++i)
    {
      T x(i + 1);
      T y = p2.evaluate(x) + p3.evaluate(x) - p1.evaluate(x);
      T tmp(1.0);
      for (int j = 0; j < MIN_DEGREE; ++j)
        tmp *= x;
      for (int j = 0; j < SIZE; ++j)
      {
        A(i, j) = tmp;
        tmp *= x;
      }
      b[i] = y;
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> x = A.colPivHouseholderQr().solve(b);

    calibration::GlobalModel::Data::Ptr global_data = boost::make_shared<calibration::GlobalModel::Data>(calibration::Size2(2, 2));
    for (int i = 0; i < 3 * calibration::MathTraits<calibration::GlobalPolynomial>::Size; ++i)
      global_data->container().data()[0 * calibration::MathTraits<calibration::GlobalPolynomial>::Size + i] = global_undistortion[i];
    for (int i = 0; i < SIZE; ++i)
      global_data->container().data()[3 * calibration::MathTraits<calibration::GlobalPolynomial>::Size + i] = x[i];

    std::cout << "OPT 444" << std::endl;

    calibration::GlobalModel::Ptr global_model = boost::make_shared<calibration::GlobalModel>(images_size_);
    global_model->setMatrix(global_data);
    std::cout << "OPT 555" <<std::endl;

    calibration::GlobalMatrixEigen global(global_model);
    std::cout << "OPT 666" <<std::endl;

    typename calibration::Types<T>::Cloud3 depth_points(depth_points_.size());
    depth_points.container() = depth_points_.container().cast<T>();
    std::vector<double> depth_cam_K_para =  depth_cam_model->getK();
    cv::Matx33d K;
    K << depth_cam_K_para[0], 0, depth_cam_K_para[2],
        0, depth_cam_K_para[1], depth_cam_K_para[3],
        0, 0, 1;

    for (int j = 0; j < depth_points.size().y(); ++j)
    {
      for (int i = 0; i < depth_points.size().x(); ++i)
      {
        T z = depth_points(i, j).z();
        typename calibration::Types<T>::Point2 normalized_pixel((i - (K(0, 2) + delta[2])) / (K(0, 0) * delta[0]),
                                                                (j - (K(1, 2) + delta[3])) / (K(1, 1) * delta[1]));
        Eigen::Vector2d tmp_normalized_pixel(normalized_pixel[0], normalized_pixel[1]);
        Eigen::Vector3d Xc;
        depth_cam_model->normalliftProjective(tmp_normalized_pixel, Xc);
        Xc = Xc * z;
        depth_points(i, j).x() = Xc.x();
        depth_points(i, j).y() = Xc.y();
      }
    }
    std::cout << "OPT 888" <<std::endl;
    
    global.undistort(depth_points);

    std::cout << "OPT 999" <<std::endl;
    
    typename calibration::Types<T>::Cloud3 cb_corners(chessboard_corner_colrow);

    cb_corners.container() = color_sensor_pose_eigen * checkerboard_pose_eigen * chessboard_Xw.container().cast<T>();
    
    std::cout << "OPT 1010" <<std::endl;
    
    calibration::Polynomial<T, 2> depth_error_function(depth_error_function_.coefficients().cast<T>());
    typename calibration::Types<T>::Plane cb_plane = calibration::Types<T>::Plane::Through(cb_corners(0, 0), cb_corners(0, 1), cb_corners(1, 0));

    std::cout << "plane_indices_.size()=" << plane_indices_.size() << std::endl;

    Eigen::Map<Eigen::Matrix<T, 3, Eigen::Dynamic>> residual_map_dist(residuals, 3, plane_indices_.size()); //3 row n col

    for (calibration::Size1 i = 0; i < plane_indices_.size(); ++i)
    {
      if (depth_points[plane_indices_[i]].z() > 0)
      {
        calibration::Line line(calibration::Point3::Zero(), depth_points[plane_indices_[i]].normalized());
        // std::cout << " depth_points[plane_indices_[i]].normalized() =" << depth_points[plane_indices_[i]].normalized() << std::endl;
        // std::cout << "line.intersectionPoint(cb_plane)=" << line.intersectionPoint(cb_plane) << std::endl;
        // std::cout << "epth_points[plane_indices_[i]])=" << depth_points[plane_indices_[i]] << std::endl;
        // std::cout << "depth input =" << depth_points[plane_indices_[i]].z() << std::endl;
        // std::cout << "depth ouput=" << ceres::poly_eval(depth_error_function.coefficients(), depth_points[plane_indices_[i]].z()) << std::endl;
#if 1
        residual_map_dist.col(i) = (line.intersectionPoint(cb_plane) - depth_points[plane_indices_[i]]) /
                                   (std::sqrt(T(plane_indices_.size())) *
                                    ceres::poly_eval(depth_error_function.coefficients(),
                                                     depth_points[plane_indices_[i]].z()));
#else
        double poly_depth = ceres::poly_eval(depth_error_function.coefficients(), depth_points[plane_indices_[i]].z());
        residual_map_dist.col(i) = (line.intersectionPoint(cb_plane) - depth_points[plane_indices_[i]]) / (10000 * poly_depth);
#endif
      }
      else
      {
        residual_map_dist.col(i)[0] = 0;
        residual_map_dist.col(i)[1] = 0;
        residual_map_dist.col(i)[2] = 0;
      }
    }
    return true;
  }


private:
  CameraPtr depth_cam_model;
  std::vector<cv::Point3f> Xw_vec;
  calibration::Cloud3 chessboard_Xw;
  const calibration::Cloud3 depth_points_;
  const calibration::Indices plane_indices_;
  calibration::Size2 chessboard_corner_colrow;
  const calibration::Polynomial<double, 2> depth_error_function_;
  const calibration::Size2 images_size_;
};

typedef ceres::NumericDiffCostFunction<TransformDistortionError, ceres::CENTRAL, ceres::DYNAMIC, 4, 3,
                                       3 * calibration::MathTraits<calibration::GlobalPolynomial>::Size, 4, 3, 4>
    TransformDistortionCostFunction;
