#pragma once
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace Sophus {

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Sophus::SO3Base<Derived> &q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = q.unit_quaternion();
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w();
    ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec();
    ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity()
                                     + Sophus::SO3<typename Derived::Scalar>::hat(qq.vec());
    return ans;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Sophus::SO3Base<Derived> &q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = q.unit_quaternion();
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w();
    ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec();
    ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity()
                                     - Sophus::SO3<typename Derived::Scalar>::hat(qq.vec());
    return ans;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> R2ypr(const Sophus::SO3Base<Derived>& q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R = q.matrix();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> ypr;
    ypr(0) = std::atan2(R(1, 0), R(0, 0));
    ypr(1) = std::atan2(-R(2, 0), R(0, 0) * std::cos(ypr(0)) + R(1, 0) * std::sin(ypr(0)));
    ypr(2) = std::atan2(R(2, 1), R(2, 2));
    return ypr;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> R2ypr(const Eigen::QuaternionBase<Derived>& q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R = q.toRotationMatrix();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> ypr;
    ypr(0) = std::atan2(R(1, 0), R(0, 0));
    ypr(1) = std::atan2(-R(2, 0), R(0, 0) * std::cos(ypr(0)) + R(1, 0) * std::sin(ypr(0)));
    ypr(2) = std::atan2(R(2, 1), R(2, 2));
    return ypr;
}

template<class Scalar>
Sophus::SO3<Scalar> ypr2R(Scalar yaw, Scalar pitch, Scalar roll) {
    Sophus::SO3<Scalar> q(Eigen::AngleAxisd(yaw  , Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(roll , Eigen::Vector3d::UnitX()));
    return q;
}

// Returns the 3D cross product skew symmetric matrix of a given 3D vector
template<typename T>
Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& vec)
{
    return (Eigen::Matrix<T, 3, 3>() << T(0), -vec(2), vec(1),
                                        vec(2), T(0), -vec(0),
                                        -vec(1), vec(0), T(0)).finished();
}

//-------------------------------jacobian function.

// right jacobian of SO(3)
static Eigen::Matrix3d JacobianR(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d Jr = Eigen::Matrix3d::Identity();
    double theta = w.norm();
    if (theta < 0.00001)
    {
        return Jr; // = Matrix3d::Identity();
    }
    else
    {
        Eigen::Vector3d k = w.normalized(); // k - unit direction vector of w
        Eigen::Matrix3d K = skew(k);
        Jr = Matrix3d::Identity() - (1 - cos(theta)) / theta * K + (1 - sin(theta) / theta) * K * K;
    }
    return Jr;
}


//using example
static Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d Jrinv = Eigen::Matrix3d::Identity();
    double theta = w.norm();

    // very small angle
    if (theta < 0.00001)
    {
        return Jrinv;
    }
    else
    {
        Eigen::Vector3d k = w.normalized(); // k - unit direction vector of w
        Eigen::Matrix3d K = Sophus::SO3d::hat(k);
        Jrinv = Eigen::Matrix3d::Identity() + 0.5 * Sophus::SO3d::hat(w) + (1.0 - (1.0 + cos(theta)) * theta / (2.0 * sin(theta))) * K * K;
    }

    return Jrinv;
    /*
         * in gtsam:
         *
         *   double theta2 = omega.dot(omega);
         *  if (theta2 <= std::numeric_limits<double>::epsilon()) return I_3x3;
         *  double theta = std::sqrt(theta2);  // rotation angle
         *  * Right Jacobian for Log map in SO(3) - equation (10.86) and following equations in
         *   * G.S. Chirikjian, "Stochastic Models, Information Theory, and Lie Groups", Volume 2, 2008.
         *   * logmap( Rhat * expmap(omega) ) \approx logmap( Rhat ) + Jrinv * omega
         *   * where Jrinv = LogmapDerivative(omega);
         *   * This maps a perturbation on the manifold (expmap(omega))
         *   * to a perturbation in the tangent space (Jrinv * omega)
         *
         *  const Matrix3 W = skewSymmetric(omega); // element of Lie algebra so(3): W = omega^
         *  return I_3x3 + 0.5 * W +
         *         (1 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) *
         *             W * W;
         *
         * */
}

// left jacobian of SO(3), Jl(x) = Jr(-x)
static Eigen::Matrix3d JacobianL(const Eigen::Vector3d &w)
{
    return JacobianR(-w);
}

// left jacobian inverse
static Eigen::Matrix3d JacobianLInv(const Eigen::Vector3d &w)
{
    return JacobianRInv(-w);
}

inline Eigen::Quaterniond normalizeRotationQ(const Eigen::Quaterniond &r)
{
    Eigen::Quaterniond _r(r);
    if (_r.w() < 0)
    {
        _r.coeffs() *= -1;
    }
    return _r.normalized();
}

inline Eigen::Matrix3d normalizeRotationM(const Eigen::Matrix3d &R)
{
    Eigen::Quaterniond qr(R);
    return normalizeRotationQ(qr).toRotationMatrix();
}


}
