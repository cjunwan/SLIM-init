

#ifndef IMUPREINTEGRATED_HPP
#define IMUPREINTEGRATED_HPP
#include "basicTypes.hpp"

namespace vio {

class IMUPreintegrated
{
public:

    struct integrable
    {

        integrable(const Eigen::Vector3d &w, const Eigen::Vector3d &a , const double &t):w_(w),a_(a),t_(t){}
        Eigen::Vector3d w_;
        Eigen::Vector3d a_;
        double t_;
    };

    IMUPreintegrated(){}
    IMUPreintegrated(const IMUBias &bias, const IMUCalibParam *calib, double start_time, double end_time);

    void initialize(const IMUBias &bias);
    void reinitialize(const IMUBias &bias);

    void integrate_new_measurement(const Eigen::Vector3d &gyro_mea, const Eigen::Vector3d &acc_mea, const double &dt);
    Sophus::SO3d reintegrate_gyro_measurement(const Eigen::Vector3d& biasg);
    void reintegrate(const IMUBias &bias);
    void reintegrate(const IMUBias &bias, const std::vector<integrable>& new_imu_meas);

    Matrix15d      get_information();
    Eigen::Vector3d get_delta_velocity(const IMUBias &new_bias) const;
    Eigen::Vector3d get_delta_position(const IMUBias &new_bias) const;
    Sophus::SO3d    get_delta_rotation(const IMUBias &new_bias) const;
    template<typename T>
    Sophus::SO3<T> get_delta_rotation(const Eigen::Matrix<T, 3, 1>& bg) const
    {
        return dR_.template cast<T>() * Sophus::SO3<T>::exp(JRg_.template cast<T>() * (bias_.bg_.template cast<T>() - bg));
    }

    IMUBias bias_;

    double sum_dt_;        
    Matrix6d Nga_;        
    Matrix6d NgaWalk_;    
    Matrix15d Cov_;       
    Matrix15d Info_;      
    Sophus::SO3d dR_;
    Eigen::Vector3d dV_;
    Eigen::Vector3d dP_;
    Eigen::Matrix3d JRg_;
    Eigen::Matrix3d JVg_;
    Eigen::Matrix3d JVa_;
    Eigen::Matrix3d JPg_;
    Eigen::Matrix3d JPa_;


    std::vector<integrable> imu_measurements_;
    double start_t_ns;
    double end_t_ns;


};

} 

#endif