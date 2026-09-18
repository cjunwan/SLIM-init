
#ifndef DERVIO_FEATUREMANAGER_H
#define DERVIO_FEATUREMANAGER_H

#include "utils/eigenTypes.h"
#include <map>
#include <vector>

using namespace Eigen;
using FeatureID = int;
using TimeFrameId = double;
using FeatureTrackerResulst = Eigen::aligned_map<
        int, Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

struct MotionData {
    double timestamp;
    Eigen::Vector3d imu_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gyro = Eigen::Vector3d::Zero();
};

class FeaturePerFrame {
public:
    FeaturePerFrame() = default;

    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
        normalpoint.x() = _point(0);
        normalpoint.y() = _point(1);
        normalpoint.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }

    double cur_td;
    Vector3d normalpoint;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class SFMFeature {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SFMFeature() = default;

    SFMFeature(FeatureID _feature_id, TimeFrameId _start_frame)
            : feature_id(_feature_id), kf_id(_start_frame) {}

    FeatureID feature_id{};
    TimeFrameId kf_id{};

    bool state = false; 
    Vector3d p3d; 
    Eigen::aligned_map<TimeFrameId, FeaturePerFrame> obs;
};
class LinePerFrame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LinePerFrame() = default;

    LinePerFrame(const Eigen::Vector3d& sp, const Eigen::Vector3d& ep,
                 double len_px = 0.0, double det_score = 0.0,
                 double match_sc = 0.0, double obs_q = 0.0)
        : start_point(sp), end_point(ep),
          length_px(len_px), detector_score(det_score),
          match_score(match_sc), obs_quality(obs_q) {}

    Eigen::Vector3d start_point;
    Eigen::Vector3d end_point;
    double length_px = 0.0;
    double detector_score = 0.0;
    double match_score = 0.0;
    double obs_quality = 0.0;
    int vp_id = -1;
};

class SFMLine {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SFMLine() = default;
    SFMLine(FeatureID _line_id) : line_id(_line_id) {}

    FeatureID line_id{};
    int track_count = 0;
    
    Eigen::aligned_map<TimeFrameId, LinePerFrame> obs;
};

#endif 
