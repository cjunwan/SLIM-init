#include "featureTracker/parameters.h"
#include <iostream>
#include <experimental/filesystem>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

vector<Eigen::Matrix3d> RIC;
vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
string EX_CALIB_RESULT_PATH;
string VINS_RESULT_PATH;
string OUTPUT_PATH;
double ROW, COL;
double TD, TR;



int FOCAL_LENGTH;
string IMAGE_TOPIC;
string IMU_TOPIC;
string FISHEYE_MASK;
vector<string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int FREQ;
int CAMERA_FREQ;
int IMU_FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
bool STEREO_TRACK;
int EQUALIZE;
int FISHEYE;
bool PUB_THIS_FRAME;

int LINE_MAX_CNT;
double LINE_MIN_LENGTH;
double LINE_NMS_DIST;
double LINE_NMS_ANGLE_DEG;
int LINE_GRADIENT_THRESHOLD;
double LINE_MATCH_MAX_DIST;
double LINE_MATCH_MAX_ANGLE;
double LINE_MATCH_MIN_OVERLAP;

double LINE_RESIDUAL_WEIGHT;
double LINE_NORMAL_WEIGHT;
double LINE_EPIPOLAR_WEIGHT;
int LINE_USE_NORMAL_RESIDUAL;
int LINE_USE_EPIPOLAR_RESIDUAL;
int LINE_DISABLE_FILTERING;
int LINE_MIN_USED_FOR_ALIGNMENT;

int EPIPOLAR_MIN_TRACK_COUNT = 4;
double EPIPOLAR_MIN_PAIR_COVERAGE = 0.50;
double EPIPOLAR_MIN_MATCH_QUALITY = 0.35;
double EPIPOLAR_MIN_OBS_QUALITY = 0.40;
double EPIPOLAR_MIN_SCORE = 0.60;
int EPIPOLAR_MAX_LINES_GLOBAL = 4;

double VP_USE_RESIDUAL = 0.0;
int ACC_CHECK_ENABLE = 1;
int ACC_CHECK_MAX_NEAR_G_SEGMENTS = 1;



void readParameters(string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "1 readParameters ERROR: Wrong path to settings!" << endl;
        return;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;

    FOCAL_LENGTH = 460;
    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_PATH;
    fs::path config_path;
    try {
        config_path = fs::absolute(fs::path(config_file));
    } catch (...) {
        config_path = fs::path(config_file);
    }
    fs::path config_dir = config_path.empty() ? fs::path(".") : config_path.parent_path();
    fs::path project_root = config_dir.parent_path();
    if (config_dir.empty()) config_dir = fs::path(".");
    if (project_root.empty()) project_root = config_dir;
    if (OUTPUT_PATH.empty()) {
        OUTPUT_PATH = (project_root / "result").string();
        std::cout << "[WARN] output_path is not set. Falling back to " << OUTPUT_PATH << std::endl;
    } else {
        fs::path out_path(OUTPUT_PATH);
        if (out_path.is_relative()) {
            OUTPUT_PATH = (config_dir / out_path).string();
        }
    }
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.txt";

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0){
            cout << " fix extrinsic param " << endl;
        }
        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
    }
    else
    {
        TR = 0;
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    CAMERA_FREQ = fsSettings["camera_freq"];
    IMU_FREQ = fsSettings["imu_freq"];

    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    LINE_MAX_CNT = 200;
    LINE_MIN_LENGTH = 30.0;
    LINE_NMS_DIST = 8.0;
    LINE_NMS_ANGLE_DEG = 10.0;
    LINE_GRADIENT_THRESHOLD = 36;
    if (!fsSettings["line_max_cnt"].empty()) fsSettings["line_max_cnt"] >> LINE_MAX_CNT;
    if (!fsSettings["line_min_length"].empty()) fsSettings["line_min_length"] >> LINE_MIN_LENGTH;
    if (!fsSettings["line_gradient_threshold"].empty()) fsSettings["line_gradient_threshold"] >> LINE_GRADIENT_THRESHOLD;
    if (!fsSettings["line_nms_dist"].empty()) fsSettings["line_nms_dist"] >> LINE_NMS_DIST;
    if (!fsSettings["line_nms_angle_deg"].empty()) fsSettings["line_nms_angle_deg"] >> LINE_NMS_ANGLE_DEG;
    LINE_MATCH_MAX_DIST = 10.0;
    LINE_MATCH_MAX_ANGLE = 5.0;
    LINE_MATCH_MIN_OVERLAP = 0.7;
    if (!fsSettings["line_match_max_dist"].empty()) fsSettings["line_match_max_dist"] >> LINE_MATCH_MAX_DIST;
    if (!fsSettings["line_match_max_angle"].empty()) fsSettings["line_match_max_angle"] >> LINE_MATCH_MAX_ANGLE;
    if (!fsSettings["line_match_min_overlap"].empty()) fsSettings["line_match_min_overlap"] >> LINE_MATCH_MIN_OVERLAP;
    LINE_RESIDUAL_WEIGHT = 0.02; 
    LINE_NORMAL_WEIGHT = 1.0;    
    LINE_EPIPOLAR_WEIGHT = 1.0;  
    LINE_USE_NORMAL_RESIDUAL = 1; 
    LINE_USE_EPIPOLAR_RESIDUAL = 1; 
    LINE_DISABLE_FILTERING = 0;
    LINE_MIN_USED_FOR_ALIGNMENT = 20; 
    if (!fsSettings["line_residual_weight"].empty()) fsSettings["line_residual_weight"] >> LINE_RESIDUAL_WEIGHT;
    if (!fsSettings["line_normal_weight"].empty()) fsSettings["line_normal_weight"] >> LINE_NORMAL_WEIGHT;
    if (!fsSettings["line_epipolar_weight"].empty()) fsSettings["line_epipolar_weight"] >> LINE_EPIPOLAR_WEIGHT;
    if (!fsSettings["line_use_normal_residual"].empty()) fsSettings["line_use_normal_residual"] >> LINE_USE_NORMAL_RESIDUAL;
    if (!fsSettings["line_use_epipolar_residual"].empty()) fsSettings["line_use_epipolar_residual"] >> LINE_USE_EPIPOLAR_RESIDUAL;
    if (!fsSettings["line_disable_filtering"].empty()) fsSettings["line_disable_filtering"] >> LINE_DISABLE_FILTERING;
    if (!fsSettings["line_min_used_for_alignment"].empty()) fsSettings["line_min_used_for_alignment"] >> LINE_MIN_USED_FOR_ALIGNMENT;

    if (!fsSettings["epipolar_min_track_count"].empty()) fsSettings["epipolar_min_track_count"] >> EPIPOLAR_MIN_TRACK_COUNT;
    if (!fsSettings["epipolar_min_pair_coverage"].empty()) fsSettings["epipolar_min_pair_coverage"] >> EPIPOLAR_MIN_PAIR_COVERAGE;
    if (!fsSettings["epipolar_min_match_quality"].empty()) fsSettings["epipolar_min_match_quality"] >> EPIPOLAR_MIN_MATCH_QUALITY;
    if (!fsSettings["epipolar_min_obs_quality"].empty()) fsSettings["epipolar_min_obs_quality"] >> EPIPOLAR_MIN_OBS_QUALITY;
    if (!fsSettings["epipolar_min_score"].empty()) fsSettings["epipolar_min_score"] >> EPIPOLAR_MIN_SCORE;
    if (!fsSettings["epipolar_max_lines_global"].empty()) fsSettings["epipolar_max_lines_global"] >> EPIPOLAR_MAX_LINES_GLOBAL;
    if (EPIPOLAR_MAX_LINES_GLOBAL < 0) EPIPOLAR_MAX_LINES_GLOBAL = 0;

    VP_USE_RESIDUAL = 0.0; 
    if (!fsSettings["vp_use_residual"].empty()) fsSettings["vp_use_residual"] >> VP_USE_RESIDUAL;
    ACC_CHECK_ENABLE = 1; 
    if (!fsSettings["acc_check_enable"].empty()) {
        fsSettings["acc_check_enable"] >> ACC_CHECK_ENABLE;
        ACC_CHECK_ENABLE = (ACC_CHECK_ENABLE > 0) ? 1 : 0;
    }
    ACC_CHECK_MAX_NEAR_G_SEGMENTS = 1; 
    if (!fsSettings["acc_check_max_near_g_segments"].empty()) {
        fsSettings["acc_check_max_near_g_segments"] >> ACC_CHECK_MAX_NEAR_G_SEGMENTS;
        if (ACC_CHECK_MAX_NEAR_G_SEGMENTS < 0) ACC_CHECK_MAX_NEAR_G_SEGMENTS = 0;
    }
    CAM_NAMES.push_back(config_file);
    STEREO_TRACK = false;
    PUB_THIS_FRAME = false;

    if (FREQ == 0){
        FREQ = 10;
    }
    fsSettings.release();

    cout << "1 readParameters:  "
        <<  "\n  INIT_DEPTH: " << INIT_DEPTH
        <<  "\n  MIN_PARALLAX: " << MIN_PARALLAX
        <<  "\n  ACC_N: " <<ACC_N
        <<  "\n  ACC_W: " <<ACC_W
        <<  "\n  GYR_N: " <<GYR_N
        <<  "\n  GYR_W: " <<GYR_W
        <<  "\n  RIC:   " << RIC[0]
        <<  "\n  TIC:   " <<TIC[0].transpose()
        <<  "\n  G:     " <<G.transpose()
        <<  "\n  OUTPUT_PATH:"<<OUTPUT_PATH
        <<  "\n  BIAS_ACC_THRESHOLD:"<<BIAS_ACC_THRESHOLD
        <<  "\n  BIAS_GYR_THRESHOLD:"<<BIAS_GYR_THRESHOLD
        <<  "\n  SOLVER_TIME:"<<SOLVER_TIME
        <<  "\n  NUM_ITERATIONS:"<<NUM_ITERATIONS
        <<  "\n  ESTIMATE_EXTRINSIC:"<<ESTIMATE_EXTRINSIC
        <<  "\n  ESTIMATE_TD:"<<ESTIMATE_TD
        <<  "\n  ROLLING_SHUTTER:"<<ROLLING_SHUTTER
        <<  "\n  ROW:"<<ROW
        <<  "\n  COL:"<<COL
        <<  "\n  TD:"<<TD
        <<  "\n  TR:"<<TR
        <<  "\n  FOCAL_LENGTH:"<<FOCAL_LENGTH
        <<  "\n  IMAGE_TOPIC:"<<IMAGE_TOPIC
        <<  "\n  IMU_TOPIC:"<<IMU_TOPIC
        <<  "\n  FISHEYE_MASK:"<<FISHEYE_MASK
        <<  "\n  CAM_NAMES[0]:"<<CAM_NAMES[0]
        <<  "\n  MAX_CNT:"<<MAX_CNT
        <<  "\n  MIN_DIST:"<<MIN_DIST
        <<  "\n  FREQ:"<<FREQ
        <<  "\n  CAMERA FREQ:"<<CAMERA_FREQ
        <<  "\n  IMU FREQ:"<< IMU_FREQ
        <<  "\n  F_THRESHOLD:"<<F_THRESHOLD
        <<  "\n  SHOW_TRACK:"<<SHOW_TRACK
        <<  "\n  STEREO_TRACK:"<<STEREO_TRACK
        <<  "\n  EQUALIZE:"<<EQUALIZE
        <<  "\n  FISHEYE:"<<FISHEYE
        <<  "\n  PUB_THIS_FRAME:"<<PUB_THIS_FRAME
        <<  "\n  LINE_MAX_CNT:"<<LINE_MAX_CNT
        <<  "\n  LINE_MIN_LENGTH:"<<LINE_MIN_LENGTH
        <<  "\n  LINE_NMS_DIST:"<<LINE_NMS_DIST
        <<  "\n  LINE_NMS_ANGLE_DEG:"<<LINE_NMS_ANGLE_DEG
        << "\n  LINE_GRADIENT_THRESHOLD:"<<LINE_GRADIENT_THRESHOLD

        <<  "\n  LINE_RESIDUAL_WEIGHT:"<<LINE_RESIDUAL_WEIGHT
        <<  "\n  LINE_NORMAL_WEIGHT:"<<LINE_NORMAL_WEIGHT
        <<  "\n  LINE_EPIPOLAR_WEIGHT:"<<LINE_EPIPOLAR_WEIGHT
        <<  "\n  LINE_USE_NORMAL_RESIDUAL:"<<LINE_USE_NORMAL_RESIDUAL
        <<  "\n  LINE_USE_EPIPOLAR_RESIDUAL:"<<LINE_USE_EPIPOLAR_RESIDUAL
        <<  "\n  LINE_DISABLE_FILTERING:"<<LINE_DISABLE_FILTERING
        <<  "\n  LINE_MIN_USED_FOR_ALIGNMENT:"<<LINE_MIN_USED_FOR_ALIGNMENT
        <<  "\n  EPIPOLAR_MIN_TRACK_COUNT:"<<EPIPOLAR_MIN_TRACK_COUNT
        <<  "\n  EPIPOLAR_MIN_PAIR_COVERAGE:"<<EPIPOLAR_MIN_PAIR_COVERAGE
        <<  "\n  EPIPOLAR_MIN_MATCH_QUALITY:"<<EPIPOLAR_MIN_MATCH_QUALITY
        <<  "\n  EPIPOLAR_MIN_OBS_QUALITY:"<<EPIPOLAR_MIN_OBS_QUALITY
        <<  "\n  EPIPOLAR_MIN_SCORE:"<<EPIPOLAR_MIN_SCORE
        <<  "\n  EPIPOLAR_MAX_LINES_GLOBAL:"<<EPIPOLAR_MAX_LINES_GLOBAL
        <<  "\n  VP_USE_RESIDUAL:"<<VP_USE_RESIDUAL
        <<  "\n  ACC_CHECK_ENABLE:"<<ACC_CHECK_ENABLE
        <<  "\n  ACC_CHECK_MAX_NEAR_G_SEGMENTS:"<<ACC_CHECK_MAX_NEAR_G_SEGMENTS
    << endl;

}
