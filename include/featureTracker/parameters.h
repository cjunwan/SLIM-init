#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>
#include <fstream>
const int NUM_OF_CAM = 1;

extern int FOCAL_LENGTH;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int FREQ;
extern int CAMERA_FREQ;
extern int IMU_FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern bool STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;
extern int LINE_MAX_CNT;          
extern double LINE_MIN_LENGTH;    
extern double LINE_NMS_DIST;      
extern double LINE_NMS_ANGLE_DEG; 
extern int LINE_GRADIENT_THRESHOLD;   
extern double LINE_MATCH_MAX_DIST;    
extern double LINE_MATCH_MAX_ANGLE;   
extern double LINE_MATCH_MIN_OVERLAP; 

extern double LINE_RESIDUAL_WEIGHT;   
extern double LINE_NORMAL_WEIGHT;     
extern double LINE_EPIPOLAR_WEIGHT;   
extern int LINE_USE_NORMAL_RESIDUAL;  
extern int LINE_USE_EPIPOLAR_RESIDUAL; 
extern int LINE_DISABLE_FILTERING;    
extern int LINE_MIN_USED_FOR_ALIGNMENT; 

extern int EPIPOLAR_MIN_TRACK_COUNT;
extern double EPIPOLAR_MIN_PAIR_COVERAGE;
extern double EPIPOLAR_MIN_MATCH_QUALITY;
extern double EPIPOLAR_MIN_OBS_QUALITY;
extern double EPIPOLAR_MIN_SCORE;
extern int EPIPOLAR_MAX_LINES_GLOBAL;

extern double VP_USE_RESIDUAL;          
extern int ACC_CHECK_ENABLE;            
extern int ACC_CHECK_MAX_NEAR_G_SEGMENTS; 
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 6,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 3
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
