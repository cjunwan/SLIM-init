
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <eigen3/Eigen/Dense>
#include <map>
#include <vector>

#include "camodocal/camera_models/CameraFactory.h"
#include "parameters.h"
#include "featureManager.h"
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/edge_drawing.hpp>

namespace DRT {

using namespace cv::line_descriptor;

class LineFeatureTracker {
public:
    LineFeatureTracker();

    void readImage(const cv::Mat& _img, double _cur_time, const Eigen::Matrix3d& R_pred = Eigen::Matrix3d::Identity());
    void readIntrinsicParameter(const std::string& calib_file);

    const std::map<FeatureID, SFMLine>& getTrackedLines() const { return tracked_lines; }
    const std::map<int, Eigen::Vector3d>& getVanishingPoints() const { return cur_vps; }
    const std::vector<KeyLine>& getCurrentKeylines() const { return cur_keylines; }
    std::vector<KeyLine> getMatchedLines() const;
    std::vector<KeyLine> getVPAssignedLines() const;
    struct LineFeatureStats {
        int num_extracted_lines = 0;           
        int num_matched_lines = 0;             
        double matching_rate = 0.0;             
        int num_vp_lines = 0;                   
        double vp_inlier_ratio = 0.0;           
        double avg_vp_line_length = 0.0;        
        double avg_vp_angle_error = 0.0;       
        bool vp_valid = false;                  
        Eigen::Vector2d vp_position;             
        double vp_position_change = 0.0;       
    };
    LineFeatureStats getCurrentFrameStats() const;
    bool projectVanishingPointToImage(const Eigen::Vector3d& vp_dir, Eigen::Vector2d& img_pt) const;
    camodocal::CameraPtr getCamera() const { return m_camera; }
    int getVPTrackId() const { return cur_vp_track_id_; }
    int getVPTrackAge() const { return cur_vp_track_age_; }
    bool isVPSwitchDetected() const { return vp_switch_detected_; }
    int getVPInlierCount() const { return last_vp_inlier_count_; }
    double getVPMedianAbsDot() const { return last_vp_median_abs_dot_; }
    double getVPP90AbsDot() const { return last_vp_p90_abs_dot_; }
    double getVPCondRatio() const { return last_vp_cond_ratio_; }
    bool isVPQualityValid() const { return last_vp_quality_valid_; }
    const cv::Mat& getCurrentImage() const { return cur_img; }
    const std::vector<KeyLine>& getKeylines() const { return cur_keylines; }
    cv::Mat getUndistortedImage(const cv::Mat& raw_img);
    std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& pts) const;
    const std::vector<cv::Vec4f>& getRawSegments() const { return debug_raw_segments_; }

private:
    void ensureUndistortMaps(const cv::Size& image_size);
    void ensureEdlinesConfigured();

    void lineExtraction(const cv::Mat& img, std::vector<KeyLine>& keylines, cv::Mat& descriptors);
    void lineMatching(const cv::Mat& prev_desc, const cv::Mat& cur_desc, 
                      const std::vector<KeyLine>& prev_lines, const std::vector<KeyLine>& cur_lines,
                      std::vector<cv::DMatch>& good_matches,
                      const cv::Mat& cur_img_raw,
                      const Eigen::Matrix3d& R_pred = Eigen::Matrix3d::Identity());
    void updateTrackedLines(const std::vector<KeyLine>& new_lines, const std::vector<cv::DMatch>& matches);
    void predictLinesWithLK(const cv::Mat& prev_img, const cv::Mat& cur_img, 
                           const std::vector<KeyLine>& prev_lines, 
                           std::vector<KeyLine>& predicted_lines, 
                           std::vector<bool>& is_valid_prediction);
    void samplePointsOnLine(const KeyLine& line, std::vector<cv::Point2f>& points);
    KeyLine fitLineFromPoints(const std::vector<cv::Point2f>& points, const KeyLine& original_line);

    cv::Mat prev_img, cur_img;
    std::vector<KeyLine> prev_keylines, cur_keylines;
    cv::Mat prev_descriptors, cur_descriptors;
    std::vector<bool> cur_line_matched;  

    std::map<FeatureID, SFMLine> tracked_lines;
    std::map<int, Eigen::Vector3d> cur_vps;
    std::map<int, Eigen::Vector3d> prev_vps;  
    bool cur_vp_found_this_frame = false;
    int cur_vp_track_id_ = 0;          
    int cur_vp_track_age_ = 0;         
    bool vp_switch_detected_ = false;  
    static constexpr double VP_TRACK_KEEP_THRESHOLD = 0.97;    
    static constexpr double VP_TRACK_SWITCH_THRESHOLD = 0.90;  
    int last_vp_inlier_count_ = 0;
    double last_vp_median_abs_dot_ = 1.0;
    double last_vp_p90_abs_dot_ = 1.0;
    double last_vp_cond_ratio_ = 0.0;
    bool last_vp_quality_valid_ = false;
    


    camodocal::CameraPtr m_camera;
    double cur_time;
    
    static int n_id;
    
    cv::Ptr<cv::ximgproc::EdgeDrawing> edlines_detector_;
    bool edlines_configured_ = false;
    std::vector<cv::Vec4f> debug_raw_segments_;
    std::vector<Eigen::Vector3d> debug_vp_candidates_;
    cv::Mat undist_map1_;
    cv::Mat undist_map2_;
    cv::Mat undist_K_;  
    cv::Size undist_image_size_{0, 0};
    static constexpr double VP_SMOOTHING_ALPHA = 0.7;  
    static constexpr double VP_SIMILARITY_THRESHOLD = 0.95;  
    struct LineNormalInfo {
        Eigen::Vector3d normal;  
        double length;           
        int line_idx;            
    };
    struct SphereCell {
        Eigen::Vector3d center;  
        double vote;             
        std::vector<int> line_indices;  
    };
    struct VPCandidate {
        Eigen::Vector3d direction;
        double score;
        std::vector<int> inlier_indices;
    };
    void calculateVanishingPointsRobust(std::vector<KeyLine>& keylines, const Eigen::Matrix3d& R_imu = Eigen::Matrix3d::Identity());
    void buildSphereCells();
    std::vector<LineNormalInfo> computeLineNormals(const std::vector<KeyLine>& keylines);
    void accumulateVotes(const std::vector<LineNormalInfo>& line_normals, const std::vector<bool>& is_line_used);
    std::vector<VPCandidate> detectVPPeaks(const std::vector<LineNormalInfo>& line_normals);
    Eigen::Vector3d refineVPWithWLS(const std::vector<LineNormalInfo>& line_normals,
                                     const std::vector<int>& inlier_indices);
    Eigen::Vector3d filterVPWithIMU(const Eigen::Vector3d& vp_measured,
                                     const Eigen::Matrix3d& R_imu);
    void updateVPTrackId(const Eigen::Vector3d& vp_current, const Eigen::Matrix3d& R_pred);
    std::vector<SphereCell> sphere_cells_;
    bool sphere_initialized_ = false;
    static constexpr int SPHERE_SUBDIVISION_LEVEL = 3;  
    static constexpr double VP_INLIER_THRESHOLD = 0.03;  
    static constexpr int VP_MIN_INLIERS = 3;  
};

} 
