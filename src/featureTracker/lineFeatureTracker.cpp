
#include "lineFeatureTracker.h"
#include "parameters.h"
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric> 
#include <array>   

namespace DRT {

int LineFeatureTracker::n_id = 0;

LineFeatureTracker::LineFeatureTracker() {
    n_id = 0; 
}

void LineFeatureTracker::readIntrinsicParameter(const std::string& calib_file) {
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void LineFeatureTracker::ensureEdlinesConfigured() {
    if (edlines_configured_ && edlines_detector_) return;
    
    edlines_detector_ = cv::ximgproc::createEdgeDrawing();
    cv::ximgproc::EdgeDrawing::Params params;
    params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::PREWITT;
    params.GradientThresholdValue = LINE_GRADIENT_THRESHOLD;
    params.AnchorThresholdValue = 8;
    params.ScanInterval = 1;
    params.MinPathLength = 10;
    params.MinLineLength = static_cast<int>(LINE_MIN_LENGTH);
    params.NFAValidation = true;   
    params.PFmode = false;
    params.Sigma = 1.0;
    edlines_detector_->setParams(params);
    
    edlines_configured_ = true;
}

void LineFeatureTracker::ensureUndistortMaps(const cv::Size& image_size) {
    if (!m_camera) return;
    if (image_size.width <= 0 || image_size.height <= 0) return;

    if (undist_image_size_ == image_size && !undist_map1_.empty() && !undist_map2_.empty()) {
        return; 
    }

    undist_map1_.release();
    undist_map2_.release();
    m_camera->initUndistortRectifyMap(
        undist_map1_, undist_map2_,
        -1.0f, -1.0f,
        image_size,
        -1.0f, -1.0f,
        cv::Mat::eye(3, 3, CV_32F)
    );
    undist_K_ = m_camera->initUndistortRectifyMap(undist_map1_, undist_map2_,
        -1.0f, -1.0f,
        image_size,
        -1.0f, -1.0f,
        cv::Mat::eye(3, 3, CV_32F)
    );
    
    if (undist_K_.type() != CV_32FC1) {
        undist_K_.convertTo(undist_K_, CV_32FC1);
    }

    undist_image_size_ = image_size;
}

void LineFeatureTracker::readImage(const cv::Mat& _img, double _cur_time, const Eigen::Matrix3d& R_pred) {
    cur_time = _cur_time;
    cv::Mat gray, img;  
    if (_img.empty()) return;

    if (_img.channels() == 3) {
        cv::cvtColor(_img, gray, cv::COLOR_BGR2GRAY);
    } else if (_img.channels() == 4) {
        cv::cvtColor(_img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        if (_img.depth() != CV_8U) {
            double minVal = 0.0, maxVal = 0.0;
            cv::minMaxLoc(_img, &minVal, &maxVal);
            double scale = (maxVal > minVal) ? 255.0 / (maxVal - minVal) : 1.0;
            double shift = -minVal * scale;
            _img.convertTo(gray, CV_8U, scale, shift);
        } else {
            gray = _img;
        }
    }
    cv::Mat undist_gray;
    bool undist_applied = false;
    if (m_camera) {
        ensureUndistortMaps(gray.size());
        if (!undist_map1_.empty() && !undist_map2_.empty()) {
            cv::remap(gray, undist_gray, undist_map1_, undist_map2_, cv::INTER_LINEAR);
            undist_applied = true;
        }
    }
    if (!undist_applied) {
        undist_gray = gray;
    }
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(undist_gray, img);  
    }
    std::vector<KeyLine> new_keylines;
    cv::Mat new_descriptors;
    ensureEdlinesConfigured();
    lineExtraction(img, new_keylines, new_descriptors);
    std::vector<cv::DMatch> good_matches;
    if (!prev_keylines.empty() && !new_keylines.empty()) {
        lineMatching(prev_descriptors, new_descriptors, prev_keylines, new_keylines, good_matches, img, R_pred);
    }
    updateTrackedLines(new_keylines, good_matches);
    cur_line_matched.clear();
    cur_line_matched.resize(cur_keylines.size(), false);
    for (const auto& m : good_matches) {
        int cur_idx = m.trainIdx;
        if (cur_idx >= 0 && cur_idx < static_cast<int>(cur_keylines.size())) {
            int prev_idx = m.queryIdx;
            if (prev_idx >= 0 && prev_idx < static_cast<int>(prev_keylines.size())) {
                if (prev_keylines[prev_idx].class_id != -1) {
                    cur_line_matched[cur_idx] = true;
                }
            }
        }
    }
    calculateVanishingPointsRobust(cur_keylines, R_pred);
    cur_line_matched.clear();
    cur_line_matched.resize(cur_keylines.size(), false);
    for (const auto& m : good_matches) {
        int cur_idx = m.trainIdx;
        if (cur_idx >= 0 && cur_idx < static_cast<int>(cur_keylines.size())) {
            int prev_idx = m.queryIdx;
            if (prev_idx >= 0 && prev_idx < static_cast<int>(prev_keylines.size())) {
                if (prev_keylines[prev_idx].class_id != -1) {
                    cur_line_matched[cur_idx] = true;
                }
            }
        }
    }
    for (const auto& kl : cur_keylines) {
        if (kl.class_id != -1 && kl.octave != -1) { 
             if (tracked_lines.count(kl.class_id) && tracked_lines.at(kl.class_id).obs.count(cur_time)) {
                 tracked_lines.at(kl.class_id).obs.at(cur_time).vp_id = kl.octave;
             }
        }
    }
    prev_img = cur_img.clone();  
    cur_img = img.clone();       
    prev_keylines = cur_keylines;
    prev_descriptors = new_descriptors.clone();
    prev_vps = cur_vps;  
}

void LineFeatureTracker::lineExtraction(const cv::Mat& img, std::vector<KeyLine>& keylines, cv::Mat& descriptors) {
    struct CandidateLine {
        KeyLine kl;
        double score = 0.0; 
        float angle = 0.0f; 
        cv::Point2f mid;
    };

    std::vector<CandidateLine> candidates;
    candidates.reserve(300);
    edlines_detector_->detectEdges(img);
    std::vector<cv::Vec4f> raw_lines;
    edlines_detector_->detectLines(raw_lines);
    debug_raw_segments_ = raw_lines;
    for (const auto& seg : raw_lines) {
        const double len = std::hypot(seg[0] - seg[2], seg[1] - seg[3]);

        CandidateLine c;
        c.kl.startPointX = seg[0]; c.kl.startPointY = seg[1];
        c.kl.endPointX = seg[2];   c.kl.endPointY = seg[3];
        c.kl.lineLength = static_cast<float>(len);
        c.kl.class_id = -1;
        c.kl.octave = 0;
        c.score = len;  
        c.kl.response = static_cast<float>(c.score);
        c.mid = (c.kl.getStartPoint() + c.kl.getEndPoint()) * 0.5f;
        c.angle = std::atan2(c.kl.endPointY - c.kl.startPointY, c.kl.endPointX - c.kl.startPointX);
        candidates.push_back(c);
    }
    
    std::vector<CandidateLine> kept;
    if (LINE_DISABLE_FILTERING) {
        kept = candidates;
    } else {
        const float PI = 3.14159265358979323846f;
        const float ang_th = static_cast<float>(LINE_NMS_ANGLE_DEG) * PI / 180.0f;
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidateLine& a, const CandidateLine& b) { return a.score > b.score; });

        kept.reserve(candidates.size());
        for (const auto& c : candidates) {
            bool suppressed = false;
            for (const auto& k : kept) {
                if (cv::norm(c.mid - k.mid) > static_cast<float>(LINE_NMS_DIST)) continue;
                float d = std::fabs(c.angle - k.angle);
                d = std::min(d, 2.0f * PI - d);
                if (d >= ang_th) continue;
                cv::Point2f ks = k.kl.getStartPoint();
                cv::Point2f ke = k.kl.getEndPoint();
                cv::Point2f cs = c.kl.getStartPoint();
                cv::Point2f ce = c.kl.getEndPoint();
                cv::Point2f dir = ke - ks;
                float dir_norm = std::sqrt(dir.dot(dir));
                if (dir_norm < 1e-3f) continue;
                dir *= (1.0f / dir_norm);

                auto proj = [&](const cv::Point2f& p) -> float { return (p - ks).dot(dir); };
                float k0 = std::min(proj(ks), proj(ke));
                float k1 = std::max(proj(ks), proj(ke));
                float c0 = std::min(proj(cs), proj(ce));
                float c1 = std::max(proj(cs), proj(ce));
                float inter = std::max(0.0f, std::min(k1, c1) - std::max(k0, c0));
                float min_len = std::min(k1 - k0, c1 - c0);
                float overlap_ratio = (min_len > 1e-3f) ? (inter / min_len) : 0.0f;
                if (overlap_ratio > 0.5f) { suppressed = true; break; }
            }
            if (!suppressed) kept.push_back(c);
        }
    }
    if (LINE_MAX_CNT > 0 && static_cast<int>(kept.size()) > LINE_MAX_CNT) {
        kept.resize(static_cast<size_t>(LINE_MAX_CNT));
    }

    keylines.clear();
    keylines.reserve(kept.size());
    for (auto& c : kept) keylines.push_back(c.kl);
}
float getGradient(const cv::Mat& img, const KeyLine& kl) {
    cv::Point2f start = kl.getStartPoint();
    cv::Point2f end = kl.getEndPoint();
    cv::Point2f dir = end - start;
    float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
    if (len < 1e-3) return 0.0f;
    dir.x /= len;
    dir.y /= len;
    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f mid = (start + end) * 0.5f;
    float offset = 2.0f;
    cv::Point2f p_left = mid - normal * offset;
    cv::Point2f p_right = mid + normal * offset;
    auto safe_get = [&](const cv::Point2f& p) -> float {
        int x = cvRound(p.x);
        int y = cvRound(p.y);
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
            return (float)img.at<uchar>(y, x);
        return 0.0f;
    };
    
    float val_left = safe_get(p_left);
    float val_right = safe_get(p_right);
    
    return val_right - val_left;
}
void LineFeatureTracker::samplePointsOnLine(const KeyLine& line, std::vector<cv::Point2f>& points) {
    float len = line.lineLength;
    int num_samples = std::max(5, static_cast<int>(len / 10.0f)); 
    
    cv::Point2f start(line.startPointX, line.startPointY);
    cv::Point2f end(line.endPointX, line.endPointY);
    cv::Point2f vec = end - start;
    
    for (int i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / (num_samples - 1);
        points.push_back(start + vec * t);
    }
}
KeyLine LineFeatureTracker::fitLineFromPoints(const std::vector<cv::Point2f>& points, const KeyLine& original_line) {
    KeyLine fitting_line = original_line;
    if (points.size() < 2) return fitting_line;

    cv::Vec4f line_params;
    cv::fitLine(points, line_params, cv::DIST_L2, 0, 0.01, 0.01);
    
    cv::Point2f dir(line_params[0], line_params[1]);
    cv::Point2f center(line_params[2], line_params[3]);
    float norm = std::sqrt(dir.x*dir.x + dir.y*dir.y);
    if (norm < 1e-6) return fitting_line; 
    dir *= (1.0f / norm);
    float min_t = 1e9, max_t = -1e9;
    for (const auto& p : points) {
        float t = (p.x - center.x) * dir.x + (p.y - center.y) * dir.y;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }
    
    cv::Point2f p_start = center + dir * min_t;
    cv::Point2f p_end = center + dir * max_t;
    
    fitting_line.startPointX = p_start.x;
    fitting_line.startPointY = p_start.y;
    fitting_line.endPointX = p_end.x;
    fitting_line.endPointY = p_end.y;
    fitting_line.lineLength = cv::norm(p_end - p_start);
    
    return fitting_line;
}

void LineFeatureTracker::predictLinesWithLK(const cv::Mat& prev_img, const cv::Mat& cur_img, 
                                            const std::vector<KeyLine>& prev_lines, 
                                            std::vector<KeyLine>& predicted_lines, 
                                            std::vector<bool>& is_valid_prediction) {
    if (prev_img.empty() || cur_img.empty()) return;

    predicted_lines.resize(prev_lines.size());
    is_valid_prediction.assign(prev_lines.size(), false);

    std::vector<cv::Point2f> all_prev_pts;
    std::vector<int> pt_to_line_idx;
    for (size_t i = 0; i < prev_lines.size(); ++i) {
        std::vector<cv::Point2f> pts;
        samplePointsOnLine(prev_lines[i], pts);
        for (const auto& p : pts) {
            all_prev_pts.push_back(p);
            pt_to_line_idx.push_back(i);
        }
    }
    
    if (all_prev_pts.empty()) return;
    std::vector<cv::Point2f> all_cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);
    
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, all_prev_pts, all_cur_pts, status, err, cv::Size(21, 21), 3, criteria);
    std::vector<std::vector<cv::Point2f>> line_tracked_pts(prev_lines.size());
    for (size_t k = 0; k < all_prev_pts.size(); ++k) {
        if (status[k]) {
            int line_idx = pt_to_line_idx[k];
            line_tracked_pts[line_idx].push_back(all_cur_pts[k]);
        }
    }
    for (size_t i = 0; i < prev_lines.size(); ++i) {
        if (line_tracked_pts[i].size() >= 2) {
             predicted_lines[i] = fitLineFromPoints(line_tracked_pts[i], prev_lines[i]);
             is_valid_prediction[i] = true;
        } else {
             predicted_lines[i] = prev_lines[i]; 
             is_valid_prediction[i] = false;
        }
    }
}

void LineFeatureTracker::lineMatching(const cv::Mat& prev_desc, const cv::Mat& cur_desc,
                                      const std::vector<KeyLine>& prev_lines, const std::vector<KeyLine>& cur_lines,
                                      std::vector<cv::DMatch>& good_matches,
                                      const cv::Mat& cur_img_raw,
                                      const Eigen::Matrix3d& R_pred) {
    good_matches.clear();
    if (prev_lines.empty() || cur_lines.empty()) return;
    const float max_dist = static_cast<float>(LINE_MATCH_MAX_DIST);
    const float max_angle_diff = static_cast<float>(LINE_MATCH_MAX_ANGLE) * CV_PI / 180.0f;
    const float min_overlap_ratio = static_cast<float>(LINE_MATCH_MIN_OVERLAP);
    std::vector<KeyLine> predicted_prev_lines(prev_lines.size());
    std::vector<cv::Point2f> predicted_dirs(prev_lines.size());
    std::vector<bool> lk_valid(prev_lines.size(), false);
    if (!this->cur_img.empty() && !cur_img_raw.empty()) {
        predictLinesWithLK(this->cur_img, cur_img_raw, prev_lines, predicted_prev_lines, lk_valid);
    }
    
    for(size_t i=0; i<prev_lines.size(); ++i) {
        if (!lk_valid[i]) {
            predicted_prev_lines[i] = prev_lines[i];
        }
        float dx = predicted_prev_lines[i].endPointX - predicted_prev_lines[i].startPointX;
        float dy = predicted_prev_lines[i].endPointY - predicted_prev_lines[i].startPointY;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len > 1e-3) {
            predicted_dirs[i] = cv::Point2f(dx/len, dy/len);
        } else {
            predicted_dirs[i] = cv::Point2f(0,0);
        }
    }

    for (int i = 0; i < static_cast<int>(prev_lines.size()); ++i) {
        const auto& l1 = prev_lines[i]; 
        const auto& l1_pred = predicted_prev_lines[i];
        if (predicted_dirs[i] == cv::Point2f(0,0)) continue; 

        const cv::Point2f& dir1 = predicted_dirs[i];
        const float min_cos = std::cos(max_angle_diff);
        float grad1 = getGradient(cur_img, l1);

        float best_score = -1.0f; 
        int best_j = -1;

        for (int j = 0; j < static_cast<int>(cur_lines.size()); ++j) {
            const auto& l2 = cur_lines[j];
            float dx2 = l2.endPointX - l2.startPointX;
            float dy2 = l2.endPointY - l2.startPointY;
            float len2 = l2.lineLength;
            if (len2 < 1e-3) continue;

            cv::Point2f dir2(dx2/len2, dy2/len2);            
            float dot = dir1.dot(dir2);
            if (std::abs(dot) < min_cos) continue; 
            cv::Point2f mid2 = (l2.getStartPoint() + l2.getEndPoint()) * 0.5f;
            
            cv::Point2f V = mid2 - l1_pred.getStartPoint();
            float t = V.dot(dir1);
            cv::Point2f proj = l1_pred.getStartPoint() + dir1 * t;
            float perp_dist = cv::norm(mid2 - proj);
            
            if (perp_dist > max_dist) continue;
            cv::Point2f V_start = l2.getStartPoint() - l1_pred.getStartPoint();
            cv::Point2f V_end = l2.getEndPoint() - l1_pred.getStartPoint();
            float t_start = V_start.dot(dir1);
            float t_end = V_end.dot(dir1);
            
            if (t_start > t_end) std::swap(t_start, t_end);
            
            float l1_len = std::sqrt(std::pow(l1_pred.endPointX - l1_pred.startPointX, 2) + 
                                     std::pow(l1_pred.endPointY - l1_pred.startPointY, 2));
            
            float intersect_start = std::max(0.0f, t_start);
            float intersect_end = std::min(l1_len, t_end);
            
            float overlap_len = std::max(0.0f, intersect_end - intersect_start);
            float overlap_ratio = overlap_len / std::min(l1_len, len2);
            
            if (overlap_ratio < min_overlap_ratio) continue;
            float grad2 = getGradient(cur_img_raw, l2);
            float sign_dot = (dot >= 0) ? 1.0f : -1.0f;
            float dist_score = 1.0f - (perp_dist / max_dist);
            float angle_score = (std::abs(dot) - min_cos) / (1.0f - min_cos);
            float score = 0.5f * dist_score + 0.5f * angle_score;
            
            if (score > best_score) {
                best_score = score;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            good_matches.emplace_back(i, best_j, 1.0f - best_score);
        }
    }
}

void LineFeatureTracker::updateTrackedLines(const std::vector<KeyLine>& new_lines, const std::vector<cv::DMatch>& matches) {
    cur_keylines = new_lines;
    std::vector<bool> is_matched(cur_keylines.size(), false);
    auto compute_obs_quality = [](const KeyLine& kl) -> double {
        const double len = std::max(0.0, static_cast<double>(kl.lineLength));
        const double len_ref = std::max(1.0, LINE_MIN_LENGTH * 3.0);
        const double len_score = std::clamp(len / len_ref, 0.0, 1.0);
        const double det_raw = std::max(0.0, static_cast<double>(kl.response));
        const double det_score = 1.0 - std::exp(-det_raw / 20.0);
        const double q = 0.9 * len_score + 0.1 * std::clamp(det_score, 0.0, 1.0);
        return std::clamp(q, 0.0, 1.0);
    };
    for (const auto& m : matches) {
        int prev_idx = m.queryIdx;
        int cur_idx = m.trainIdx;
        int id = prev_keylines[prev_idx].class_id;

        if (id != -1) {
            cur_keylines[cur_idx].class_id = id;
            is_matched[cur_idx] = true;
            Eigen::Vector3d sp, ep;
            double sx = cur_keylines[cur_idx].startPointX;
            double sy = cur_keylines[cur_idx].startPointY;
            double ex = cur_keylines[cur_idx].endPointX;
            double ey = cur_keylines[cur_idx].endPointY;

            if (!undist_map1_.empty()) {
                if (undist_map1_.type() == CV_32FC1 && undist_map2_.type() == CV_32FC1) {
                    int u_s = std::max(0, std::min((int)std::round(sx), undist_map1_.cols - 1));
                    int v_s = std::max(0, std::min((int)std::round(sy), undist_map1_.rows - 1));
                    sx = undist_map1_.at<float>(v_s, u_s);
                    sy = undist_map2_.at<float>(v_s, u_s);

                    int u_e = std::max(0, std::min((int)std::round(ex), undist_map1_.cols - 1));
                    int v_e = std::max(0, std::min((int)std::round(ey), undist_map1_.rows - 1));
                    ex = undist_map1_.at<float>(v_e, u_e);
                    ey = undist_map2_.at<float>(v_e, u_e);
                }
            }

            m_camera->liftProjective(Eigen::Vector2d(sx, sy), sp);
            m_camera->liftProjective(Eigen::Vector2d(ex, ey), ep);

            const double match_score = std::clamp(1.0 - static_cast<double>(m.distance), 0.0, 1.0);
            const double obs_quality = compute_obs_quality(cur_keylines[cur_idx]);
            tracked_lines[id].obs[cur_time] = LinePerFrame(
                sp, ep,
                static_cast<double>(cur_keylines[cur_idx].lineLength),
                static_cast<double>(cur_keylines[cur_idx].response),
                match_score,
                obs_quality);
            tracked_lines[id].track_count++;
        }
    }
    for (size_t i = 0; i < cur_keylines.size(); ++i) {
        if (!is_matched[i]) {
            int new_id = n_id++;
            cur_keylines[i].class_id = new_id;
            
            SFMLine sfm_line(new_id);
            Eigen::Vector3d sp, ep;
            
            double sx = cur_keylines[i].startPointX;
            double sy = cur_keylines[i].startPointY;
            double ex = cur_keylines[i].endPointX;
            double ey = cur_keylines[i].endPointY;
            
            if (!undist_map1_.empty()) {
                if (undist_map1_.type() == CV_32FC1 && undist_map2_.type() == CV_32FC1) {
                    int u_s = std::max(0, std::min((int)std::round(sx), undist_map1_.cols - 1));
                    int v_s = std::max(0, std::min((int)std::round(sy), undist_map1_.rows - 1));
                    sx = undist_map1_.at<float>(v_s, u_s);
                    sy = undist_map2_.at<float>(v_s, u_s);

                    int u_e = std::max(0, std::min((int)std::round(ex), undist_map1_.cols - 1));
                    int v_e = std::max(0, std::min((int)std::round(ey), undist_map1_.rows - 1));
                    ex = undist_map1_.at<float>(v_e, u_e);
                    ey = undist_map2_.at<float>(v_e, u_e);
                }
            }

            m_camera->liftProjective(Eigen::Vector2d(sx, sy), sp);
            m_camera->liftProjective(Eigen::Vector2d(ex, ey), ep);
            
            const double obs_quality = compute_obs_quality(cur_keylines[i]);
            sfm_line.obs[cur_time] = LinePerFrame(
                sp, ep,
                static_cast<double>(cur_keylines[i].lineLength),
                static_cast<double>(cur_keylines[i].response),
                0.0,
                obs_quality);
            sfm_line.track_count = 1;
            tracked_lines[new_id] = sfm_line;
        }
    }
}

bool LineFeatureTracker::projectVanishingPointToImage(const Eigen::Vector3d& vp_dir, Eigen::Vector2d& img_pt) const {
    if (!m_camera) return false;
    if (vp_dir(2) <= 0) {
        return false;
    }
    double nx = vp_dir(0) / vp_dir(2);
    double ny = vp_dir(1) / vp_dir(2);
    
    if (!undist_K_.empty()) {
        float fx = undist_K_.at<float>(0, 0);
        float fy = undist_K_.at<float>(1, 1);
        float cx = undist_K_.at<float>(0, 2);
        float cy = undist_K_.at<float>(1, 2);
        
        img_pt(0) = fx * nx + cx;
        img_pt(1) = fy * ny + cy;
    } else {
        Eigen::Vector3d vp_normalized(nx, ny, 1.0);
        m_camera->spaceToPlane(vp_normalized, img_pt);
    }
    
    return true;
}

std::vector<KeyLine> LineFeatureTracker::getMatchedLines() const {
    std::vector<KeyLine> matched_lines;
    for (const auto& kl : cur_keylines) {
        if (kl.class_id != -1) {
            matched_lines.push_back(kl);
        }
    }
    return matched_lines;
}

std::vector<KeyLine> LineFeatureTracker::getVPAssignedLines() const {
    std::vector<KeyLine> vp_lines;
    for (const auto& kl : cur_keylines) {
        if (kl.octave != -1) {
            vp_lines.push_back(kl);
        }
    }
    return vp_lines;
}

LineFeatureTracker::LineFeatureStats LineFeatureTracker::getCurrentFrameStats() const {
    LineFeatureStats stats;
    stats.num_extracted_lines = cur_keylines.size();
    int matched_count = 0;
    if (cur_line_matched.size() == cur_keylines.size()) {
        for (size_t i = 0; i < cur_line_matched.size(); ++i) {
            if (cur_line_matched[i]) {
                matched_count++;
            }
        }
    } else {
        matched_count = 0;
    }
    stats.num_matched_lines = matched_count;
    stats.matching_rate = (stats.num_extracted_lines > 0) ? 
                         (double)matched_count / stats.num_extracted_lines : 0.0;
    std::vector<KeyLine> vp_lines = getVPAssignedLines();
    stats.num_vp_lines = vp_lines.size();
    stats.vp_inlier_ratio = (stats.num_extracted_lines > 0) ? 
                           (double)stats.num_vp_lines / stats.num_extracted_lines : 0.0;
    double total_length = 0.0;
    for (const auto& kl : vp_lines) {
        total_length += kl.lineLength;
    }
    stats.avg_vp_line_length = (stats.num_vp_lines > 0) ? 
                               total_length / stats.num_vp_lines : 0.0;
    if (stats.num_vp_lines > 0 && !cur_vps.empty()) {
        Eigen::Vector3d vp_dir = cur_vps.begin()->second;
        double total_angle_error = 0.0;
        int valid_angle_count = 0;
        
        for (const auto& kl : vp_lines) {
            Eigen::Vector3d sp, ep;
            m_camera->liftProjective(Eigen::Vector2d(kl.startPointX, kl.startPointY), sp);
            m_camera->liftProjective(Eigen::Vector2d(kl.endPointX, kl.endPointY), ep);
            Eigen::Vector3d line_normal = sp.cross(ep).normalized();
            double angle_error = std::abs(line_normal.dot(vp_dir));
            total_angle_error += angle_error;
            valid_angle_count++;
        }
        stats.avg_vp_angle_error = (valid_angle_count > 0) ? 
                                   total_angle_error / valid_angle_count : 0.0;
    }
    if (!cur_vps.empty() && cur_vp_found_this_frame) {
        Eigen::Vector2d vp_img_pt;
        if (projectVanishingPointToImage(cur_vps.begin()->second, vp_img_pt)) {
            stats.vp_valid = true;
            stats.vp_position = vp_img_pt;
            if (!prev_vps.empty()) {
                Eigen::Vector2d prev_vp_img_pt;
                if (projectVanishingPointToImage(prev_vps.begin()->second, prev_vp_img_pt)) {
                    stats.vp_position_change = (vp_img_pt - prev_vp_img_pt).norm();
                }
            }
        } else {
            stats.vp_valid = false;
        }
    } else {
        stats.vp_valid = false;
    }
    
    return stats;
}

void LineFeatureTracker::buildSphereCells() {
    if (sphere_initialized_) return;
    
    sphere_cells_.clear();
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;  
    const double norm = std::sqrt(1.0 + phi * phi);
    
    std::vector<Eigen::Vector3d> vertices = {
        {-1/norm,  phi/norm, 0}, { 1/norm,  phi/norm, 0},
        {-1/norm, -phi/norm, 0}, { 1/norm, -phi/norm, 0},
        {0, -1/norm,  phi/norm}, {0,  1/norm,  phi/norm},
        {0, -1/norm, -phi/norm}, {0,  1/norm, -phi/norm},
        { phi/norm, 0, -1/norm}, { phi/norm, 0,  1/norm},
        {-phi/norm, 0, -1/norm}, {-phi/norm, 0,  1/norm}
    };
    std::vector<std::array<int, 3>> faces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };
    auto midpoint = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return ((a + b) / 2.0).normalized();
    };
    for (int level = 0; level < SPHERE_SUBDIVISION_LEVEL; ++level) {
        std::vector<std::array<int, 3>> new_faces;
        std::map<std::pair<int,int>, int> edge_midpoints;
        
        auto getMidpoint = [&](int i, int j) -> int {
            auto key = (i < j) ? std::make_pair(i, j) : std::make_pair(j, i);
            if (edge_midpoints.count(key)) return edge_midpoints[key];
            int idx = vertices.size();
            vertices.push_back(midpoint(vertices[i], vertices[j]));
            edge_midpoints[key] = idx;
            return idx;
        };
        
        for (const auto& f : faces) {
            int a = f[0], b = f[1], c = f[2];
            int ab = getMidpoint(a, b);
            int bc = getMidpoint(b, c);
            int ca = getMidpoint(c, a);
            new_faces.push_back({a, ab, ca});
            new_faces.push_back({b, bc, ab});
            new_faces.push_back({c, ca, bc});
            new_faces.push_back({ab, bc, ca});
        }
        faces = std::move(new_faces);
    }
    for (const auto& f : faces) {
        Eigen::Vector3d center = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.0;
        center.normalize();
        if (center.z() > 0) {
            SphereCell cell;
            cell.center = center;
            cell.vote = 0.0;
            sphere_cells_.push_back(cell);
        }
    }
    
    sphere_initialized_ = true;
}

std::vector<LineFeatureTracker::LineNormalInfo> LineFeatureTracker::computeLineNormals(
    const std::vector<KeyLine>& keylines) {
    
    std::vector<LineNormalInfo> normals;
    normals.reserve(keylines.size());
    
    for (size_t i = 0; i < keylines.size(); ++i) {
        const auto& kl = keylines[i];
        Eigen::Vector3d sp, ep;
        
        if (!undist_K_.empty()) {
            float fx = undist_K_.at<float>(0, 0);
            float fy = undist_K_.at<float>(1, 1);
            float cx = undist_K_.at<float>(0, 2);
            float cy = undist_K_.at<float>(1, 2);
            
            sp = Eigen::Vector3d((kl.startPointX - cx) / fx, (kl.startPointY - cy) / fy, 1.0);
            ep = Eigen::Vector3d((kl.endPointX - cx) / fx, (kl.endPointY - cy) / fy, 1.0);
        } else {
            m_camera->liftProjective(Eigen::Vector2d(kl.startPointX, kl.startPointY), sp);
            m_camera->liftProjective(Eigen::Vector2d(kl.endPointX, kl.endPointY), ep);
        }
        Eigen::Vector3d normal = sp.cross(ep);
        double norm = normal.norm();
        if (norm < 1e-8) continue;
        normal /= norm;
        
        LineNormalInfo info;
        info.normal = normal;
        info.length = kl.lineLength;
        info.line_idx = static_cast<int>(i);
        normals.push_back(info);
    }
    
    return normals;
}

void LineFeatureTracker::accumulateVotes(const std::vector<LineNormalInfo>& line_normals, const std::vector<bool>& is_line_used) {
    for (auto& cell : sphere_cells_) {
        cell.vote = 0.0;
        cell.line_indices.clear();
    }
    for (size_t k = 0; k < line_normals.size(); ++k) {
        if (!is_line_used.empty() && is_line_used[k]) continue; 

        const auto& ln = line_normals[k];
        for (size_t i = 0; i < sphere_cells_.size(); ++i) {
            double dot = std::abs(ln.normal.dot(sphere_cells_[i].center));
            if (dot < VP_INLIER_THRESHOLD) {
                double w = ln.length * ln.length;
                sphere_cells_[i].vote += w;
                sphere_cells_[i].line_indices.push_back(ln.line_idx);
            }
        }
    }
}

std::vector<LineFeatureTracker::VPCandidate> LineFeatureTracker::detectVPPeaks(
    const std::vector<LineNormalInfo>& line_normals) {
    
    std::vector<VPCandidate> candidates;
    for (size_t i = 0; i < sphere_cells_.size(); ++i) {
        const auto& cell = sphere_cells_[i];
        if (cell.vote < 1e-6) continue;
        if (static_cast<int>(cell.line_indices.size()) < VP_MIN_INLIERS) continue;
        bool is_max = true;
        const double neighbor_threshold = 0.15;  
        
        for (size_t j = 0; j < sphere_cells_.size() && is_max; ++j) {
            if (i == j) continue;
            double dist = 1.0 - std::abs(cell.center.dot(sphere_cells_[j].center));
            if (dist < neighbor_threshold && sphere_cells_[j].vote > cell.vote) {
                is_max = false;
            }
        }
        
        if (is_max) {
            VPCandidate candidate;
            candidate.direction = cell.center;
            int inlier_count = static_cast<int>(cell.line_indices.size());
            candidate.score = cell.vote * std::log2(static_cast<double>(inlier_count) + 1.0);
            candidate.inlier_indices = cell.line_indices;
            candidates.push_back(candidate);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const VPCandidate& a, const VPCandidate& b) {
                  return a.score > b.score;
              });
    
    return candidates;
}

Eigen::Vector3d LineFeatureTracker::refineVPWithWLS(
    const std::vector<LineNormalInfo>& line_normals,
    const std::vector<int>& inlier_indices) {
    
    if (inlier_indices.empty()) {
        return Eigen::Vector3d(0, 0, 1);  
    }
    std::map<int, const LineNormalInfo*> idx_to_normal;
    for (const auto& ln : line_normals) {
        idx_to_normal[ln.line_idx] = &ln;
    }
    const int max_iterations = 3;
    const double strict_inlier_threshold = 0.02;  
    
    std::vector<int> current_inliers = inlier_indices;
    Eigen::Vector3d vp(0, 0, 1);
    
    for (int iter = 0; iter < max_iterations && !current_inliers.empty(); ++iter) {
        Eigen::Matrix3d WtW = Eigen::Matrix3d::Zero();
        double total_weight = 0.0;
        
        for (int idx : current_inliers) {
            auto it = idx_to_normal.find(idx);
            if (it == idx_to_normal.end()) continue;
            
            const LineNormalInfo* ln = it->second;
            double weight = ln->length;
            WtW += weight * ln->normal * ln->normal.transpose();
            total_weight += weight;
        }
        
        if (total_weight < 1e-8) {
            break;
        }
        
        WtW /= total_weight;  
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(WtW);
        if (solver.info() != Eigen::Success) {
            break;
        }
        vp = solver.eigenvectors().col(0);
        if (vp.z() < 0) vp = -vp;
        vp.normalize();
        std::vector<int> verified_inliers;
        for (int idx : current_inliers) {
            auto it = idx_to_normal.find(idx);
            if (it == idx_to_normal.end()) continue;
            
            double dot = std::abs(it->second->normal.dot(vp));
            if (dot < strict_inlier_threshold) {
                verified_inliers.push_back(idx);
            }
        }
        if (verified_inliers.size() < static_cast<size_t>(VP_MIN_INLIERS)) {
            break;  
        }
        
        current_inliers = verified_inliers;
    }
    
    return vp.normalized();
}

Eigen::Vector3d LineFeatureTracker::filterVPWithIMU(
    const Eigen::Vector3d& vp_measured,
    const Eigen::Matrix3d& R_imu) {
    
    if (prev_vps.empty()) {
        return vp_measured;
    }
    return vp_measured;
}

void LineFeatureTracker::updateVPTrackId(const Eigen::Vector3d& vp_current, const Eigen::Matrix3d& R_pred) {
    vp_switch_detected_ = false;

    auto prev_it = prev_vps.find(0);
    if (prev_it == prev_vps.end()) {
        cur_vp_track_age_ = 1;
        return;
    }

    const Eigen::Vector3d& vp_prev = prev_it->second;
    if (vp_prev.norm() < 1e-8 || vp_current.norm() < 1e-8) {
        cur_vp_track_age_ = 1;
        return;
    }
    Eigen::Vector3d vp_prev_pred = R_pred * vp_prev.normalized();
    const double pred_norm = vp_prev_pred.norm();
    if (pred_norm < 1e-8) {
        cur_vp_track_age_ = 1;
        return;
    }
    vp_prev_pred /= pred_norm;

    const Eigen::Vector3d vp_cur_n = vp_current.normalized();
    const double sim = std::abs(vp_prev_pred.dot(vp_cur_n));
    if (sim >= VP_TRACK_KEEP_THRESHOLD) {
        cur_vp_track_age_ = std::max(1, cur_vp_track_age_ + 1);
        return;
    }
    if (sim < VP_TRACK_SWITCH_THRESHOLD) {
        ++cur_vp_track_id_;
        vp_switch_detected_ = true;
        cur_vp_track_age_ = 1;
        return;
    }
    cur_vp_track_age_ = std::max(1, cur_vp_track_age_ + 1);
}

void LineFeatureTracker::calculateVanishingPointsRobust(
    std::vector<KeyLine>& keylines,
    const Eigen::Matrix3d& R_imu) {
    
    cur_vps.clear();
    cur_vp_found_this_frame = false;
    vp_switch_detected_ = false;
    last_vp_inlier_count_ = 0;
    last_vp_median_abs_dot_ = 1.0;
    last_vp_p90_abs_dot_ = 1.0;
    last_vp_cond_ratio_ = 0.0;
    last_vp_quality_valid_ = false;
    for (auto& kl : keylines) {
        kl.octave = -1;
    }
    
    if (keylines.size() < 3) {
        cur_vp_track_age_ = 0;
        return;
    }
    buildSphereCells();
    std::vector<LineNormalInfo> line_normals = computeLineNormals(keylines);
    std::vector<bool> is_line_used(keylines.size(), false);  
    accumulateVotes(line_normals, is_line_used);
    std::vector<VPCandidate> candidates = detectVPPeaks(line_normals);
    
    if (candidates.empty()) {
        cur_vp_track_age_ = 0;
        return;
    }
    debug_vp_candidates_.clear();
    for (size_t i = 0; i < std::min<size_t>(3, candidates.size()); ++i) {
        Eigen::Vector3d d = candidates[i].direction;
        d.normalize();
        debug_vp_candidates_.push_back(d);
    }
    Eigen::Vector3d vp_final = Eigen::Vector3d::Zero();
    bool found_valid_vp = false;
    
    int img_w = cur_img.cols > 0 ? cur_img.cols : COL;
    int img_h = cur_img.rows > 0 ? cur_img.rows : ROW;
    
    for (const auto& cand : candidates) {
        Eigen::Vector3d vp_dir = cand.direction;
        vp_dir.normalize();
        Eigen::Vector2d vp_img;
        bool in_front = projectVanishingPointToImage(vp_dir, vp_img);
        
        if (in_front && vp_img.x() >= 0 && vp_img.x() < img_w &&
            vp_img.y() >= 0 && vp_img.y() < img_h) {
            continue;
        }
        int inlier_cnt = 0;
        const double assignment_threshold = VP_INLIER_THRESHOLD;
        for (const auto& ln : line_normals) {
             if (std::abs(ln.normal.dot(vp_dir)) < assignment_threshold) {
                 inlier_cnt++;
             }
        }
        
        if (inlier_cnt < 10) {
            continue;
        }

        vp_final = vp_dir;
        found_valid_vp = true;
        break;
    }
    
    if (!found_valid_vp) {
        cur_vp_track_age_ = 0;
        return;
    }
    double cond_ratio_final = 0.0;
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        for (const auto& ln : line_normals) {
            double dot = std::abs(ln.normal.dot(vp_final));
            if (dot < VP_INLIER_THRESHOLD) {
                double w = ln.length * ln.length;  
                A += w * ln.normal * ln.normal.transpose();
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(A);
        if (solver.info() == Eigen::Success) {
            auto evals = solver.eigenvalues();
            double cond_ratio = (evals(2) > 1e-10) ? evals(1) / evals(2) : 0.0;
            cond_ratio_final = cond_ratio;
            
            if (cond_ratio > 0.01) {  
                Eigen::Vector3d refined = solver.eigenvectors().col(0);
                refined.normalize();
                if (refined.dot(vp_final) < 0) refined = -refined;
                double similarity = std::abs(refined.dot(vp_final));
                if (similarity > 0.7) {  
                    vp_final = refined;
                }
            }
        }
    }
    updateVPTrackId(vp_final, R_imu);
    cur_vps[0] = vp_final;
    cur_vp_found_this_frame = true;
    int final_inlier_count = 0;
    std::vector<double> inlier_abs_dots;
    inlier_abs_dots.reserve(line_normals.size());
    for (const auto& ln : line_normals) {
         double dot = std::abs(ln.normal.dot(vp_final));
         if (dot < VP_INLIER_THRESHOLD) {
             keylines[ln.line_idx].octave = 0;
             ++final_inlier_count;
             inlier_abs_dots.push_back(dot);
         }
    }

    last_vp_inlier_count_ = final_inlier_count;
    last_vp_cond_ratio_ = cond_ratio_final;
    if (!inlier_abs_dots.empty()) {
        std::sort(inlier_abs_dots.begin(), inlier_abs_dots.end());
        const size_t n = inlier_abs_dots.size();
        last_vp_median_abs_dot_ = inlier_abs_dots[n / 2];
        const size_t p90_idx = std::min(n - 1, static_cast<size_t>(std::floor(0.9 * static_cast<double>(n - 1))));
        last_vp_p90_abs_dot_ = inlier_abs_dots[p90_idx];
    }
    last_vp_quality_valid_ = (final_inlier_count >= VP_MIN_INLIERS);
}

std::vector<cv::Point2f> LineFeatureTracker::undistortPoints(const std::vector<cv::Point2f>& pts) const {
    if (!m_camera) return pts;
    std::vector<cv::Point2f> un_pts;
    un_pts.reserve(pts.size());
    double fx = FOCAL_LENGTH; 
    double fy = FOCAL_LENGTH;
    double cx = COL / 2.0;
    double cy = ROW / 2.0;
    
    for (const auto& p : pts) {
        Eigen::Vector3d ray;
        m_camera->liftProjective(Eigen::Vector2d(p.x, p.y), ray);
        
        if (ray.z() <= 0) {
            un_pts.push_back(p); 
            continue;
        }
        
        double u = fx * ray.x() / ray.z() + cx;
        double v = fy * ray.y() / ray.z() + cy;
        un_pts.emplace_back(u, v);
    }
    return un_pts;
}

cv::Mat LineFeatureTracker::getUndistortedImage(const cv::Mat& raw_img) {
    if (!m_camera || raw_img.empty()) return raw_img.clone();
    
    cv::Mat gray;
    if (raw_img.channels() == 3) {
        cv::cvtColor(raw_img, gray, cv::COLOR_BGR2GRAY);
    } else if (raw_img.channels() == 4) {
        cv::cvtColor(raw_img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = raw_img.clone();
    }
    
    ensureUndistortMaps(gray.size());
    
    if (undist_map1_.empty() || undist_map2_.empty()) return gray;
    
    cv::Mat undist_img;
    cv::remap(gray, undist_img, undist_map1_, undist_map2_, cv::INTER_LINEAR);
    return undist_img;
}

} 
