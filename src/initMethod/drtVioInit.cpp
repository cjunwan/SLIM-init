#include "initMethod/drtVioInit.h"
#include <iostream>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unsupported/Eigen/Polynomials>
#include "featureTracker/parameters.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>


namespace DRT {

    using namespace vio;

    
    static bool findPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs,
                                                   Eigen::VectorXd* real,
                                                   Eigen::VectorXd* imag) {
        const int degree = static_cast<int>(coeffs.size()) - 1;
        if (degree < 1) {
            return false;
        }
        const double leading = coeffs[0];
        if (std::abs(leading) < 1e-12) {
            return false;
        }
        Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(degree, degree);
        for (int i = 0; i < degree; ++i) {
            companion(0, i) = -coeffs[i + 1] / leading;
        }
        for (int i = 1; i < degree; ++i) {
            companion(i, i - 1) = 1.0;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> es(companion,  false);
        if (es.info() != Eigen::Success) {
            return false;
        }

        const auto& evals = es.eigenvalues();
        real->resize(degree);
        imag->resize(degree);
        for (int i = 0; i < degree; ++i) {
            (*real)(i) = evals(i).real();
            (*imag)(i) = evals(i).imag();
        }
        return true;
    }
    static Eigen::VectorXd RealRoots(const Eigen::VectorXd& real,
                                     const Eigen::VectorXd& imag,
                                     double tol = 1e-8) {
        std::vector<double> roots;
        roots.reserve(real.size());
        for (int i = 0; i < real.size(); ++i) {
            if (std::abs(imag(i)) < tol) {
                roots.push_back(real(i));
            }
        }
        Eigen::VectorXd out(roots.size());
        for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
            out(i) = roots[i];
        }
        return out;
    }

    
    struct VPConsistencyCostFunctor {
        VPConsistencyCostFunctor(
            const Eigen::Vector3d& vp_i,   
            const Eigen::Vector3d& vp_j,   
            const vio::IMUPreintegrated& imu_integ,
            const Eigen::Matrix3d& Rbc,
            double weight = 1.0)
            : vp_i_(vp_i.normalized()), vp_j_(vp_j.normalized()),
              imu_data_(imu_integ), Rbc_(Rbc), weight_(weight) {}

        template <typename T>
        bool operator()(const T* const bias_g, T* residual) const {
            Eigen::Matrix<T, 3, 1> bg_vec;
            bg_vec << bias_g[0], bias_g[1], bias_g[2];
            Eigen::Quaterniond q_meas_d = imu_data_.dR_.unit_quaternion();
            Eigen::Quaternion<T> q_meas = q_meas_d.template cast<T>();

            Eigen::Matrix<T, 3, 3> J_rbg = imu_data_.JRg_.template cast<T>();
            Eigen::Matrix<T, 3, 1> delta_bg = imu_data_.bias_.bg_.template cast<T>() - bg_vec;
            Eigen::Matrix<T, 3, 1> theta = J_rbg * delta_bg;
            Eigen::Quaternion<T> dq(T(1.0),
                                    theta(0) * T(0.5),
                                    theta(1) * T(0.5),
                                    theta(2) * T(0.5));
            dq.normalize();
            Eigen::Quaternion<T> q_bj_bi = q_meas * dq;
            Eigen::Quaternion<T> q_bc(Rbc_.template cast<T>());
            Eigen::Quaternion<T> q_cj_ci = q_bc.inverse() * q_bj_bi * q_bc;
            Eigen::Matrix<T, 3, 1> vp_i_cast = vp_i_.template cast<T>();
            Eigen::Matrix<T, 3, 1> vp_predicted = q_cj_ci * vp_i_cast;
            Eigen::Matrix<T, 3, 1> vp_j_cast = vp_j_.template cast<T>();
            Eigen::Matrix<T, 3, 1> cross_prod = vp_j_cast.cross(vp_predicted);
            T cross_norm = cross_prod.norm();
            residual[0] = T(weight_) * cross_norm;

            return true;
        }

        static ceres::CostFunction* Create(
            const Eigen::Vector3d& vp_i, const Eigen::Vector3d& vp_j,
            const vio::IMUPreintegrated& imu, const Eigen::Matrix3d& Rbc,
            double weight = 1.0) {
            return new ceres::AutoDiffCostFunction<VPConsistencyCostFunctor, 1, 3>(
                new VPConsistencyCostFunctor(vp_i, vp_j, imu, Rbc, weight));
        }

        Eigen::Vector3d vp_i_, vp_j_;
        vio::IMUPreintegrated imu_data_;
        Eigen::Matrix3d Rbc_;
        double weight_;
    };


    drtVioInit::drtVioInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc): Rbc_(Rbc),pbc_(pbc)
    {
        biasa.setZero();
        biasg.setZero();
        time_frameid2_int_frameid.clear();
        int_frameid2_time_frameid.clear();
    }

    void drtVioInit::recomputeFrameId() {
        int_frameid2_time_frameid.clear();
        time_frameid2_int_frameid.clear();

        int local_frame_id = 0;
        for (auto it = local_active_frames.begin(); it != local_active_frames.end(); it++) {
            int_frameid2_time_frameid.insert(make_pair(local_frame_id, *it));
            time_frameid2_int_frameid.insert(make_pair(*it, local_frame_id));
            local_frame_id++;
        }
    }

    bool drtVioInit::checkAccError() {

        bool check_success = false;

        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            int j = i + 1;
            CHECK((int_frameid2_time_frameid.at(j) - int_frameid2_time_frameid.at(i)) == imu_meas[i].sum_dt_) <<
            int_frameid2_time_frameid.at(j) << " " << int_frameid2_time_frameid.at(i) << imu_meas[i].sum_dt_;
        }

        Eigen::Vector3d avgA;
        avgA.setZero();

        std::vector<int> is_bad(imu_meas.size(), 0);
        for (int i = 0; i < imu_meas.size(); i++) {
            Eigen::Vector3d acc = imu_meas[i].dV_ / imu_meas[i].sum_dt_;
            if (std::abs(acc.norm() - G.norm()) / G.norm() < 5e-3)
                is_bad[i] = 1;
            avgA += imu_meas[i].dV_ / imu_meas[i].sum_dt_;
        }

        int scoreSum = std::accumulate(is_bad.begin(), is_bad.end(), 0.0);


        avgA /= static_cast<double>(imu_meas.size());
        const double avgA_error = std::abs(avgA.norm() - G.norm()) / G.norm();

        if (avgA_error > 5e-3 and scoreSum <= ACC_CHECK_MAX_NEAR_G_SEGMENTS)
            check_success = true;
        last_window_acc_norm = avgA.norm();
        last_window_acc_error = avgA_error;
        last_window_bad_count = scoreSum;

        return check_success;
    }

    void drtVioInit::addImuMeasure(const vio::IMUPreintegrated &imuData) {
        imu_meas.push_back(imuData);
    }

    bool drtVioInit::addFeatureCheckParallax(TimeFrameId frame_id, const FeatureTrackerResulst &image,
                                             double td) {

        bool insert_image_frame = false;
        if (local_active_frames.size() == 0)
        {
            insert_image_frame = true;
        } else {

            double parallax_sum = 0;
            int parallax_num = 0;
            for (const auto &pts: image) {
                if (SFMConstruct.find(pts.first) != SFMConstruct.end()) {

                    Eigen::Vector3d cur_pt{pts.second[0].second(0),
                                           pts.second[0].second(1),
                                           pts.second[0].second(2)};

                    Eigen::Vector3d last_pt = SFMConstruct.at(pts.first).obs.at(last_image_t_ns).normalpoint;

                    parallax_sum += compensatedParallax2(cur_pt, last_pt);
                    ++parallax_num;
                }
            }

            if (std::abs(frame_id - last_image_t_ns) >= 0.22) {
                insert_image_frame = true;
            } else
            {
                insert_image_frame = false;
            }
        }

        if (insert_image_frame) {
            local_active_frames.insert(frame_id);

            recomputeFrameId();

            for (auto &id_pts: image) {
                FeaturePerFrame kpt_obs(id_pts.second[0].second, td);
                FeatureID feature_id = id_pts.first;

                if (SFMConstruct.find(feature_id) == SFMConstruct.end()) {
                    SFMConstruct[feature_id] = SFMFeature(feature_id, frame_id);
                    CHECK(frame_id != 0) << "frame_id == 0";
                    SFMConstruct[feature_id].obs[frame_id] = kpt_obs;
                } else {
                    SFMConstruct[feature_id].obs[frame_id] = kpt_obs;
                }
            }
            if (line_tracker) {
                const auto& vps = line_tracker->getVanishingPoints();
                if (!vps.empty() && vps.find(0) != vps.end()) {
                    VPInfo info;
                    info.direction = vps.at(0).normalized();
                    info.track_id = line_tracker->getVPTrackId();
                    info.track_age = line_tracker->getVPTrackAge();
                    info.inlier_count = line_tracker->getVPInlierCount();
                    info.median_abs_dot = line_tracker->getVPMedianAbsDot();
                    info.p90_abs_dot = line_tracker->getVPP90AbsDot();
                    info.cond_ratio = line_tracker->getVPCondRatio();
                    info.quality_valid = line_tracker->isVPQualityValid();
                    vp_history[frame_id] = info;
                }
            }

            last_image_t_ns = frame_id;
            return true;
        } else {
            return false;
        }
    }

    double drtVioInit::compensatedParallax2(const Eigen::Vector3d &p_i, const Eigen::Vector3d &p_j) {
        double ans = 0;
        double u_j = p_j(0);
        double v_j = p_j(1);

        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;
        double du = u_i - u_j, dv = v_i - v_j;

        ans = std::sqrt(du * du + dv * dv);

        return ans;
    }

	    bool drtVioInit::gyroBiasEstimator() {
	
	        ticToc t_optimize;
	        ceres::Problem problem;
	        ceres::LossFunction *point_loss_function = new ceres::CauchyLoss(1e-5);
        constexpr double kVpLossScale = 5e-6;
        ceres::LossFunction *vp_loss_function = new ceres::CauchyLoss(kVpLossScale);

	        int num_obs = 0;
	        int num_point_residual_blocks = 0;
        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            auto target1_tid = int_frameid2_time_frameid.at(i);
            auto target2_tid = int_frameid2_time_frameid.at(i + 1);

            std::vector<Eigen::Vector3d> fis;
            std::vector<Eigen::Vector3d> fjs;

            for (const auto &pts: SFMConstruct) {
                if (pts.second.obs.find(target1_tid) != pts.second.obs.end() &&
                    pts.second.obs.find(target2_tid) != pts.second.obs.end()) {
                    ++num_obs;
                    fis.push_back(pts.second.obs.at(target1_tid).normalpoint);
                    fjs.push_back(pts.second.obs.at(target2_tid).normalpoint);
                }
            }


	            const vio::IMUPreintegrated& imu1 = imu_meas[i];
	            ceres::CostFunction *point_cost_function = BiasSolverCostFunctor::Create(fis, fjs,
	                                                                                           Eigen::Quaterniond(Rbc_),
	                                                                                           imu1);
	            problem.AddResidualBlock(point_cost_function, point_loss_function, biasg.data());
	            ++num_point_residual_blocks;
	        }
        last_window_lines_matched = 0;
        last_window_lines_used = 0;
        last_window_lines_matched_normal = 0;
        last_window_lines_used_normal = 0;
        last_window_lines_matched_epi = 0;
        last_window_lines_used_epi = 0;
        selected_global_line_ids.clear();
        selected_normal_line_ids.clear();
        selected_epipolar_line_ids.clear();
        selected_normal_line_scores.clear();
        selected_epipolar_line_scores.clear();
        last_window_line_tracks_str = "";
        last_window_line_tracks_normal_str = "";
        last_window_line_tracks_epi_str = "";

        bool do_line_selection = !SFMLineConstruct.empty(); 
        if (do_line_selection) {
            const int window_frame_count = static_cast<int>(int_frameid2_time_frameid.size());
            const int window_pair_count = std::max(1, window_frame_count - 1);

            const int kMinTrackCountNormal = 2;
            const int kMinVisiblePairsNormal = std::max(2, LINE_MIN_USED_FOR_ALIGNMENT);
            const double kMinTrackCoverageNormal = 0.25;
            const double kMinPairCoverageNormal = 0.40;
            const double kMinObsQualityNormal = 0.45;
            const double kMinNormalScore = 0.65;
            const int kMinTrackCountEpipolar = EPIPOLAR_MIN_TRACK_COUNT;
            const int kMinVisiblePairsEpipolar = std::max(2, LINE_MIN_USED_FOR_ALIGNMENT);
            const double kMinPairCoverageEpipolar = EPIPOLAR_MIN_PAIR_COVERAGE;
            const double kMinMatchQualityEpipolar = EPIPOLAR_MIN_MATCH_QUALITY;
            const double kMinObsQualityEpipolar = EPIPOLAR_MIN_OBS_QUALITY;
            const double kMinEpiScore = EPIPOLAR_MIN_SCORE;
            const int kGlobalMaxLinesNormal = 10;
            const int kGlobalMaxLinesEpipolar = std::max(0, EPIPOLAR_MAX_LINES_GLOBAL);

            struct LineCandidate {
                const SFMLine* ptr;
                int local_count;
                double score;
            };
            std::vector<LineCandidate> normal_candidates;
            std::vector<LineCandidate> epi_candidates;

            auto clamp01 = [](double x) -> double {
                return std::clamp(x, 0.0, 1.0);
            };

            for (const auto& line_pair : SFMLineConstruct) {
                const auto& sfm_line = line_pair.second;
                if (sfm_line.obs.empty()) continue;

                int local_track_count = 0;
                int visible_pair_count = 0;
                double sum_obs_quality = 0.0;
                double sum_match_quality = 0.0;
                double sum_length_norm = 0.0;

                for (auto const& [frame_idx, time_id] : int_frameid2_time_frameid) {
                    auto obs_it = sfm_line.obs.find(time_id);
                    if (obs_it == sfm_line.obs.end()) continue;
                    ++local_track_count;

                    const auto& obs = obs_it->second;
                    const double len_ref = std::max(1.0, LINE_MIN_LENGTH * 3.0);
                    const double len_score = clamp01(obs.length_px / len_ref);
                    double obs_q = obs.obs_quality;
                    if (obs_q <= 0.0) {
                        const double det_score = clamp01(1.0 - std::exp(-std::max(0.0, obs.detector_score) / 20.0));
                        obs_q = 0.9 * len_score + 0.1 * det_score;
                    }
                    sum_obs_quality += clamp01(obs_q);
                    sum_match_quality += clamp01(obs.match_score);
                    sum_length_norm += len_score;
                }

                if (local_track_count == 0) continue;

                for (int i = 0; i < window_frame_count - 1; ++i) {
                    const auto t1 = int_frameid2_time_frameid.at(i);
                    const auto t2 = int_frameid2_time_frameid.at(i + 1);
                    if (sfm_line.obs.count(t1) && sfm_line.obs.count(t2)) {
                        ++visible_pair_count;
                    }
                }
                if (visible_pair_count == 0) continue;

                const double mean_obs_quality = sum_obs_quality / static_cast<double>(local_track_count);
                const double mean_match_quality = sum_match_quality / static_cast<double>(local_track_count);
                const double mean_length_norm = sum_length_norm / static_cast<double>(local_track_count);
                const double track_coverage = static_cast<double>(local_track_count) / static_cast<double>(window_frame_count);
                const double pair_coverage = static_cast<double>(visible_pair_count) / static_cast<double>(window_pair_count);

                const double normal_score =
                    clamp01(0.65 * mean_obs_quality + 0.15 * track_coverage + 0.20 * mean_length_norm);
                const double epi_score = clamp01(0.45 * track_coverage + 0.35 * pair_coverage + 0.20 * mean_match_quality);

                if (local_track_count >= kMinTrackCountNormal &&
                    visible_pair_count >= kMinVisiblePairsNormal &&
                    track_coverage >= kMinTrackCoverageNormal &&
                    pair_coverage >= kMinPairCoverageNormal &&
                    mean_obs_quality >= kMinObsQualityNormal &&
                    normal_score >= kMinNormalScore) {
                    normal_candidates.push_back({&sfm_line, local_track_count, normal_score});
                }
                if (local_track_count >= kMinTrackCountEpipolar &&
                    visible_pair_count >= kMinVisiblePairsEpipolar &&
                    pair_coverage >= kMinPairCoverageEpipolar &&
                    mean_match_quality >= kMinMatchQualityEpipolar &&
                    mean_obs_quality >= kMinObsQualityEpipolar &&
                    epi_score >= kMinEpiScore) {
                    epi_candidates.push_back({&sfm_line, local_track_count, epi_score});
                }
            }

            last_window_lines_matched_normal = static_cast<int>(normal_candidates.size());
            last_window_lines_matched_epi = static_cast<int>(epi_candidates.size());

            auto sort_by_score = [](const LineCandidate& a, const LineCandidate& b) {
                if (a.score == b.score) return a.local_count > b.local_count;
                return a.score > b.score;
            };
            std::sort(normal_candidates.begin(), normal_candidates.end(), sort_by_score);
            std::sort(epi_candidates.begin(), epi_candidates.end(), sort_by_score);

            auto build_selected_set = [](const std::vector<LineCandidate>& candidates,
                                         int max_count,
                                         std::vector<int>& selected_ids,
                                         std::map<FeatureID, double>& selected_scores,
                                         std::string& track_log) {
                selected_ids.clear();
                selected_scores.clear();
                std::stringstream ss;
                const int n_select = std::min(static_cast<int>(candidates.size()), max_count);
                for (int k = 0; k < n_select; ++k) {
                    const auto* line_ptr = candidates[k].ptr;
                    selected_ids.push_back(line_ptr->line_id);
                    selected_scores[line_ptr->line_id] = candidates[k].score;
                    if (k > 0) ss << "|";
                    ss << candidates[k].local_count << ":" << std::fixed << std::setprecision(2) << candidates[k].score;
                }
                if (n_select == 0) ss << "None";
                track_log = ss.str();
            };

            build_selected_set(
                normal_candidates, kGlobalMaxLinesNormal,
                selected_normal_line_ids, selected_normal_line_scores, last_window_line_tracks_normal_str);
            build_selected_set(
                epi_candidates, kGlobalMaxLinesEpipolar,
                selected_epipolar_line_ids, selected_epipolar_line_scores, last_window_line_tracks_epi_str);

            auto count_used_pairs = [&](const std::vector<int>& selected_ids) -> int {
                int used = 0;
                for (int i = 0; i < window_frame_count - 1; ++i) {
                    const auto t1 = int_frameid2_time_frameid.at(i);
                    const auto t2 = int_frameid2_time_frameid.at(i + 1);
                    for (int line_id : selected_ids) {
                        auto line_it = SFMLineConstruct.find(line_id);
                        if (line_it == SFMLineConstruct.end()) continue;
                        const auto& sfm_line = line_it->second;
                        if (sfm_line.obs.count(t1) && sfm_line.obs.count(t2)) {
                            ++used;
                        }
                    }
                }
                return used;
            };

            last_window_lines_used_normal = count_used_pairs(selected_normal_line_ids);
            last_window_lines_used_epi = count_used_pairs(selected_epipolar_line_ids);
            selected_global_line_ids = selected_normal_line_ids;
            last_window_lines_matched = last_window_lines_matched_normal;
            last_window_lines_used = last_window_lines_used_normal;
            last_window_line_tracks_str =
                "N[" + last_window_line_tracks_normal_str + "] E[" + last_window_line_tracks_epi_str + "]";
        }
        if (line_tracker && !vp_history.empty() && VP_USE_RESIDUAL > 0.0) {
            const double vp_residual_weight_base = LINE_RESIDUAL_WEIGHT;
            constexpr double kPointToVpResidualRatio = 10.0;
            const double vp_gate_min_similarity = 0.99;
            const int vp_min_inlier_count = 20;
            const double vp_max_median_abs_dot = 0.01;
            const double vp_max_p90_abs_dot = 0.02;
            const double vp_min_cond_ratio = 0.05;
            const double vp_min_rotation_deg = 2.0;
            const int vp_min_track_age = 3;
            const int vp_min_pairs_per_window = 3;

            struct VPCandidate {
                Eigen::Vector3d vp_i;
                Eigen::Vector3d vp_j;
                const vio::IMUPreintegrated* imu_pair;
            };
            std::vector<VPCandidate> vp_candidates;
            vp_candidates.reserve(int_frameid2_time_frameid.size());

            auto is_vp_reliable = [&](const VPInfo& info) -> bool {
                if (!info.quality_valid) return false;
                if (info.track_age < vp_min_track_age) return false;
                if (info.inlier_count < vp_min_inlier_count) return false;
                if (info.median_abs_dot > vp_max_median_abs_dot) return false;
                if (info.p90_abs_dot > vp_max_p90_abs_dot) return false;
                if (info.cond_ratio < vp_min_cond_ratio) return false;
                return true;
            };
            for (int i = 0; i < static_cast<int>(int_frameid2_time_frameid.size()) - 1; i++) {
                if (i >= static_cast<int>(imu_meas.size())) continue;
                const vio::IMUPreintegrated& imu_pair = imu_meas[i];
                TimeFrameId tid_i = int_frameid2_time_frameid.at(i);
                TimeFrameId tid_j = int_frameid2_time_frameid.at(i + 1);
                auto vp_i_it = vp_history.find(tid_i);
                auto vp_j_it = vp_history.find(tid_j);
                
                if (vp_i_it != vp_history.end() && vp_j_it != vp_history.end()) {
                    const auto& vp_info_i = vp_i_it->second;
                    const auto& vp_info_j = vp_j_it->second;
                    if (vp_info_i.track_id == vp_info_j.track_id) {
                        const Eigen::Vector3d vp_i = vp_info_i.direction.normalized();
                        const Eigen::Vector3d vp_j = vp_info_j.direction.normalized();

                        if (!is_vp_reliable(vp_info_i) || !is_vp_reliable(vp_info_j)) {
                            continue;
                        }

                        const double rot_angle_deg =
                                std::abs(Eigen::AngleAxisd(imu_pair.dR_.matrix()).angle()) * 180.0 / M_PI;
                        if (rot_angle_deg < vp_min_rotation_deg) {
                            continue;
                        }
                        Eigen::Quaterniond q_meas = imu_pair.dR_.unit_quaternion();
                        Eigen::Quaterniond q_bc(Rbc_);
                        Eigen::Quaterniond q_cj_ci = q_bc.inverse() * q_meas * q_bc;
                        Eigen::Vector3d vp_pred = q_cj_ci * vp_i;
                        const double pred_norm = vp_pred.norm();
                        if (pred_norm <= 1e-8) {
                            continue;
                        }
                        vp_pred /= pred_norm;
                        const double sim = std::abs(vp_pred.dot(vp_j));
                        if (sim < vp_gate_min_similarity) {
                            continue;
                        }

                        vp_candidates.push_back({vp_i, vp_j, &imu_pair});
                    }
                }
            }
            if (static_cast<int>(vp_candidates.size()) >= vp_min_pairs_per_window) {
                const int num_vp_residual_blocks = static_cast<int>(vp_candidates.size());
                const int num_point_blocks = std::max(1, num_point_residual_blocks);
                const double vp_ratio_scale = std::sqrt(
                        static_cast<double>(num_point_blocks) /
                        (kPointToVpResidualRatio * static_cast<double>(num_vp_residual_blocks)));
                const double vp_residual_weight = vp_residual_weight_base * vp_ratio_scale;
                for (const auto& c : vp_candidates) {
                    ceres::CostFunction* vp_cost = VPConsistencyCostFunctor::Create(
                            c.vp_i, c.vp_j, *(c.imu_pair), Rbc_, vp_residual_weight);
                    problem.AddResidualBlock(vp_cost, vp_loss_function, biasg.data());
                }
            }
        }

        if(num_obs / local_active_frames.size() < 30) {
            std::cout << "invalid number observation: " << num_obs / local_active_frames.size() << std::endl;
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 200;
        options.gradient_tolerance = 1e-20;
        options.function_tolerance = 1e-20;
        options.parameter_tolerance = 1e-20;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        
        ceres::Solver::Summary summary;
        try {
            ceres::Solve(options, &problem, &summary);
        } catch(...) {
            return false;
        }

        if (summary.termination_type == ceres::TerminationType::FAILURE || 
            summary.termination_type == ceres::TerminationType::USER_FAILURE) {
            std::cout << "[Step S1] Gyro Bias Estimation Failed: " << summary.message << std::endl;
            return false; 
        }

        return true;
    }


    bool drtVioInit::gravityRefine(const Eigen::MatrixXd &M,
                                   const Eigen::VectorXd &m,
                                   double Q,
                                   double gravity_mag,
                                   Eigen::VectorXd &rhs) {
        int q = M.rows() - 3;

        Eigen::MatrixXd A = 2. * M.block(0, 0, q, q);

        Eigen::MatrixXd Bt = 2. * M.block(q, 0, 3, q);
        Eigen::MatrixXd BtAi = Bt * A.inverse();

        Eigen::Matrix3d D = 2. * M.block(q, q, 3, 3);
        Eigen::Matrix3d S = D - BtAi * Bt.transpose();

        Eigen::Matrix3d Sa = S.determinant() * S.inverse();
        Eigen::Matrix3d U = S.trace() * Eigen::Matrix3d::Identity() - S;

        Eigen::Vector3d v1 = BtAi * m.head(q);
        Eigen::Vector3d m2 = m.tail<3>();

        Eigen::Matrix3d X;
        Eigen::Vector3d Xm2;
        const double c4 = 16. * (v1.dot(v1) - 2. * v1.dot(m2) + m2.dot(m2));

        X = U;
        Xm2 = X * m2;
        const double c3 = 16. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = 2. * Sa + U * U;
        Xm2 = X * m2;
        const double c2 = 4. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = Sa * U + U * Sa;
        Xm2 = X * m2;
        const double c1 = 2. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        X = Sa * Sa;
        Xm2 = X * m2;
        const double c0 = (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

        const double s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
        const double s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

        const double t1 = s00 + s11 + s22;
        const double t2 = s00 * s11 + s00 * s22 + s11 * s22
                          - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
        const double t3 = s00 * s11 * s22 + 2. * s01 * s02 * s12
                          - s00 * std::pow(s12, 2) - s11 * std::pow(s02, 2) - s22 * std::pow(s01, 2);

        Eigen::VectorXd coeffs(7);
        coeffs << 64.,
                64. * t1,
                16. * (std::pow(t1, 2) + 2. * t2),
                16. * (t1 * t2 + t3),
                4. * (std::pow(t2, 2) + 2. * t1 * t3),
                4. * t3 * t2,
                std::pow(t3, 2);

        const double G2i = 1. / std::pow(gravity_mag, 2);

        coeffs(2) -= c4 * G2i;
        coeffs(3) -= c3 * G2i;
        coeffs(4) -= c2 * G2i;
        coeffs(5) -= c1 * G2i;
        coeffs(6) -= c0 * G2i;

        Eigen::VectorXd real, imag;
        if (!findPolynomialRootsCompanionMatrix(coeffs, &real, &imag)) {
            LOG(ERROR) << "Failed to find roots\n";
            printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                   coeffs[4], coeffs[5], coeffs[6]);

            return false;
        }

        Eigen::VectorXd lambdas = RealRoots(real, imag);
        if (lambdas.size() == 0) {
            LOG(ERROR) << "No real roots found\n";
            printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3],
                   coeffs[4], coeffs[5], coeffs[6]);

            return false;
        }

        Eigen::MatrixXd W(M.rows(), M.rows());
        W.setZero();
        W.block<3, 3>(q, q) = Eigen::Matrix3d::Identity();

        Eigen::VectorXd solution;
        double min_cost = std::numeric_limits<double>::max();
        for (Eigen::VectorXd::Index i = 0; i < lambdas.size(); ++i) {
            const double lambda = lambdas(i);

            Eigen::FullPivLU<Eigen::MatrixXd> lu(2. * M + 2. * lambda * W);
            Eigen::VectorXd x_ = -lu.inverse() * m;

            double cost = x_.transpose() * M * x_;
            cost += m.transpose() * x_;
            cost += Q;

            if (cost < min_cost) {
                solution = x_;
                min_cost = cost;
            }
        }


        const double constraint = solution.transpose() * W * solution;

        if (constraint < 0.
            || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3) { 
            LOG(WARNING) << "Discarding bad solution...\n";
            printf("constraint: %.16f\n", constraint);
            printf("constraint error: %.2f\n",
                   100. * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
            return false;
        }

        rhs = solution;

        return true;

    }

}
