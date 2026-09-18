#include "initMethod/slimInit.h"
#include <algorithm>

namespace DRT
{
    slimInit::slimInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc)
    : drtVioInit(Rbc, pbc){}

    bool slimInit::process()
    {

        cout << "slim init process" << endl;

        ticToc t_solve;
        ticToc t_biasg;

        if(!gyroBiasEstimator())
            return false;


        double time_biasg = t_biasg.toc();

        ticToc t_ligt;
        vio::IMUBias solved_bias(biasg, biasa);

        for (int i = 0; i < imu_meas.size(); i++) {
            imu_meas[i].reintegrate(solved_bias);
        }
        frame_rot[int_frameid2_time_frameid.at(0)] = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d accumRot = Eigen::Matrix3d::Identity();

        for (int i = 1; i < int_frameid2_time_frameid.size(); i++) {
            Eigen::Matrix3d dRcicj = Rbc_.transpose() * imu_meas[i - 1].dR_.matrix() * Rbc_;
            accumRot = accumRot * dRcicj;
            frame_rot[int_frameid2_time_frameid.at(i)] = accumRot;
        }
        int num_view_ = int_frameid2_time_frameid.size();
        int num_pts_ = 0;

        for (const auto &pt: SFMConstruct) {
            if (pt.second.obs.size() < 3) continue;
            ++num_pts_;
        }
        Eigen::MatrixXd LTL = Eigen::MatrixXd::Zero(num_view_ * 3 - 3, num_view_ * 3 - 3);
        Eigen::MatrixXd A_lr = Eigen::MatrixXd::Zero(num_pts_, 3 * num_view_);
        build_LTL(LTL, A_lr);

        Eigen::VectorXd evectors = Eigen::VectorXd::Zero(3 * num_view_);
        if (!solve_LTL(LTL, evectors)) {
            return false;
        }
        identify_sign(A_lr, evectors);

        rotation.resize(int_frameid2_time_frameid.size());
        position.resize(int_frameid2_time_frameid.size());
        velocity.resize(int_frameid2_time_frameid.size());


        for (int i = 0; i < local_active_frames.size(); i++) {
            rotation[i] = frame_rot.at(int_frameid2_time_frameid.at(i)) * Rbc_.transpose();
            position[i] = evectors.middleRows<3>(3 * i);
        }
        double time_ligt = t_ligt.toc();

        ticToc t_velocity_gravity;
        if (linearAlignment()) {
            return true;
        } else {
            printf("solve g failed!\n");
            return false;
        }
    }

    bool slimInit::linearAlignment() {
        int all_frame_count = int_frameid2_time_frameid.size();
        int n_state = all_frame_count * 3 + 3 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();
        double Q = 0.;

        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            int j = i + 1;
            MatrixXd tmp_A(6, 10);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = imu_meas[i].sum_dt_;

            CHECK(imu_meas[i].start_t_ns == int_frameid2_time_frameid.at(i)) << "imu meas error";
            CHECK(imu_meas[i].end_t_ns == int_frameid2_time_frameid.at(j)) << "imu meas error";

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 1>(0, 6) = rotation[i].transpose() * (position[j] - position[i]) / 100.0;

            tmp_A.block<3, 3>(0, 7) = rotation[i].transpose() * dt * dt / 2 * Matrix3d::Identity() * G.norm();

            tmp_b.block<3, 1>(0, 0) = imu_meas[i].dP_ + rotation[i].transpose() * rotation[j] * pbc_ - pbc_;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = rotation[i].transpose() * rotation[j];

            tmp_A.block<3, 3>(3, 7) = rotation[i].transpose() * dt * Matrix3d::Identity() * G.norm();

            tmp_b.block<3, 1>(3, 0) = imu_meas[i].dV_;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();

            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;
            if (LINE_RESIDUAL_WEIGHT > 0) {
                int num_pts_on_the_fly = 0;
                for (const auto &pt: SFMConstruct) {
                    if (pt.second.obs.size() >= 3) ++num_pts_on_the_fly;
                }

                auto id_i = int_frameid2_time_frameid.at(i);
                auto id_j = int_frameid2_time_frameid.at(j);

                auto count_visible_in_pair = [&](const std::vector<int>& selected_ids) -> int {
                    int count = 0;
                    for (int line_id : selected_ids) {
                        auto line_it = SFMLineConstruct.find(line_id);
                        if (line_it == SFMLineConstruct.end()) continue;
                        const auto& sfm_line = line_it->second;
                        if (sfm_line.obs.count(id_i) && sfm_line.obs.count(id_j)) {
                            ++count;
                        }
                    }
                    return count;
                };

                auto quality_scale = [](double q) -> double {
                    q = std::clamp(q, 0.0, 1.0);
                    return 0.5 + 0.5 * q;
                };

                const int MAX_NORMAL_CONSTRAINTS_PER_PAIR = 10;
                const int MAX_EPI_CONSTRAINTS_PER_PAIR = 10;
                if (LINE_USE_NORMAL_RESIDUAL > 0 &&
                    last_window_lines_used_normal >= LINE_MIN_USED_FOR_ALIGNMENT) {
                    int pair_normal_count = count_visible_in_pair(selected_normal_line_ids);
                    if (pair_normal_count >= LINE_MIN_USED_FOR_ALIGNMENT) {
                        double line_weight_normal = static_cast<double>(num_pts_on_the_fly) /
                                                   static_cast<double>(pair_normal_count);
                        if (line_weight_normal > 50.0) line_weight_normal = 50.0;
                        line_weight_normal *= LINE_RESIDUAL_WEIGHT * LINE_NORMAL_WEIGHT;

                        int line_constraints_added = 0;
                        for (int line_id : selected_normal_line_ids) {
                            if (line_constraints_added >= MAX_NORMAL_CONSTRAINTS_PER_PAIR) break;
                            auto line_it = SFMLineConstruct.find(line_id);
                            if (line_it == SFMLineConstruct.end()) continue;
                            const auto& sfm_line = line_it->second;
                            if (!sfm_line.obs.count(id_i) || !sfm_line.obs.count(id_j)) continue;

                            const auto& obs_i = sfm_line.obs.at(id_i);
                            Eigen::Vector3d n_i = obs_i.start_point.cross(obs_i.end_point).normalized();
                            Eigen::RowVector3d n_b_T = n_i.transpose() * Rbc_;

                            Eigen::Vector3d visual_T_body = rotation[i].transpose() * (position[j] - position[i]) / 100.0;
                            Eigen::Vector3d imu_dP_body = imu_meas[i].dP_;
                            Eigen::Vector3d extrinsics_body = rotation[i].transpose() * rotation[j] * pbc_ - pbc_;

                            Eigen::RowVector3d J_line_v = n_b_T * (-dt);
                            double J_line_s = n_b_T * visual_T_body;
                            Eigen::RowVector3d J_line_g = n_b_T * (rotation[i].transpose() * (dt * dt / 2.0)) * G.norm();
                            double rhs_line = n_b_T * (imu_dP_body + extrinsics_body);

                            Matrix<double, 1, 10> line_A_row;
                            line_A_row.setZero();
                            line_A_row.block<1,3>(0,0) = J_line_v;
                            line_A_row(0, 6) = J_line_s;
                            line_A_row.block<1,3>(0,7) = J_line_g;

                            double q_norm = 1.0;
                            auto q_it = selected_normal_line_scores.find(line_id);
                            if (q_it != selected_normal_line_scores.end()) q_norm = q_it->second;
                            double w_norm = line_weight_normal * quality_scale(q_norm);
                            r_A += w_norm * line_A_row.transpose() * line_A_row;
                            r_b += w_norm * line_A_row.transpose() * rhs_line;
                            line_constraints_added++;
                        }
                    }
                }
                if (LINE_USE_EPIPOLAR_RESIDUAL > 0 &&
                    LINE_EPIPOLAR_WEIGHT > 0 &&
                    last_window_lines_used_epi >= LINE_MIN_USED_FOR_ALIGNMENT) {
                    int pair_epi_count = count_visible_in_pair(selected_epipolar_line_ids);
                    if (pair_epi_count >= LINE_MIN_USED_FOR_ALIGNMENT) {
                        double line_weight_epi = LINE_RESIDUAL_WEIGHT * LINE_EPIPOLAR_WEIGHT;

                        int epi_constraints_added = 0;
                        for (int line_id : selected_epipolar_line_ids) {
                            if (epi_constraints_added >= MAX_EPI_CONSTRAINTS_PER_PAIR) break;
                            auto line_it = SFMLineConstruct.find(line_id);
                            if (line_it == SFMLineConstruct.end()) continue;
                            const auto& sfm_line = line_it->second;
                            if (!sfm_line.obs.count(id_i) || !sfm_line.obs.count(id_j)) continue;

                            const auto& obs_i = sfm_line.obs.at(id_i);
                            const auto& obs_j = sfm_line.obs.at(id_j);

                            Eigen::Vector3d n_i = obs_i.start_point.cross(obs_i.end_point).normalized();
                            Eigen::Vector3d n_j = obs_j.start_point.cross(obs_j.end_point).normalized();
                            Eigen::Matrix3d R_ci_cj = Rbc_.transpose() * rotation[i].transpose() * rotation[j] * Rbc_;
                            Eigen::Vector3d d_ci = (R_ci_cj * n_j).cross(n_i);
                            double d_norm = d_ci.norm();
                            if (d_norm <= 1e-6) continue;
                            d_ci /= d_norm;
                            Eigen::RowVector3d d_b_T = d_ci.transpose() * Rbc_;
                            Eigen::Vector3d visual_T_body = rotation[i].transpose() * (position[j] - position[i]) / 100.0;
                            Eigen::Vector3d imu_dP_body = imu_meas[i].dP_;
                            Eigen::Vector3d extrinsics_body = rotation[i].transpose() * rotation[j] * pbc_ - pbc_;
                            Eigen::RowVector3d J_epi_v = d_b_T * (-dt);
                            double J_epi_s = d_b_T * visual_T_body;
                            Eigen::RowVector3d J_epi_g =
                                    d_b_T * (rotation[i].transpose() * (dt * dt / 2.0)) * G.norm();
                            double rhs_epi = d_b_T * (imu_dP_body + extrinsics_body);

                            Matrix<double, 1, 10> epi_A_row;
                            epi_A_row.setZero();
                            epi_A_row.block<1,3>(0,0) = J_epi_v;
                            epi_A_row(0, 6) = J_epi_s;
                            epi_A_row.block<1,3>(0,7) = J_epi_g;

                            double q_epi = 1.0;
                            auto q_it = selected_epipolar_line_scores.find(line_id);
                            if (q_it != selected_epipolar_line_scores.end()) q_epi = q_it->second;
                            double w_epi = line_weight_epi * quality_scale(q_epi);
                            r_A += w_epi * epi_A_row.transpose() * epi_A_row;
                            r_b += w_epi * epi_A_row.transpose() * rhs_epi;
                            epi_constraints_added++;
                        }
                    }
                }
            }



            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();

            Q += tmp_b.transpose() * cov_inv * tmp_b;
        }

        Eigen::VectorXd x;
        double s;
        Eigen::MatrixXd M_k2TM_k2 = A.bottomRightCorner<3, 3>();


        double mean_value = (M_k2TM_k2(0, 0) + M_k2TM_k2(1, 1) + M_k2TM_k2(2, 2)) / 3.0;

        double scale = 1 / mean_value;
        A = A * scale;
        b = b * scale;
        Q = Q * scale;

        if (!gravityRefine(A, -2. * b, Q, 1, x))
        {
            return false;
        }

        gravity = x.tail(3) * G.norm();
        s = x(n_state - 4) / 100.0;
        x(n_state - 4) = s;

        for (int i = int_frameid2_time_frameid.size() - 1; i >= 0; i--) {
            position[i] = s * position[i] - rotation[i] * pbc_;
            velocity[i] = rotation[i] * x.segment<3>(i * 3);
        }

        Eigen::Matrix3d rot0 = rotation[0].transpose();
        gravity = rot0 * gravity;
        for (int i = 0; i < int_frameid2_time_frameid.size(); i++) {
            rotation[i] = rot0 * rotation[i];
            position[i] = rot0 * position[i];
            velocity[i] = rot0 * velocity[i];
        }

        cout << "refine: " << gravity.norm() << " " << G.norm() << endl;

        return true;
    }




    void slimInit::build_LTL(Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr) {

        int num_view_ = int_frameid2_time_frameid.size();
        int track_id = 0;
        for (const auto &pt: SFMConstruct) {

            const auto &obs = pt.second.obs;
            if (obs.size() < 3) continue;

            TimeFrameId lbase_view_id = 0;
            TimeFrameId rbase_view_id = 0;

            select_base_views(obs,
                              lbase_view_id,
                              rbase_view_id);
            Eigen::MatrixXd tmp_LiGT_vec = Eigen::MatrixXd::Zero(3, num_view_ * 3);

            for (const auto &frame: obs) {
                TimeFrameId i_view_id = frame.first;
                if (i_view_id != lbase_view_id) {
                    Eigen::Matrix3d xi_cross = cross_product_matrix(frame.second.normalpoint);

                    Eigen::Matrix3d R_cicl =
                            frame_rot.at(i_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Matrix3d R_crcl =
                            frame_rot.at(rbase_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Vector3d a_lr_tmp_t =
                            cross_product_matrix(R_crcl * obs.at(lbase_view_id).normalpoint) *
                            obs.at(rbase_view_id).normalpoint;

                    Eigen::RowVector3d a_lr_t =
                            a_lr_tmp_t.transpose() * cross_product_matrix(obs.at(rbase_view_id).normalpoint);
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) =
                            a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) =
                            -a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    Eigen::Vector3d theta_lr_vector = cross_product_matrix(obs.at(rbase_view_id).normalpoint)
                                                      * R_crcl
                                                      * obs.at(lbase_view_id).normalpoint;

                    double theta_lr = theta_lr_vector.squaredNorm();
                    Eigen::Matrix3d Coefficient_B =
                            xi_cross * R_cicl * obs.at(lbase_view_id).normalpoint * a_lr_t *
                            frame_rot.at(rbase_view_id).transpose();
                    Eigen::Matrix3d Coefficient_C = theta_lr * cross_product_matrix(obs.at(i_view_id).normalpoint) *
                                                    frame_rot.at(i_view_id).transpose();
                    Eigen::Matrix3d Coefficient_D = -(Coefficient_B + Coefficient_C);
                    tmp_LiGT_vec.setZero();
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) += Coefficient_B;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(i_view_id) * 3) += Coefficient_C;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) += Coefficient_D;
                    Eigen::MatrixXd LTL_l_row = Coefficient_D.transpose() * tmp_LiGT_vec;
                    Eigen::MatrixXd LTL_r_row = Coefficient_B.transpose() * tmp_LiGT_vec;
                    Eigen::MatrixXd LTL_i_row = Coefficient_C.transpose() * tmp_LiGT_vec;
                    {
                        if (time_frameid2_int_frameid.at(lbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(lbase_view_id) * 3 - 3) += LTL_l_row.rightCols(
                                    LTL_l_row.cols() - 3);

                        if (time_frameid2_int_frameid.at(rbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(rbase_view_id) * 3 - 3) += LTL_r_row.rightCols(
                                    LTL_r_row.cols() - 3);

                        if (time_frameid2_int_frameid.at(i_view_id) > 0)
                            LTL.middleRows<3>(time_frameid2_int_frameid.at(i_view_id) * 3 - 3) += LTL_i_row.rightCols(
                                    LTL_i_row.cols() - 3);
                    }
                }
            }

            ++track_id;
        }
    }

    bool slimInit::solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(LTL, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        evectors.bottomRows(V.rows()) = V.col(V.cols() - 1);
        return true;

    }

    void slimInit::identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors) {
        const Eigen::VectorXd judgeValue = A_lr * evectors;
        const int positive_count = (judgeValue.array() > 0.0).cast<int>().sum();
        const int negative_count = judgeValue.rows() - positive_count;
        if (positive_count < negative_count) {
            evectors = -evectors;
        }
    }


    void slimInit::select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                                              TimeFrameId &lbase_view_id,
                                              TimeFrameId &rbase_view_id) {
        double best_criterion_value = -1.;
        std::vector<int> track_id;

        for (const auto &frame: track) {
            int id = time_frameid2_int_frameid.at(frame.first);
            track_id.push_back(id);
        }

        size_t track_size = track_id.size(); 
        for (int i = 0; i < track_size - 1; ++i) {
            for (int j = i + 1; j < track_size; ++j) {

                const TimeFrameId &i_view_id = int_frameid2_time_frameid.at(track_id[i]);
                const TimeFrameId &j_view_id = int_frameid2_time_frameid.at(track_id[j]);

                const Eigen::Vector3d &i_coord = track.at(i_view_id).normalpoint;
                const Eigen::Vector3d &j_coord = track.at(j_view_id).normalpoint;
                const Eigen::Matrix3d &R_i = frame_rot.at(i_view_id);
                const Eigen::Matrix3d &R_j = frame_rot.at(j_view_id);
                const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
                const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

                double criterion_value = theta_ij.norm();

                if (criterion_value > best_criterion_value) {

                    best_criterion_value = criterion_value;

                    if (i_view_id < j_view_id) {
                        lbase_view_id = i_view_id;
                        rbase_view_id = j_view_id;
                    } else {
                        lbase_view_id = j_view_id;
                        rbase_view_id = i_view_id;
                    }

                }
            }
        }
    }

}
