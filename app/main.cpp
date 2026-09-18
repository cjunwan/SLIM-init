
#include "io/datasetIO.h"
#include "io/datasetIOEuroc.h"
#include "featureTracker/featureTracker.h"
#include "initMethod/lineFeatureTracker.h"
#include "featureTracker/parameters.h"
#include "IMU/imuPreintegrated.hpp"
#include "initMethod/drtVioInit.h"
#include "initMethod/slimInit.h"
#include "utils/eigenUtils.hpp"
#include "utils/ticToc.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: ./run_euroc <config.yaml> <output_tag>\n"
                  << "  config.yaml : dataset, calibration and method parameters (see config/euroc.yaml)\n"
                  << "  output_tag  : name of the result file, e.g. MH_01 -> <output_path>/slim_init_MH_01.txt\n";
        return -1;
    }

    const std::string config_path = argv[1];
    const std::string output_tag = argv[2];
    if (!fs::exists(config_path)) {
        std::cerr << "[ERROR] Config file not found: " << config_path << "\n";
        return -1;
    }
    std::ofstream save_file;

    auto dataset_io = struct_vio::EurocIO();
    dataset_io.read(config_path);
    auto vio_dataset = dataset_io.get_data();
    readParameters(config_path);

    FeatureTracker trackerData;
    trackerData.readIntrinsicParameter(config_path);

    auto line_tracker = std::make_shared<DRT::LineFeatureTracker>();
    line_tracker->readIntrinsicParameter(config_path);

    if (OUTPUT_PATH.empty()) {
        OUTPUT_PATH = "result";
    }
    std::error_code out_dir_ec;
    fs::create_directories(OUTPUT_PATH, out_dir_ec);
    if (out_dir_ec) {
        std::cerr << "[WARN] Failed to create output directory: " << OUTPUT_PATH
                  << " (" << out_dir_ec.message() << ")" << std::endl;
    }
    const std::string result_path = OUTPUT_PATH + "/slim_init_" + output_tag + ".txt";
    save_file.open(result_path);
    if (!save_file.is_open()) {
        std::cerr << "[WARN] Failed to open result file: " << result_path << std::endl;
    }

    PUB_THIS_FRAME = true;
    double sf = std::sqrt(double(IMU_FREQ));
    size_t init_windows = 0;
    size_t fail_windows = 0;
    size_t acc_check_fails = 0;
    size_t skipped_windows = 0;
    size_t scale_success_windows = 0;
    size_t evaluated_windows = 0;
    double sum_scale_error_sq = 0.0;
    double sum_pose_error_sq = 0.0;
    double sum_gravity_error_sq = 0.0;
    double sum_velo_error_sq = 0.0;
    double sum_gyro_bias_error = 0.0;

    for (int i = 0; i < vio_dataset->get_image_timestamps().size() - 100; i += 10) {



        DRT::drtVioInit::Ptr  pDrtVioInit(new DRT::slimInit(RIC[0], TIC[0]));

        pDrtVioInit->line_tracker = line_tracker;

        std::vector<int> idx;
        for (int j = i; j < 100 + i; j += 1)
            idx.push_back(j);

        double last_img_t_s, cur_img_t_s;
        bool first_img = true;
        bool init_feature = true;


        trackerData.reset();
        line_tracker = std::make_shared<DRT::LineFeatureTracker>();
        line_tracker->readIntrinsicParameter(config_path);
        pDrtVioInit->line_tracker = line_tracker;

        std::vector<double> idx_time;
        std::vector<int64_t> idx_time_ns; 

        for (int i: idx) {

            int64_t t_ns = vio_dataset->get_image_timestamps()[i];

            cur_img_t_s = t_ns * 1e-9;

            cv::Mat img = vio_dataset->get_image_data(t_ns)[0].image;

            Eigen::Matrix3d R_pred = Eigen::Matrix3d::Identity();
            if (!first_img) {
                auto GyroData = vio_dataset->get_gyro_data();
                auto AccelData = vio_dataset->get_accel_data();
                std::vector<MotionData> imu_segment;
                for (size_t k = 0; k < GyroData.size(); k++) {
                    double timestamp = GyroData[k].timestamp_ns * 1e-9;
                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s) {
                        MotionData imu_data;
                        imu_data.timestamp = timestamp;
                        imu_data.imu_acc = AccelData[k].data;
                        imu_data.imu_gyro = GyroData[k].data;
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s) break; 
                }
                
                if (!imu_segment.empty()) {
                    vio::IMUBias bias;
                    vio::IMUCalibParam imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                    vio::IMUPreintegrated imu_preint_tracker(bias, &imu_calib, last_img_t_s, cur_img_t_s);
                    
                    int n = imu_segment.size() - 1;
                    for (int k = 0; k < n; k++) {
                         double dt;
                         Eigen::Vector3d gyro, acc;
                         if (k == 0 && k < (n-1)) {
                             float tab = imu_segment[k+1].timestamp - imu_segment[k].timestamp;
                             float tini = imu_segment[k].timestamp - last_img_t_s;
                             acc = (imu_segment[k+1].imu_acc + imu_segment[k].imu_acc - (imu_segment[k+1].imu_acc - imu_segment[k].imu_acc) * (tini/tab)) * 0.5f;
                             gyro = (imu_segment[k+1].imu_gyro + imu_segment[k].imu_gyro - (imu_segment[k+1].imu_gyro - imu_segment[k].imu_gyro) * (tini/tab)) * 0.5f;
                             dt = imu_segment[k+1].timestamp - last_img_t_s;
                         } else if (k < (n-1)) {
                             acc = (imu_segment[k].imu_acc + imu_segment[k+1].imu_acc) * 0.5f;
                             gyro = (imu_segment[k].imu_gyro + imu_segment[k+1].imu_gyro) * 0.5f;
                             dt = imu_segment[k+1].timestamp - imu_segment[k].timestamp;
                         } else if (k > 0 && k == n-1) {
                             float tab = imu_segment[k+1].timestamp - imu_segment[k].timestamp;
                             float tend = imu_segment[k+1].timestamp - cur_img_t_s;
                             acc = (imu_segment[k].imu_acc + imu_segment[k+1].imu_acc - (imu_segment[k+1].imu_acc - imu_segment[k].imu_acc) * (tend/tab)) * 0.5f;
                             gyro = (imu_segment[k].imu_gyro + imu_segment[k+1].imu_gyro - (imu_segment[k+1].imu_gyro - imu_segment[k].imu_gyro) * (tend/tab)) * 0.5f;
                             dt = cur_img_t_s - imu_segment[k].timestamp;
                         } else if (k==0 && k==n-1) {
                             acc = imu_segment[k].imu_acc; gyro = imu_segment[k].imu_gyro; dt = cur_img_t_s - last_img_t_s;
                         }
                         if(dt > 0) imu_preint_tracker.integrate_new_measurement(gyro, acc, dt);
                    }
                    Eigen::Matrix3d dR = imu_preint_tracker.get_delta_rotation(bias).matrix();
                    Eigen::Matrix3d R_c_b = RIC[0];
                    Eigen::Matrix3d R_ck_ckp1 = R_c_b.transpose() * dR * R_c_b;
                    R_pred = R_ck_ckp1.transpose();
                }
            }
            trackerData.readImage(img, t_ns * 1e-9);

            line_tracker->readImage(img, cur_img_t_s, R_pred);

            for (unsigned int i = 0;; i++) {
                bool completed = false;
                completed |= trackerData.updateID(i);
                if (!completed)
                    break;
            }

            auto &un_pts = trackerData.cur_un_pts;
            auto &cur_pts = trackerData.cur_pts;
            auto &ids = trackerData.ids;
            auto &pts_velocity = trackerData.pts_velocity;

            Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1 >> >>
                    image;
            for (unsigned int i = 0; i < ids.size(); i++) {
                if (trackerData.track_cnt[i] > 1) {
                    int v = ids[i];
                    int feature_id = v / NUM_OF_CAM;
                    int camera_id = v % NUM_OF_CAM;
                    double x = un_pts[i].x;
                    double y = un_pts[i].y;
                    double z = 1;
                    double p_u = cur_pts[i].x;
                    double p_v = cur_pts[i].y;
                    double velocity_x = pts_velocity[i].x;
                    double velocity_y = pts_velocity[i].y;
                    assert(camera_id == 0);
                    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                    image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
                }
            }

            if (init_feature) {
                init_feature = false;
                continue;
            }

            if (pDrtVioInit->addFeatureCheckParallax(cur_img_t_s, image, 0.0)) {

                idx_time.push_back(cur_img_t_s);
                idx_time_ns.push_back(t_ns); 

                std::cout << "add image is: " << fixed << cur_img_t_s << " image number is: " << idx_time.size()
                          << std::endl;

                if (first_img) {
                    last_img_t_s = cur_img_t_s;
                    first_img = false;
                    continue;
                }

                auto GyroData = vio_dataset->get_gyro_data();
                auto AccelData = vio_dataset->get_accel_data();

                std::vector<MotionData> imu_segment;

                for (size_t i = 0; i < GyroData.size(); i++) {
                    double timestamp = GyroData[i].timestamp_ns * 1e-9;

                    MotionData imu_data;
                    imu_data.timestamp = timestamp;
                    imu_data.imu_acc = AccelData[i].data;
                    imu_data.imu_gyro = GyroData[i].data;

                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                        break;
                    }
                }

                vio::IMUBias bias;
                vio::IMUCalibParam
                        imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_img_t_s, cur_img_t_s);

                int n = imu_segment.size() - 1;

                for (int i = 0; i < n; i++) {
                    double dt;
                    Eigen::Vector3d gyro;
                    Eigen::Vector3d acc;

                    if (i == 0 && i < (n - 1))               
                    {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tini = imu_segment[i].timestamp - last_img_t_s;
                        CHECK(tini >= 0);
                        acc = (imu_segment[i + 1].imu_acc + imu_segment[i].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tini / tab)) * 0.5f;
                        gyro = (imu_segment[i + 1].imu_gyro + imu_segment[i].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tini / tab)) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - last_img_t_s;
                    } else if (i < (n - 1))      
                    {
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                    } else if (i > 0 && i == n - 1) {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tend = imu_segment[i + 1].timestamp - cur_img_t_s;
                        CHECK(tend >= 0);
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tend / tab)) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tend / tab)) * 0.5f;
                        dt = cur_img_t_s - imu_segment[i].timestamp;
                    } else if (i == 0 && i == (n - 1)) {
                        acc = imu_segment[i].imu_acc;
                        gyro = imu_segment[i].imu_gyro;
                        dt = cur_img_t_s - last_img_t_s;
                    }

                    CHECK(dt >= 0);
                    imu_preint.integrate_new_measurement(gyro, acc, dt);
                }

                pDrtVioInit->addImuMeasure(imu_preint);

                last_img_t_s = cur_img_t_s;


            }

            if (idx_time.size() >= 10) break;

            if (SHOW_TRACK) {
                cv::Mat show_img;
                cv::Mat bg_img = line_tracker->getCurrentImage();
                if (bg_img.empty()) {
                    bg_img = img;
                }

                if (bg_img.channels() == 1) {
                    cv::cvtColor(bg_img, show_img, cv::COLOR_GRAY2RGB);
                } else {
                    show_img = bg_img.clone();
                }

                const std::vector<cv::Point2f> vis_pts = line_tracker->undistortPoints(trackerData.cur_pts);
                for (size_t j = 0; j < vis_pts.size() && j < trackerData.track_cnt.size(); ++j) {
                    const double len = std::min(1.0, static_cast<double>(trackerData.track_cnt[j]) / WINDOW_SIZE);
                    cv::circle(show_img, vis_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }

                const auto &all_lines = line_tracker->getCurrentKeylines();
                for (const auto &kl: all_lines) {
                    cv::line(show_img, cv::Point2f(kl.startPointX, kl.startPointY),
                             cv::Point2f(kl.endPointX, kl.endPointY), cv::Scalar(200, 200, 200), 1);
                }

                const auto &matched_lines = line_tracker->getMatchedLines();
                for (const auto &kl: matched_lines) {
                    cv::line(show_img, cv::Point2f(kl.startPointX, kl.startPointY),
                             cv::Point2f(kl.endPointX, kl.endPointY), cv::Scalar(255, 0, 0), 2);
                }

                const auto &vp_lines = line_tracker->getVPAssignedLines();
                for (const auto &kl: vp_lines) {
                    cv::line(show_img, cv::Point2f(kl.startPointX, kl.startPointY),
                             cv::Point2f(kl.endPointX, kl.endPointY), cv::Scalar(0, 255, 0), 3);
                }

                const auto &vps = line_tracker->getVanishingPoints();
                for (const auto &vp_pair: vps) {
                    Eigen::Vector2d vp_img_pt;
                    if (!line_tracker->projectVanishingPointToImage(vp_pair.second, vp_img_pt)) {
                        continue;
                    }
                    const cv::Point2f vp_pixel(vp_img_pt(0), vp_img_pt(1));
                    constexpr int cross_size = 15;
                    cv::line(show_img, cv::Point2f(vp_pixel.x - cross_size, vp_pixel.y),
                             cv::Point2f(vp_pixel.x + cross_size, vp_pixel.y), cv::Scalar(0, 255, 255), 3);
                    cv::line(show_img, cv::Point2f(vp_pixel.x, vp_pixel.y - cross_size),
                             cv::Point2f(vp_pixel.x, vp_pixel.y + cross_size), cv::Scalar(0, 255, 255), 3);
                    cv::circle(show_img, vp_pixel, 20, cv::Scalar(0, 255, 255), 2);
                }

                cv::namedWindow("IMAGE", WINDOW_AUTOSIZE);
                cv::imshow("IMAGE", show_img);
                cv::waitKey(1);
            }



        }

        if (idx_time.size() < 10) {
            std::cout << "[WINDOW_CHECK] Insufficient keyframes: " << idx_time.size() << " / 10. Skipping window." << std::endl;
            skipped_windows++;
            continue;
        }


        bool is_good = true;
        if (ACC_CHECK_ENABLE > 0) {
            is_good = pDrtVioInit->checkAccError();
        }

        if (!is_good)
        {
            std::cout << "[ACC_CHECK] Acceleration check FAILED. Skipping window." << std::endl;
            acc_check_fails++;
            init_windows++;
            continue;
        }

        pDrtVioInit->SFMLineConstruct = line_tracker->getTrackedLines();

        bool process_result = pDrtVioInit->process();
        if (!process_result) {
            fail_windows++;
        }
        init_windows++;
        evaluated_windows++;  // every window on which initialization was attempted

        if (!process_result) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: "  << "nan" << std::endl;
            save_file << "gt_vel_rot: " << "nan" << " " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }
        std::vector<Eigen::Vector3d> gt_pos;
        std::vector<Eigen::Matrix3d> gt_rot;
        std::vector<Eigen::Vector3d> gt_vel;
        std::vector<Eigen::Vector3d> gt_g_imu;
        std::vector<Eigen::Vector3d> gt_angluar_vel;
        Eigen::Vector3d avgBg;
        avgBg.setZero();

        // Ground truth is rarely sampled exactly at a keyframe time. EuRoC logs
        // it at 200 Hz so an exact match nearly always exists, but a dataset
        // recorded at 10 Hz against a 30 Hz camera would lose most of its
        // windows to a match-only lookup. Fall back to interpolating between the
        // two bracketing samples.
        auto get_traj = [&](double timeStamp, struct_vio::GtData &rhs) -> bool {
            const auto &gt_data = vio_dataset->get_gt_state_data();
            if (gt_data.empty()) return false;

            auto upper = gt_data.lower_bound(timeStamp);
            if (upper != gt_data.end() && std::abs(upper->first - timeStamp) < 1e-3) {
                rhs = upper->second;
                rhs.timestamp_ns = static_cast<int64_t>(timeStamp * 1e9);
                return true;
            }
            if (upper == gt_data.begin() || upper == gt_data.end()) {
                return false;  // query falls outside the ground-truth interval
            }

            auto lower = std::prev(upper);
            const double t0 = lower->first;
            const double t1 = upper->first;
            const double dt = t1 - t0;
            if (dt <= 1e-12) return false;

            double alpha = (timeStamp - t0) / dt;
            alpha = std::min(1.0, std::max(0.0, alpha));

            const auto &g0 = lower->second;
            const auto &g1 = upper->second;
            rhs.timestamp_ns = static_cast<int64_t>(timeStamp * 1e9);
            rhs.position = (1.0 - alpha) * g0.position + alpha * g1.position;
            rhs.velocity = (1.0 - alpha) * g0.velocity + alpha * g1.velocity;
            rhs.bias_gyr = (1.0 - alpha) * g0.bias_gyr + alpha * g1.bias_gyr;
            rhs.bias_acc = (1.0 - alpha) * g0.bias_acc + alpha * g1.bias_acc;
            rhs.rotation = g0.rotation.slerp(alpha, g1.rotation).normalized();
            return true;
        };

        try {
            for (auto &t: idx_time) {
                struct_vio::GtData rhs;
                if (get_traj(t, rhs)) {
                    gt_pos.emplace_back(rhs.position);
                    gt_vel.emplace_back(rhs.velocity);
                    gt_rot.emplace_back(rhs.rotation.toRotationMatrix());
                    gt_g_imu.emplace_back(rhs.rotation.inverse() * G);

                    avgBg += rhs.bias_gyr;
                } else {
                    std::cout << "no gt pose,fail" << std::endl;
                    throw -1;
                }
            }
        } catch (...) {
            evaluated_windows--;  // no ground truth for this window, so it cannot be evaluated
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }

        avgBg /= idx_time.size();

        double rot_rmse = 0;
        for (int i = 0; i < idx_time.size() - 1; i++) {
            int j = i + 1;
            Eigen::Matrix3d rij_est = pDrtVioInit->rotation[i].transpose() * pDrtVioInit->rotation[j];
            Eigen::Matrix3d rij_gt = gt_rot[i].transpose() * gt_rot[j];
            Eigen::Quaterniond qij_est = Eigen::Quaterniond(rij_est);
            Eigen::Quaterniond qij_gt = Eigen::Quaterniond(rij_gt);
            double error =
                    std::acos(((qij_gt * qij_est.inverse()).toRotationMatrix().trace() - 1.0) / 2.0) * 180.0 / M_PI;
            rot_rmse += error * error;
        }
        rot_rmse /= (idx_time.size() - 1);
        rot_rmse = std::sqrt(rot_rmse);
        Eigen::Matrix<double, 3, Eigen::Dynamic> est_aligned_pose(3, idx_time.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> gt_aligned_pose(3, idx_time.size());

        for (int i = 0; i < idx_time.size(); i++) {
            est_aligned_pose(0, i) = pDrtVioInit->position[i](0);
            est_aligned_pose(1, i) = pDrtVioInit->position[i](1);
            est_aligned_pose(2, i) = pDrtVioInit->position[i](2);

            gt_aligned_pose(0, i) = gt_pos[i](0);
            gt_aligned_pose(1, i) = gt_pos[i](1);
            gt_aligned_pose(2, i) = gt_pos[i](2);
        }


        Eigen::Matrix4d Tts = Eigen::umeyama(est_aligned_pose, gt_aligned_pose, true);
        Eigen::Matrix3d cR = Tts.block<3, 3>(0, 0);
        Eigen::Vector3d t = Tts.block<3, 1>(0, 3);
        double s = cR.determinant();
        s = pow(s, 1.0 / 3);
        Eigen::Matrix3d R = cR / s;

        double pose_rmse = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            Eigen::Vector3d target_pose = R * est_aligned_pose.col(i) + t;
            pose_rmse += (target_pose - gt_aligned_pose.col(i)).dot(target_pose - gt_aligned_pose.col(i));

        }
        pose_rmse /= idx_time.size();
        pose_rmse = std::sqrt(pose_rmse);

        std::cout << "vins sfm pose rmse: " << pose_rmse << std::endl;
        double gravity_error =
                180. * std::acos(pDrtVioInit->gravity.normalized().dot(gt_g_imu[0].normalized())) / EIGEN_PI;
        Eigen::Vector3d Bgs = pDrtVioInit->biasg;

        LOG(INFO) << "calculate bias: " << Bgs.x() << " " << Bgs.y() << " " << Bgs.z();
        LOG(INFO) << "gt bias: " << avgBg.x() << " " << avgBg.y() << " " << avgBg.z();

        const double scale_error = std::abs(s - 1.);
        const double gyro_bias_error = 100. * std::abs(Bgs.norm() - avgBg.norm()) / avgBg.norm();
        const double gyro_bias_error2 = 180. * std::acos(Bgs.normalized().dot(avgBg.normalized())) / EIGEN_PI;
        const double pose_error = pose_rmse;
        const double rot_error = rot_rmse;
        double velo_norm_rmse = 0;
        double mean_velo = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            velo_norm_rmse += (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm()) *
                              (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm());
            mean_velo += gt_vel[i].norm();
        }

        velo_norm_rmse /= idx_time.size();
        velo_norm_rmse = std::sqrt(velo_norm_rmse);
        mean_velo = mean_velo / idx_time.size();
        double v0_error = std::abs(gt_vel[0].norm() - pDrtVioInit->velocity[0].norm());

        std::cout << "integrate time: " << fixed << *idx_time.begin() << " " << *idx_time.rbegin() << " "
                  << *idx_time.rbegin() - *idx_time.begin() << std::endl;
        std::cout << "pose error: " << pose_error << " m" << std::endl;
        std::cout << "biasg error: " << gyro_bias_error << " %" << std::endl;
        std::cout << "gravity_error: " << gravity_error << std::endl;
        std::cout << "scale error: " << scale_error * 100 << " %" << std::endl;
        std::cout << "velo error: " << velo_norm_rmse << " m/s" << std::endl;
        std::cout << "v0_error: " << v0_error << std::endl;
        std::cout << "rot error: " << rot_error << std::endl;

        if (scale_error < 0.5) {
            scale_success_windows++;
            sum_scale_error_sq += scale_error * scale_error;
            sum_pose_error_sq += pose_error * pose_error;
            sum_gravity_error_sq += gravity_error * gravity_error;
            sum_velo_error_sq += velo_norm_rmse * velo_norm_rmse;
            sum_gyro_bias_error += gyro_bias_error;
        }

        if (std::abs(s - 1) > 0.5 or std::abs(gravity_error) > 10) {
            LOG(INFO) << "===scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " scale_gravity_fail" << std::endl;
            save_file << "scale_error: " << scale_error * 100 << " %" << std::endl;
            save_file << "pose_error: " << pose_error << " m" << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << " %" << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << " m/s" << std::endl;
            save_file << "gravity_error: " << gravity_error << std::endl;
            save_file << "rot_error: " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        } else {
            LOG(INFO) << "***scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " good" << std::endl;
            save_file << "scale_error: " << scale_error * 100 << " %" << std::endl;
            save_file << "pose_error: " << pose_error << " m" << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << " %" << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << " m/s" << std::endl;
            save_file << "gravity_error: " << gravity_error <<  std::endl;
            save_file << "rot_error: " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        }

    }
    std::cout << "\n========================================" << std::endl;
    std::cout << "SUMMARY STATISTICS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Init Windows (>=10 keyframes): " << init_windows << std::endl;
    std::cout << "Acc Check Fails: " << acc_check_fails << std::endl;
    std::cout << "Skipped (insufficient keyframes): " << skipped_windows << std::endl;
    std::cout << "Solver Failures: " << fail_windows << std::endl;
    save_file << "init_windows_ge10: " << init_windows << std::endl;
    save_file << "acc_check_fails: " << acc_check_fails << std::endl;
    save_file << "skipped_windows: " << skipped_windows << std::endl;
    save_file << "solver_failures: " << fail_windows << std::endl;
    std::cout << "Evaluated Windows: " << evaluated_windows << std::endl;
    std::cout << "Scale Success (<100%): " << scale_success_windows << " / " << evaluated_windows << std::endl;
    save_file << "evaluated_windows: " << evaluated_windows << std::endl;
    save_file << "scale_success_windows: " << scale_success_windows << std::endl;
    if (evaluated_windows > 0) {
        double success_rate = 100.0 * scale_success_windows / static_cast<double>(evaluated_windows);
        std::cout << "Init Success Rate: " << success_rate << " %" << std::endl;
        save_file << "init_success_rate: " << success_rate << " %" << std::endl;
    }
    if (scale_success_windows > 0) {
        double n = static_cast<double>(scale_success_windows);
        double scale_rmse = std::sqrt(sum_scale_error_sq / n);
        double pose_rmse = std::sqrt(sum_pose_error_sq / n);
        double gravity_rmse = std::sqrt(sum_gravity_error_sq / n);
        double velo_rmse = std::sqrt(sum_velo_error_sq / n);
        double gyro_bias_mean = sum_gyro_bias_error / n;

        std::cout << "\n=== Success-Only Metrics (" << scale_success_windows << " windows) ===" << std::endl;
        std::cout << "  Scale RMSE:             " << scale_rmse << std::endl;
        std::cout << "  Pose RMSE:              " << pose_rmse << " m" << std::endl;
        std::cout << "  Gyro Bias Error (mean): " << gyro_bias_mean << " %" << std::endl;
        std::cout << "  Gravity Dir. RMSE:      " << gravity_rmse << " deg" << std::endl;
        std::cout << "  Velocity RMSE:          " << velo_rmse << " m/s" << std::endl;

        save_file << "success_scale_rmse: " << scale_rmse << std::endl;
        save_file << "success_pose_rmse: " << pose_rmse << " m" << std::endl;
        save_file << "success_gyro_bias_mean: " << gyro_bias_mean << " %" << std::endl;
        save_file << "success_gravity_rmse: " << gravity_rmse << " deg" << std::endl;
        save_file << "success_velocity_rmse: " << velo_rmse << " m/s" << std::endl;
    }

}
