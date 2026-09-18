
#ifndef DATASET_IO_EUROC_H
#define DATASET_IO_EUROC_H

#include <experimental/filesystem>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#include "io/datasetIO.h"
#include "utils/eigenTypes.h"

namespace fs = std::experimental::filesystem;

namespace struct_vio {

    class EurocVioDataset : public VioDataset {
    public:

        size_t num_cams;

        std::string dataset_base_path;
        std::vector<std::string> image_folder;
        std::string imu_path;

        std::vector<int64_t> image_timestamps;
        std::unordered_map<int64_t, std::string> image_path;

        Eigen::vector<AccelData> accel_data;
        Eigen::vector<GyroData> gyro_data;

        std::vector<int64_t> gt_timestamps;       
        Eigen::map<double, GtData> gt_data;


    public:
        ~EurocVioDataset() {};

        size_t get_num_cams() const { return num_cams; }

        std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

        const Eigen::vector<AccelData> &get_accel_data() const { return accel_data; }

        const Eigen::vector<GyroData> &get_gyro_data() const { return gyro_data; }

        const std::vector<int64_t> &get_gt_timestamps() const {
            return gt_timestamps;
        }

        const Eigen::map<double, GtData> &get_gt_state_data() const {
            return gt_data;
        }

        std::vector<ImageData> get_image_data(int64_t t_ns) {
            std::vector<ImageData> res(num_cams);

            for (size_t i = 0; i < num_cams; i++) {
                std::string full_image_path =
                        dataset_base_path + image_folder[i] + "/data/" + image_path[t_ns];

                if (file_exists(full_image_path)) {
                    cv::Mat image = cv::imread(full_image_path, -1);
                    res[i].image = image;
                }
            }

            return res;
        } 
    };

    class EurocIO : public DatasetIoInterface {
    public:
        EurocIO() {};

        void read(const std::string &config_path) {

            cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
            if (!fsSettings.isOpened()) {
                LOG(ERROR) << "ERROR: Wrong path to settings";
            }

            std::string data_path = fsSettings["data_path"];

            LOG(INFO) << "data_path: " << data_path << std::endl;

            if (!fs::exists(data_path)) {
                LOG(ERROR) << "No dataset found in " << data_path << std::endl;
            }

            int camera_num = fsSettings["camera_num"];

            std::vector<std::string> image_names;

            for (const auto &node: fsSettings["camera_name"]) {
                image_names.emplace_back(node);
            }
            LOG(INFO) << "image_names size: " << image_names.size();

            std::string imu_path = fsSettings["imu_path"];

            data.reset(new EurocVioDataset);

            data->dataset_base_path = data_path;
            data->num_cams = camera_num;
            data->image_folder = image_names;
            read_image_timestamps(data_path + image_names[0] + "/");
            read_imu_data(data_path + imu_path + "/");
            bool has_ground_gtruth = int(fsSettings["has_ground_truth"]);
            if (has_ground_gtruth) {
                std::string ground_truth_path = fsSettings["ground_truth_path"];
                read_gt_data_state(data_path + ground_truth_path + "/");

            }
        }

        void reset() { data.reset(); }

        VioDatasetPtr get_data() { return data; }

    private:
        void read_image_timestamps(const std::string &path) {

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            std::string line;
            while (std::getline(f, line)) {
                if (line[0] == '#')
                    continue;

                std::stringstream ss(line);

                char tmp;
                int64_t t_ns;
                std::string path;

                ss >> t_ns >> tmp >> path;

                data->image_timestamps.emplace_back(t_ns);
                data->image_path[t_ns] = path;
            }
        } 

        void read_imu_data(const std::string &path) {
            data->accel_data.clear();
            data->gyro_data.clear();

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            std::string line;
            while (std::getline(f, line)) {
                if (line[0] == '#')
                    continue;

                std::stringstream ss(line);

                char tmp;
                uint64_t timestamp;
                Eigen::Vector3d gyro, accel;

                ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
                   tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];

                data->accel_data.emplace_back();
                data->accel_data.back().timestamp_ns = timestamp;
                data->accel_data.back().data = accel;

                data->gyro_data.emplace_back();
                data->gyro_data.back().timestamp_ns = timestamp;
                data->gyro_data.back().data = gyro;
            }
        }

        void read_gt_data_state(const std::string &path) {
            data->gt_timestamps.clear();
            data->gt_data.clear();

            std::ifstream f(path + "data.csv");
            if (!f) {
                LOG(INFO) << "fail to open the file: " << path + "data.csv";
            }

            bool need_velocity_fallback = false;
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#')
                    continue;

                std::stringstream ss(line);
                std::vector<std::string> tokens;
                std::string token;
                while (std::getline(ss, token, ',')) {
                    tokens.emplace_back(token);
                }

                if (tokens.size() < 8) {
                    LOG(WARNING) << "invalid groundtruth row (too few columns): " << line;
                    continue;
                }

                uint64_t timestamp = 0;
                Eigen::Quaterniond q;
                Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                Eigen::Vector3d vel = Eigen::Vector3d::Zero();
                Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
                Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();

                try {
                    timestamp = static_cast<uint64_t>(std::stoll(tokens[0]));
                    pos[0] = std::stod(tokens[1]);
                    pos[1] = std::stod(tokens[2]);
                    pos[2] = std::stod(tokens[3]);
                    q.w() = std::stod(tokens[4]);
                    q.x() = std::stod(tokens[5]);
                    q.y() = std::stod(tokens[6]);
                    q.z() = std::stod(tokens[7]);

                    if (tokens.size() >= 11) {
                        vel[0] = std::stod(tokens[8]);
                        vel[1] = std::stod(tokens[9]);
                        vel[2] = std::stod(tokens[10]);
                    } else {
                        need_velocity_fallback = true;
                    }

                    if (tokens.size() >= 14) {
                        gyro_bias[0] = std::stod(tokens[11]);
                        gyro_bias[1] = std::stod(tokens[12]);
                        gyro_bias[2] = std::stod(tokens[13]);
                    }

                    if (tokens.size() >= 17) {
                        accel_bias[0] = std::stod(tokens[14]);
                        accel_bias[1] = std::stod(tokens[15]);
                        accel_bias[2] = std::stod(tokens[16]);
                    }
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse groundtruth row: " << line << " (" << e.what() << ")";
                    continue;
                }

                q.normalize();

                data->gt_timestamps.emplace_back(timestamp);

                GtData gt;
                gt.timestamp_ns = timestamp;
                gt.rotation = q;
                gt.position = pos;
                gt.velocity = vel;
                gt.bias_gyr = gyro_bias;
                gt.bias_acc = accel_bias;
                data->gt_data[timestamp * 1e-9] = gt;
            }

            if (need_velocity_fallback && !data->gt_data.empty()) {
                std::vector<double> gt_times_s;
                std::vector<Eigen::Vector3d> gt_positions;
                gt_times_s.reserve(data->gt_data.size());
                gt_positions.reserve(data->gt_data.size());
                for (const auto &traj : data->gt_data) {
                    gt_times_s.emplace_back(traj.first);
                    gt_positions.emplace_back(traj.second.position);
                }

                std::vector<Eigen::Vector3d> fallback_vel(gt_times_s.size(), Eigen::Vector3d::Zero());
                if (gt_times_s.size() >= 2) {
                    for (size_t i = 0; i < gt_times_s.size(); ++i) {
                        size_t i0 = (i == 0) ? 0 : i - 1;
                        size_t i1 = (i + 1 >= gt_times_s.size()) ? gt_times_s.size() - 1 : i + 1;
                        if (i0 == i1) {
                            fallback_vel[i].setZero();
                            continue;
                        }

                        const double dt = gt_times_s[i1] - gt_times_s[i0];
                        if (std::abs(dt) < 1e-12) {
                            fallback_vel[i].setZero();
                            continue;
                        }
                        fallback_vel[i] = (gt_positions[i1] - gt_positions[i0]) / dt;
                    }
                }

                size_t idx = 0;
                for (auto &traj : data->gt_data) {
                    traj.second.velocity = fallback_vel[idx++];
                }

                LOG(WARNING) << "groundtruth velocity columns are missing; fallback velocity is estimated from position and timestamps.";
            }
        }


        std::shared_ptr<EurocVioDataset> data;

    };

} 

#endif 
