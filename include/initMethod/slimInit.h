
#ifndef DRT_VIO_SLIMINIT_H
#define DRT_VIO_SLIMINIT_H
#include "drtVioInit.h"
#include "utils/eigenUtils.hpp"
#include <memory>
namespace DRT {


    class slimInit : public drtVioInit
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        slimInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc);

        virtual bool process();

        void build_LTL(Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr);

        bool solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors);

        void identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors);

        void select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                               TimeFrameId &lbase_view_id,
                               TimeFrameId &rbase_view_id);

        bool linearAlignment();

        using Ptr = std::shared_ptr<slimInit>;
    };


}
#endif 
