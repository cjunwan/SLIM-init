#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "camodocal/gpl/gpl.h"


Eigen::VectorXd polyfit(Eigen::VectorXd& xVec, Eigen::VectorXd& yVec, int poly_order) {
    assert(poly_order > 0);
    assert(xVec.size() > poly_order);
    assert(xVec.size() == yVec.size());

    Eigen::MatrixXd A(xVec.size(), poly_order+1);
    Eigen::VectorXd B(xVec.size());

    for(int i = 0; i < xVec.size(); ++i) {
        const double x = xVec(i);
        const double y = yVec(i);

        double x_pow_k = 1.0;

        for(int k=0; k<=poly_order; ++k) {
            A(i,k) = x_pow_k;
            x_pow_k *= x;
        }

        B(i) = y;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd x = svd.solve(B);

    return x;
}

namespace camodocal
{

OCAMCamera::Parameters::Parameters()
 : Camera::Parameters(SCARAMUZZA)
 , m_C(0.0)
 , m_D(0.0)
 , m_E(0.0)
 , m_center_x(0.0)
 , m_center_y(0.0)
{
    memset(m_poly, 0, sizeof(double) * SCARAMUZZA_POLY_SIZE);
    memset(m_inv_poly, 0, sizeof(double) * SCARAMUZZA_INV_POLY_SIZE);
}



bool
OCAMCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (!boost::iequals(sModelType, "scaramuzza"))
        {
            return false;
        }
    }

    m_modelType = SCARAMUZZA;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);

    cv::FileNode n = fs["poly_parameters"];
    for(int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
        m_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

    n = fs["inv_poly_parameters"];
    for(int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
        m_inv_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

    n = fs["affine_parameters"];
    m_C = static_cast<double>(n["ac"]);
    m_D = static_cast<double>(n["ad"]);
    m_E = static_cast<double>(n["ae"]);

    m_center_x = static_cast<double>(n["cx"]);
    m_center_y = static_cast<double>(n["cy"]);

    return true;
}

void
OCAMCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "scaramuzza";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    fs << "poly_parameters";
    fs << "{";
    for(int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
        fs << std::string("p") + boost::lexical_cast<std::string>(i) << m_poly[i];
    fs << "}";

    fs << "inv_poly_parameters";
    fs << "{";
    for(int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
        fs << std::string("p") + boost::lexical_cast<std::string>(i) << m_inv_poly[i];
    fs << "}";

    fs << "affine_parameters";
    fs << "{" << "ac" << m_C
              << "ad" << m_D
              << "ae" << m_E
              << "cx" << m_center_x
              << "cy" << m_center_y << "}";

    fs.release();
}

OCAMCamera::Parameters&
OCAMCamera::Parameters::operator=(const OCAMCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_cameraName = other.m_cameraName;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_C = other.m_C;
        m_D = other.m_D;
        m_E = other.m_E;
        m_center_x = other.m_center_x;
        m_center_y = other.m_center_y;

        memcpy(m_poly, other.m_poly, sizeof(double) * SCARAMUZZA_POLY_SIZE);
        memcpy(m_inv_poly, other.m_inv_poly, sizeof(double) * SCARAMUZZA_INV_POLY_SIZE);
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const OCAMCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "scaramuzza" << std::endl;
    out << "   camera_name " << params.m_cameraName << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;

    out << std::fixed << std::setprecision(10);

    out << "Poly Parameters" << std::endl;
    for(int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
        out << std::string("p") + boost::lexical_cast<std::string>(i) << ": " << params.m_poly[i] << std::endl;

    out << "Inverse Poly Parameters" << std::endl;
    for(int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
        out << std::string("p") + boost::lexical_cast<std::string>(i) << ": " << params.m_inv_poly[i] << std::endl;

    out << "Affine Parameters" << std::endl;
    out << "            ac " << params.m_C << std::endl
        << "            ad " << params.m_D << std::endl
        << "            ae " << params.m_E << std::endl;
    out << "            cx " << params.m_center_x << std::endl
        << "            cy " << params.m_center_y << std::endl;

    return out;
}

OCAMCamera::OCAMCamera()
 : m_inv_scale(0.0)
{

}

OCAMCamera::OCAMCamera(const OCAMCamera::Parameters& params)
 : mParameters(params)
{
    m_inv_scale = 1.0 / (params.C() - params.D() * params.E());
}

Camera::ModelType
OCAMCamera::modelType(void) const
{
    return mParameters.modelType();
}

const std::string&
OCAMCamera::cameraName(void) const
{
    return mParameters.cameraName();
}

int
OCAMCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
OCAMCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}

void
OCAMCamera::estimateIntrinsics(const cv::Size& boardSize,
                               const std::vector< std::vector<cv::Point3f> >& objectPoints,
                               const std::vector< std::vector<cv::Point2f> >& imagePoints)
{
    std::vector< Eigen::Matrix3d > RList;
    std::vector< Eigen::Vector3d > TList;

    RList.reserve(imagePoints.size());
    TList.reserve(imagePoints.size());
    for (size_t image_index = 0; image_index < imagePoints.size(); ++image_index)
    {
        const std::vector<cv::Point3f>& objPts = objectPoints.at(image_index);
        const std::vector<cv::Point2f>& imgPts = imagePoints.at(image_index);

        assert(objPts.size() == imgPts.size());
        assert(objPts.size() == static_cast<unsigned int>(boardSize.width * boardSize.height));

        Eigen::MatrixXd M(objPts.size(), 6);

        for(size_t corner_index = 0; corner_index < objPts.size(); ++corner_index) {
            double X = objPts.at(corner_index).x;
            double Y = objPts.at(corner_index).y;
            assert(objPts.at(corner_index).z==0.0);
            
            double u = imgPts.at(corner_index).x;
            double v = imgPts.at(corner_index).y;

            M(corner_index, 0) = -v * X;
            M(corner_index, 1) = -v * Y;
            M(corner_index, 2) =  u * X;
            M(corner_index, 3) =  u * Y;
            M(corner_index, 4) = -v;
            M(corner_index, 5) =  u;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        assert(svd.matrixV().cols() == 6);
        Eigen::VectorXd h = -svd.matrixV().col(5);
        const double sr11 = h(0);
        const double sr12 = h(1);
        const double sr21 = h(2);
        const double sr22 = h(3);
        const double st1  = h(4);
        const double st2  = h(5);

        const double AA = square(sr11*sr12 + sr21*sr22);
        const double BB = square(sr11) + square(sr21);
        const double CC = square(sr12) + square(sr22);

        const double sr32_squared_1 = (- (CC-BB) + sqrt(square(CC-BB) + 4.0 * AA)) / 2.0;
        const double sr32_squared_2 = (- (CC-BB) - sqrt(square(CC-BB) + 4.0 * AA)) / 2.0;

        std::vector<double> sr32_squared_values;
        if (sr32_squared_1 > 0) sr32_squared_values.push_back(sr32_squared_1);
        if (sr32_squared_2 > 0) sr32_squared_values.push_back(sr32_squared_2);
        assert(!sr32_squared_values.empty());

        std::vector<double> sr32_values;
        std::vector<double> sr31_values;
        for (auto sr32_squared : sr32_squared_values) {
            for(int sign = -1; sign <= 1; sign += 2) {
                const double sr32 = static_cast<double>(sign) * std::sqrt(sr32_squared);
                sr32_values.push_back( sr32 );
                if (sr32_squared == 0.0) {
                    sr31_values.push_back(std::sqrt(CC-BB));
                    sr32_values.push_back( sr32 );
                    sr31_values.push_back(-std::sqrt(CC-BB));
                    
                    break; 
                } else {
                    sr31_values.push_back(- (sr11*sr12 + sr21*sr22) / sr32);
                }
            }
        }

        assert(!sr31_values.empty());
        assert(sr31_values.size() == sr32_values.size());
        
        std::vector<Eigen::Matrix3d> H_values;
        for(size_t i=0;i<sr31_values.size(); ++i) {
            const double sr31 = sr31_values.at(i);
            const double sr32 = sr32_values.at(i);
            const double lambda = 1.0 / sqrt(sr11*sr11 + sr21*sr21 + sr31*sr31);
            Eigen::Matrix3d H;
            H.setZero();
            H(0,0) = sr11; H(0,1) = sr12; H(0,2) = st1;
            H(1,0) = sr21; H(1,1) = sr22; H(1,2) = st2;
            H(2,0) = sr31; H(2,1) = sr32; H(2,2) = 0;

            H_values.push_back( lambda * H);
            H_values.push_back(-lambda * H);
        }

        for(auto& H : H_values) {
            Eigen::Matrix3d R;
            R.col(0) = H.col(0);
            R.col(1) = H.col(1);
            R.col(2) = H.col(0).cross(H.col(1));
        }
        
        std::vector<Eigen::Matrix3d> H_candidates;

        for (auto& H : H_values)
        {
            Eigen::MatrixXd A_mat(2 * imagePoints.at(image_index).size(), 4);
            Eigen::VectorXd B_vec(2 * imagePoints.at(image_index).size());
            A_mat.setZero();
            B_vec.setZero();

            size_t line_index = 0;
            const double& r11 = H(0,0);
            const double& r12 = H(0,1);
            const double& r21 = H(1,0);
            const double& r22 = H(1,1);
            const double& r31 = H(2,0);
            const double& r32 = H(2,1);
            const double& t1  = H(0);
            const double& t2  = H(1);
            for(size_t j=0; j<imagePoints.at(image_index).size(); ++j) {
                assert(line_index == 2 * j);

                const double& X = objectPoints.at(image_index).at(j).x;
                const double& Y = objectPoints.at(image_index).at(j).y;
                const double& u = imagePoints.at(image_index).at(j).x;
                const double& v = imagePoints.at(image_index).at(j).y;

                double A = r21 * X + r22 * Y + t2;
                double B = v * (r31 * X + r32 * Y);
                double C = r11 * X + r12 * Y + t1;
                double D = u * (r31 * X + r32 * Y);
                double rou = std::sqrt(u*u + v*v);

                
                A_mat(line_index+0, 0) = A;
                A_mat(line_index+1, 0) = C;
                A_mat(line_index+0, 1) = A * rou;
                A_mat(line_index+1, 1) = C * rou;
                A_mat(line_index+0, 2) = A * rou * rou;
                A_mat(line_index+1, 2) = C * rou * rou;
                
                A_mat(line_index+0, 3) = -v;
                A_mat(line_index+1, 3) = -u;
                B_vec(line_index+0) = B;
                B_vec(line_index+1) = D;

                line_index += 2;
            }

            assert(line_index == static_cast<unsigned int>(A_mat.rows()));
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

                Eigen::VectorXd x = svd.solve(B_vec);

                if (x(2) > 0 && x(3) > 0) {
                    H_candidates.push_back(H);
                }
            }
        }
        assert(H_candidates.size()==1);

        Eigen::Matrix3d& H = H_candidates.front();

        Eigen::Matrix3d R; 
        R.col(0) = H.col(0); 
        R.col(1) = H.col(1); 
        R.col(2) = H.col(0).cross(H.col(1)); 

        Eigen::Vector3d T = H.col(2);
        RList.push_back(R);
        TList.push_back(T);
    }
    Eigen::MatrixXd A_mat(2 * imagePoints.size() * imagePoints.at(0).size(), SCARAMUZZA_POLY_SIZE-1 + imagePoints.size());
    Eigen::VectorXd B_vec(2 * imagePoints.size() * imagePoints.at(0).size());
    A_mat.setZero();
    B_vec.setZero();

    size_t line_index = 0;
    for(size_t i = 0; i < imagePoints.size(); ++i) {
        const double& r11 = RList.at(i)(0,0);
        const double& r12 = RList.at(i)(0,1);
        const double& r21 = RList.at(i)(1,0);
        const double& r22 = RList.at(i)(1,1);
        const double& r31 = RList.at(i)(2,0);
        const double& r32 = RList.at(i)(2,1);
        const double& t1  = TList.at(i)(0);
        const double& t2  = TList.at(i)(1);
        for(size_t j=0; j<imagePoints.at(i).size(); ++j) {
            assert(line_index == 2 * (i * imagePoints.at(0).size() + j));

            const double& X = objectPoints.at(i).at(j).x;
            const double& Y = objectPoints.at(i).at(j).y;
            const double& u = imagePoints.at(i).at(j).x;
            const double& v = imagePoints.at(i).at(j).y;

            double A = r21 * X + r22 * Y + t2;
            double B = v * (r31 * X + r32 * Y);
            double C = r11 * X + r12 * Y + t1;
            double D = u * (r31 * X + r32 * Y);
            double rou = std::sqrt(u*u + v*v);

            for(int k=1;k<=SCARAMUZZA_POLY_SIZE-1;++k) {
                double pow_rou = 0.0;
                if (k == 1) {
                    pow_rou = 1.0;
                }
                else {
                    pow_rou = std::pow(rou, k);
                }

                A_mat(line_index+0, k-1) = A * pow_rou;
                A_mat(line_index+1, k-1) = C * pow_rou;
            }
            
            A_mat(line_index+0, SCARAMUZZA_POLY_SIZE-1+i) = -v;
            A_mat(line_index+1, SCARAMUZZA_POLY_SIZE-1+i) = -u;
            B_vec(line_index+0) = B;
            B_vec(line_index+1) = D;

            line_index += 2;
        }
    }

    assert(line_index == static_cast<unsigned int>(A_mat.rows()));

    Eigen::Matrix<double, SCARAMUZZA_POLY_SIZE, 1> poly_coeff;
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::VectorXd x = svd.solve(B_vec);

        poly_coeff[0] = x(0);
        poly_coeff[1] = 0.0;
        for(int i=1;i<poly_coeff.size()-1;++i) {
            poly_coeff[i+1] = x(i);
        }
        assert(x.size() == static_cast<unsigned int>(SCARAMUZZA_POLY_SIZE-1+TList.size()));
    }

    Parameters params = getParameters();
    params.C() = 1.0;
    params.D() = 0.0;
    params.E() = 0.0;

    params.center_x() = params.imageWidth() / 2.0;
    params.center_y() = params.imageHeight() / 2.0;
    
    for(size_t i=0; i<SCARAMUZZA_POLY_SIZE; ++i) {
        params.poly(i) = poly_coeff[i];
    }
    {
        std::vector<double> rou_vec;
        std::vector<double> z_vec;
        for(double rou = 0.0; rou <= (params.imageWidth() + params.imageHeight())/2; rou += 0.1) {
            double rou_pow_k = 1.0;
            double z = 0.0;

            for (int k = 0; k < SCARAMUZZA_POLY_SIZE; k++)
            {
                z += rou_pow_k * params.poly(k);
                rou_pow_k *= rou;
            }

            rou_vec.push_back(rou);
            z_vec.push_back(z);
        }

        assert(rou_vec.size() == z_vec.size());
        Eigen::VectorXd xVec(rou_vec.size());
        Eigen::VectorXd yVec(rou_vec.size());

        for(size_t i=0; i<rou_vec.size(); ++i) {
            xVec(i) = std::atan2(-z_vec.at(i), rou_vec.at(i));
            yVec(i) = rou_vec.at(i);
        }
        const int poly_fit_order = 4;
        Eigen::VectorXd inv_poly_coeff = polyfit(xVec, yVec, poly_fit_order);
        
        for(int i=0; i<=poly_fit_order; ++i) {
            params.inv_poly(i) = inv_poly_coeff(i);
        }
    }

    setParameters(params);

    std::cout << "initial params:\n" << params << std::endl; 
}


void
OCAMCamera::liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    liftProjective(p, P);
    P.normalize();
}


void
OCAMCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    Eigen::Vector2d xc(p[0] - mParameters.center_x(), p[1] - mParameters.center_y());
    Eigen::Vector2d xc_a(
        m_inv_scale * (xc[0] - mParameters.D() * xc[1]),
        m_inv_scale * (-mParameters.E() * xc[0] + mParameters.C() * xc[1])
    );

    double phi = std::sqrt(xc_a[0] * xc_a[0] + xc_a[1] * xc_a[1]);
    double phi_i = 1.0;
    double z = 0.0;

    for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++)
    {
        z += phi_i * mParameters.poly(i);
        phi_i *= phi;
    }

    P << xc[0], xc[1], -z;
}



void
OCAMCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const
{
    double norm = std::sqrt(P[0] * P[0] + P[1] * P[1]);
    double theta = std::atan2(-P[2], norm);
    double rho = 0.0;
    double theta_i = 1.0;

    for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
    {
        rho += theta_i * mParameters.inv_poly(i);
        theta_i *= theta;
    }

    double invNorm = 1.0 / norm;
    Eigen::Vector2d xn(
        P[0] * invNorm * rho,
        P[1] * invNorm * rho
    );

    p << xn[0] * mParameters.C() + xn[1] * mParameters.D() + mParameters.center_x(),
         xn[0] * mParameters.E() + xn[1]                   + mParameters.center_y();
}



void
OCAMCamera::undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const
{
    Eigen::Vector3d P(p_u[0], p_u[1], 1.0);
    spaceToPlane(P, p);
}


#if 0
void
OCAMCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale) const
{
    cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            double xi = mParameters.xi();
            double d2 = mx_u * mx_u + my_u * my_u;

            Eigen::Vector3d P;
            P << mx_u, my_u, 1.0 - xi * (d2 + 1.0) / (xi + sqrt(1.0 + (1.0 - xi * xi) * d2));

            Eigen::Vector2d p;
            spaceToPlane(P, p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}
#endif

cv::Mat
OCAMCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx, float fy,
                                    cv::Size imageSize,
                                    float cx, float cy,
                                    cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
    {
        imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f K_rect;

    K_rect << fx, 0, cx < 0 ? imageSize.width / 2 : cx,
              0, fy, cy < 0 ? imageSize.height / 2 : cy,
              0, 0, 1;

    if (fx < 0 || fy < 0)
    {
        throw std::string(std::string(__FUNCTION__) + ": Focal length must be specified");
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

int
OCAMCamera::parameterCount(void) const
{
    return SCARAMUZZA_CAMERA_NUM_PARAMS;
}

const OCAMCamera::Parameters&
OCAMCamera::getParameters(void) const
{
    return mParameters;
}

void
OCAMCamera::setParameters(const OCAMCamera::Parameters& parameters)
{
    mParameters = parameters;

    m_inv_scale = 1.0 / (parameters.C() - parameters.D() * parameters.E());
}

void
OCAMCamera::readParameters(const std::vector<double>& parameterVec)
{
    if ((int)parameterVec.size() != parameterCount())
    {
        return;
    }

    Parameters params = getParameters();

    params.C() = parameterVec.at(0);
    params.D() = parameterVec.at(1);
    params.E() = parameterVec.at(2);
    params.center_x() = parameterVec.at(3);
    params.center_y() = parameterVec.at(4);
    for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
        params.poly(i) = parameterVec.at(5+i);
    for (int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
        params.inv_poly(i) = parameterVec.at(5 + SCARAMUZZA_POLY_SIZE + i);

    setParameters(params);
}

void
OCAMCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());
    parameterVec.at(0) = mParameters.C();
    parameterVec.at(1) = mParameters.D();
    parameterVec.at(2) = mParameters.E();
    parameterVec.at(3) = mParameters.center_x();
    parameterVec.at(4) = mParameters.center_y();
    for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
        parameterVec.at(5+i) = mParameters.poly(i);
    for (int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
        parameterVec.at(5 + SCARAMUZZA_POLY_SIZE + i) = mParameters.inv_poly(i);
}

void
OCAMCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
OCAMCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

}
