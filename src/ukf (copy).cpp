#define SMALLVAL 0.001
#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

   // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // initial timestamp
  time_us_ = 0;

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);


  //set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }



}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float r = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float r_dot = meas_package.raw_measurements_[2];

      // The direction of the car is always x, so we switched
      // x and y from the Extended Kalman Filter's version.
      float px = r * cos(phi);
      float py = r * sin(phi);
      x_ << px, py, 0, 0, 0;

    }
    else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      // The direction of the car is always x, so we switched
      // x and y from the Extended Kalman Filter's version.
      float x = meas_package.raw_measurements_[0];
      float y = meas_package.raw_measurements_[1];
      x_ << y, x, 0, 0, 0;
    }
    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  Prediction(dt);


  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Convert radar data to vector accepted by UKF.

    UpdateRadar(meas_package);
  }
  else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(delta_t, &Xsig_aug, &Xsig_pred_);
  PredictMeanAndCovariance(&x_, &P_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd z = VectorXd(2);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1];
  z_ = z;
  PredictLidarMeasurement();
  UpdateLaserState();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];
  z_ = z;

  PredictRadarMeasurement();
  UpdateRadarState();
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = n_x_;

  //set augmented dimension
  int n_aug = n_aug_;

  //Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = std_a_;

  //Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = std_yawdd_;

  //define spreading parameter
  double lambda = lambda_;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  VectorXd x = x_;
  MatrixXd P = P_;

  // -- TESTING --

  // //set example state
  // VectorXd x = VectorXd(n_x);
  // x <<   5.7441,
  //        1.3800,
  //        2.2049,
  //        0.5015,
  //        0.3528;

  // //create example covariance matrix
  // MatrixXd P = MatrixXd(n_x, n_x);
  // P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
  //         -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
  //          0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
  //         -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
  //         -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  // -- TESTING END --


  //create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P;
  P_aug(5,5) = std_a*std_a;
  P_aug(6,6) = std_yawdd*std_yawdd;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda+n_aug) * L.col(i);
    Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda+n_aug) * L.col(i);
  }

  //print result
  cout << "Xsig_aug = " << endl << Xsig_aug << endl;

  *Xsig_out = Xsig_aug;

/* expected result:
   Xsig_aug =
  5.7441  5.85768   5.7441   5.7441   5.7441   5.7441   5.7441   5.7441  5.63052   5.7441   5.7441   5.7441   5.7441   5.7441   5.7441
    1.38  1.34566  1.52806     1.38     1.38     1.38     1.38     1.38  1.41434  1.23194     1.38     1.38     1.38     1.38     1.38
  2.2049  2.28414  2.24557  2.29582   2.2049   2.2049   2.2049   2.2049  2.12566  2.16423  2.11398   2.2049   2.2049   2.2049   2.2049
  0.5015  0.44339 0.631886 0.516923 0.595227   0.5015   0.5015   0.5015  0.55961 0.371114 0.486077 0.407773   0.5015   0.5015   0.5015
  0.3528 0.299973 0.462123 0.376339  0.48417 0.418721   0.3528   0.3528 0.405627 0.243477 0.329261  0.22143 0.286879   0.3528   0.3528
       0        0        0        0        0        0  0.34641        0        0        0        0        0        0 -0.34641        0
       0        0        0        0        0        0        0  0.34641        0        0        0        0        0        0 -0.34641
*/

}

void UKF::SigmaPointPrediction(double delta_t, MatrixXd* Xsig_aug_pt, MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = n_x_;

  //set augmented dimension
  int n_aug = n_aug_;

  //prediction result
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  MatrixXd Xsig_aug = *Xsig_aug_pt;

  // -- TESTING --
  // delta_t = 0.1;

  // //create example sigma point matrix
  // MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  //    Xsig_aug <<
  //   5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
  //     1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
  //   2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
  //   0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
  //   0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
  //        0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
  //        0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
  // -- TESTING END --

  //predict sigma points
  for (int i = 0; i< 2*n_aug+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    // if (yaw < SMALLVAL) {
    //   yaw = SMALLVAL;
    // }
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > SMALLVAL) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  *Xsig_out = Xsig_pred;

  // expected result:
  // Xsig_pred =

  // 5.93553 6.06251 5.92217 5.9415 5.92361 5.93516 5.93705 5.93553 5.80832 5.94481 5.92935 5.94553 5.93589 5.93401 5.93553

  // 1.48939 1.44673 1.66484 1.49719 1.508 1.49001 1.49022 1.48939 1.5308 1.31287 1.48182 1.46967 1.48876 1.48855 1.48939

  // 2.2049 2.28414 2.24557 2.29582 2.2049 2.2049 2.23954 2.2049 2.12566 2.16423 2.11398 2.2049 2.2049 2.17026 2.2049

  // 0.53678 0.473387 0.678098 0.554557 0.643644 0.543372 0.53678 0.538512 0.600173 0.395462 0.519003 0.429916 0.530188 0.53678 0.535048

  // 0.3528 0.299973 0.462123 0.376339 0.48417 0.418721 0.3528 0.387441 0.405627 0.243477 0.329261 0.22143 0.286879 0.3528 0.318159
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

  //set state dimension
  int n_x = n_x_;

  //set augmented dimension
  int n_aug = n_aug_;

  //define spreading parameter
  double lambda = lambda_;

  MatrixXd Xsig_pred = Xsig_pred_;
  // cout << Xsig_pred.col(0) << endl;
  // cout << "cols: " << Xsig_pred.cols() << endl;
  // cout << "rows: " << Xsig_pred.rows() << endl;

  // -- TESTING --
  //create example matrix with predicted sigma points
  // MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  // Xsig_pred <<
  //        5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
  //          1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
  //         2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
  //        0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
  //         0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  // -- TESTING END --

  //create vector for weights
  VectorXd weights = weights_;
  
  //create vector for predicted state
  VectorXd x = VectorXd(n_x);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  //iterate over sigma points
    x = x+ weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    x_diff(3) = NormalizeAngle(x_diff(3));

    P = P + weights(i) * x_diff * x_diff.transpose() ;
  }

  //print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P << std::endl;

  //write result
  *x_out = x;
  *P_out = P;

  // expected result x:
  //  x =
  // 5.93637

  // 1.49035

  // 2.20528

  // 0.536853

  // 0.353577

  // expected result p:
  //  P =
  // 0.00543425 -0.0024053 0.00341576 -0.00348196 -0.00299378

  // -0.0024053 0.010845 0.0014923 0.00980182 0.00791091

  // 0.00341576 0.0014923 0.00580129 0.000778632 0.000792973

  // -0.00348196 0.00980182 0.000778632 0.0119238 0.0112491

  // -0.00299378 0.00791091 0.000792973 0.0112491 0.0126972
}

void UKF::PredictRadarMeasurement() {
  //raw inputs
  VectorXd& z = z_;

  //set state dimension
  int n_x = n_x_;

  //set augmented dimension
  int n_aug = n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  double lambda = lambda_;

  //set vector for weights
  VectorXd& weights = weights_;

  //radar measurement noise standard deviation radius in m
  double std_radr = std_radr_;

  //radar measurement noise standard deviation angle in rad
  double std_radphi = std_radphi_;

  //radar measurement noise standard deviation radius change in m/s
  double std_radrd = std_radrd_;

  MatrixXd& Xsig_pred = Xsig_pred_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  //create example vector for predicted state mean
  VectorXd& x = x_;

  //create example matrix for predicted state covariance
  MatrixXd& P = P_;

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);
    // if (yaw < SMALLVAL) {
    //   yaw = SMALLVAL;
    // }
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    double denom = sqrt(p_x*p_x + p_y*p_y);
    if (fabs(denom) < SMALLVAL) {
      denom = SMALLVAL;
    }
    Zsig(0,i) = denom;                        //r
    Zsig(1,i) = atan2(p_y,p_x);               //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / denom;   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr*std_radr, 0, 0,
          0, std_radphi*std_radphi, 0,
          0, 0,std_radrd*std_radrd;
  S = S + R;

  //print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;
  
  z_pred_ = z_pred;
  S_ = S;
  Zsig_ = Zsig;
  z_ = z;

}

void UKF::UpdateRadarState() {
  //set state dimension
  int& n_x = n_x_;

  //set augmented dimension
  int& n_aug = n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //set vector for weights
  VectorXd& weights = weights_;


  MatrixXd& Xsig_pred = Xsig_pred_;
  VectorXd& x = x_;
  MatrixXd& P = P_;
  MatrixXd& Zsig = Zsig_;
  VectorXd& z_pred = z_pred_;
  MatrixXd& S = S_;
  VectorXd& z = z_;

  // -- TESTING --

 //  //create example matrix with predicted sigma points
 //  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
 //  Xsig_pred <<
 //         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
 //           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
 //          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
 //         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
 //          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

 //  //create example vector for predicted state mean
 //  VectorXd x = VectorXd(n_x);
 //  x <<
 //     5.93637,
 //     1.49035,
 //     2.20528,
 //    0.536853,
 //    0.353577;

 //  //create example matrix for predicted state covariance
 //  MatrixXd P = MatrixXd(n_x,n_x);
 //  P <<
 //  0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
 //  -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
 //  0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
 // -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
 // -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;

 //  //create example matrix with sigma points in measurement space
 //  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
 //  Zsig <<
 //      6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
 //     0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
 //      2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

 //  //create example vector for mean predicted measurement
 //  VectorXd z_pred = VectorXd(n_z);
 //  z_pred <<
 //      6.12155,
 //     0.245993,
 //      2.10313;

 //  //create example matrix for predicted measurement covariance
 //  MatrixXd S = MatrixXd(n_z,n_z);
 //  S <<
 //      0.0946171, -0.000139448,   0.00407016,
 //   -0.000139448,  0.000617548, -0.000770652,
 //     0.00407016, -0.000770652,    0.0180917;

 //  //create example vector for incoming radar measurement
 //  VectorXd z = VectorXd(n_z);
 //  z <<
 //      5.9214,
 //      0.2187,
 //      2.0062;

  // -- TESTING END --

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();


  //write result
  x_ = x;
  P_ = P;

  //print result
  std::cout << "Updated state x: " << std::endl << x << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

  VectorXd NIS = z_diff.transpose() * Si * z_diff;
  NIS_radar_ = NIS(0);
  /* expected result x:
    x =
   5.92276
   1.41823
   2.15593
  0.489274
  0.321338
      */

    /* expected result P:
       P =
    0.00361579 -0.000357881   0.00208316 -0.000937196  -0.00071727
  -0.000357881   0.00539867   0.00156846   0.00455342   0.00358885
    0.00208316   0.00156846   0.00410651   0.00160333   0.00171811
  -0.000937196   0.00455342   0.00160333   0.00652634   0.00669436
   -0.00071719   0.00358884   0.00171811   0.00669426   0.00881797
    */
}

void UKF::PredictLidarMeasurement() {
  //raw inputs
  VectorXd z = z_;

  //set state dimension
  int n_x = n_x_;

  //set augmented dimension
  int n_aug = n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //define spreading parameter
  double lambda = lambda_;

  //set vector for weights
  VectorXd weights = weights_;

  //laser x measurement noise standard deviation radius in m
  double std_laspx = std_laspx_;

  //laser y measurement noise standard deviation radius in m
  double std_laspy = std_laspy_;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = Xsig_pred_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  //create example vector for predicted state mean
  VectorXd x = x_;

  //create example matrix for predicted state covariance
  MatrixXd P = P_;

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points

    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx*std_laspx, 0,
          0, std_laspy*std_laspy;
  S = S + R;

  //print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  z_pred_ = z_pred;
  S_ = S;
  Zsig_ = Zsig;
  z_ = z;
}

void UKF::UpdateLaserState() {
  //set state dimension
  int& n_x = n_x_;

  //set augmented dimension
  int& n_aug = n_aug_;

  //set measurement dimension, leser can measure px and py
  int n_z = 2;

  //set vector for weights
  VectorXd& weights = weights_;


  MatrixXd& Xsig_pred = Xsig_pred_;
  VectorXd& x = x_;
  MatrixXd& P = P_;
  MatrixXd& Zsig = Zsig_;
  VectorXd& z_pred = z_pred_;
  MatrixXd& S = S_;
  VectorXd& z = z_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 sigma points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd Si = S_.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();

  //print result
  std::cout << "Updated state x: " << std::endl << x << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

  //write result
  x_ = x;
  P_ = P;

  VectorXd NIS = z_diff.transpose() * Si * z_diff;
  NIS_laser_ = NIS(0);
}

double UKF::NormalizeAngle(double angle) {
  double new_angle = fmod(angle+M_PI, 2 * M_PI);
  if (new_angle < 0) {
    new_angle += M_PI;
  }
  return new_angle-M_PI;
}