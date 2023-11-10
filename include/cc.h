#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    
    void computeSlow();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    //////////////////////////////////////////// Donghyeon RL /////////////////////////////////////////
    void loadNetwork();
    void processNoise();
    void processObservation();
    void processObservation2();
    void feedforwardPolicy();
    void initVariable();
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);

    static const int num_action = 13;
    static const int num_actuator_action = 12;
    static const int num_cur_state = 44;
    static const int num_cur_internal_state = 31;
    static const int num_state_skip = 2;
    static const int num_state_hist = 5;
    static const int num_state = num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1); //rui 207
    static const int num_hidden = 256;
    

    Eigen::MatrixXd policy_net_w0_;
    Eigen::MatrixXd policy_net_b0_;
    Eigen::MatrixXd policy_net_w2_;
    Eigen::MatrixXd policy_net_b2_;
    Eigen::MatrixXd action_net_w_;
    Eigen::MatrixXd action_net_b_;
    Eigen::MatrixXd hidden_layer1_;
    Eigen::MatrixXd hidden_layer2_;
    Eigen::MatrixXd rl_action_;

    Eigen::MatrixXd value_net_w0_;
    Eigen::MatrixXd value_net_b0_;
    Eigen::MatrixXd value_net_w2_;
    Eigen::MatrixXd value_net_b2_;
    Eigen::MatrixXd value_net_w_;
    Eigen::MatrixXd value_net_b_;
    Eigen::MatrixXd value_hidden_layer1_;
    Eigen::MatrixXd value_hidden_layer2_;
    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    Eigen::MatrixXd state_;
    Eigen::MatrixXd state_cur_;
    Eigen::MatrixXd state_buffer_;
    Eigen::MatrixXd state_buffer_2000_; // rui
    Eigen::MatrixXd state_temp_; // rui
    Eigen::MatrixXd state_mean_;
    Eigen::MatrixXd state_var_;
    Eigen::MatrixXd state_2000Hz_;
    Eigen::MatrixXd observation_buffer_;
    Eigen::MatrixXd action_buffer_2000_; // rui

    std::ofstream writeFile;

    float phase_ = 0.0;
    float phase_2 = 0.0;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_observation_pre_ = 0.0; //rui testing
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;
    double action_dt_accumulate_2 = 0.0; //rui testing

    Eigen::Vector3d euler_angle_;

    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;

    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;

    float freq_scaler_ = 1/40;
    float freq_tester_2000HZ = 1/2000.0;
    int action_delay = 1;
    int observation_delay = 1;
    int frameskip_custom = 50;//rui frameskip 250Hz -> 8, 200Hz -> 10, 150Hz -> 13, 125Hz -> 16, 100Hz -> 20, 62.5Hz -> 32, 50Hz -> 40, 40Hz -> 50 size
    bool just_after_init = true;
    int action_buffer_length = 0;
    Eigen::MatrixXd rl_action_2000_; //rui


private:
    Eigen::VectorQd ControlVal_;
};