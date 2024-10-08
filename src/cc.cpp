#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();
    

//? yaml getparam{ 
    if (is_on_robot_) {
        ros::param::get("/tocabi_controller/data_path_real", data_path_);
    }
    else{
        ros::param::get("/tocabi_controller/data_path_sim", data_path_);
    }
    std::cout << "data_path : " << data_path_ << std::endl;
    ros::param::get("/tocabi_controller/target_vel_x", target_vel_x_yaml_); 
    std::cout << "target_vel_x : " << target_vel_x_yaml_ << std::endl;
    ros::param::get("/tocabi_controller/target_vel_y", target_vel_y_yaml_); 
    std::cout << "target_vel_y : " << target_vel_y_yaml_ << std::endl;
    ros::param::get("/tocabi_controller/delay_action", action_delay_); 
    std::cout << "action_delay : " << action_delay_ << std::endl;
    ros::param::get("/tocabi_controller/delay_observation", observation_delay_); 
    std::cout << "obs_delay : " << observation_delay_ << std::endl;
    ros::param::get("/tocabi_controller/frameskip", frameskip_); 
    std::cout << "frameskip : " << frameskip_ << std::endl;
    ros::param::get("/tocabi_controller/freq_scaler", freq_scaler_); 
    std::cout << "freq_scaler : " << freq_scaler_ << std::endl;
    
    action_buffer_length = 0;

//? yaml }

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open(data_path_, std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open(data_path_, std::ofstream::out | std::ofstream::app);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadNetwork();

    // joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this); //rui if joy callback exist uncomment this
    // xbox_joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &CustomController::xBoxJoyCallback, this);
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();

    // nh_.getParam("/weight/AMP", weight_dir_);
    string cur_path = "/home/dyros/tocabi_ws/src/tocabi_cc/weight/AMP/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/weight/AMP/";
    }
    std::ifstream file[22];

    file[0].open(cur_path+"a2c_network_actor_mlp_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"a2c_network_actor_mlp_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"a2c_network_actor_mlp_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"a2c_network_actor_mlp_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"a2c_network_mu_weight.txt", std::ios::in);
    file[5].open(cur_path+"a2c_network_mu_bias.txt", std::ios::in);
    file[6].open(cur_path+"running_mean_std_running_mean.txt", std::ios::in);
    file[7].open(cur_path+"running_mean_std_running_var.txt", std::ios::in);
    file[8].open(cur_path+"a2c_network_critic_mlp_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"a2c_network_critic_mlp_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"a2c_network_critic_mlp_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"a2c_network_critic_mlp_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"a2c_network_value_weight.txt", std::ios::in);
    file[13].open(cur_path+"a2c_network_value_bias.txt", std::ios::in);

    file[14].open(cur_path+"a2c_network__disc_mlp_0_weight.txt", std::ios::in);
    file[15].open(cur_path+"a2c_network__disc_mlp_0_bias.txt", std::ios::in);
    file[16].open(cur_path+"a2c_network__disc_mlp_2_weight.txt", std::ios::in);
    file[17].open(cur_path+"a2c_network__disc_mlp_2_bias.txt", std::ios::in);
    file[18].open(cur_path+"a2c_network__disc_logits_weight.txt", std::ios::in);
    file[19].open(cur_path+"a2c_network__disc_logits_bias.txt", std::ios::in);
    file[20].open(cur_path+"amp_running_mean_std_running_mean.txt", std::ios::in);
    file[21].open(cur_path+"amp_running_mean_std_running_var.txt", std::ios::in);

    for (int i = 0; i < 22; i++)
    {
        if(!file[i].is_open())
        {
            std::cout<<"Can not find the weight file "<< i <<std::endl;
        }
    }
    
    float temp;
    auto loadMatrix = [&](std::ifstream& file, Eigen::MatrixXd& matrix) {
        int row = 0, col = 0;
        while (!file.eof() && row != matrix.rows()) {
            file >> temp;
            if (file.fail()) break; // Ensure we don't read past the end of file
            matrix(row, col) = temp;
            col++;
            if (col == matrix.cols()) {
                col = 0;
                row++;
            }
        }
    };
    loadMatrix(file[0], policy_net_w0_);
    loadMatrix(file[1], policy_net_b0_);
    loadMatrix(file[2], policy_net_w2_);
    loadMatrix(file[3], policy_net_b2_);
    loadMatrix(file[4], action_net_w_);
    loadMatrix(file[5], action_net_b_);
    loadMatrix(file[6], state_mean_);
    loadMatrix(file[7], state_var_);
    loadMatrix(file[8], value_net_w0_);
    loadMatrix(file[9], value_net_b0_);
    loadMatrix(file[10], value_net_w2_);
    loadMatrix(file[11], value_net_b2_);
    loadMatrix(file[12], value_net_w_);
    loadMatrix(file[13], value_net_b_);  

    loadMatrix(file[14], disc_net_w0_);
    loadMatrix(file[15], disc_net_b0_);
    loadMatrix(file[16], disc_net_w2_);
    loadMatrix(file[17], disc_net_b2_);
    loadMatrix(file[18], disc_net_w_);
    loadMatrix(file[19], disc_net_b_);
    loadMatrix(file[20], disc_state_mean_);
    loadMatrix(file[21], disc_state_var_);    

}

void CustomController::initVariable()
{    
    policy_net_w0_.resize(num_hidden1, num_state);
    policy_net_b0_.resize(num_hidden1, 1);
    policy_net_w2_.resize(num_hidden2, num_hidden1);
    policy_net_b2_.resize(num_hidden2, 1);
    action_net_w_.resize(num_action, num_hidden2);
    action_net_b_.resize(num_action, 1);

    hidden_layer1_.resize(num_hidden1, 1);
    hidden_layer2_.resize(num_hidden2, 1);
    rl_action_.resize(num_action, 1);

    value_net_w0_.resize(num_hidden1, num_state);
    value_net_b0_.resize(num_hidden1, 1);
    value_net_w2_.resize(num_hidden2, num_hidden1);
    value_net_b2_.resize(num_hidden2, 1);
    value_net_w_.resize(1, num_hidden2);
    value_net_b_.resize(1, 1);

    value_hidden_layer1_.resize(num_hidden1, 1);
    value_hidden_layer2_.resize(num_hidden2, 1);
    
    state_cur_.resize(num_cur_state, 1);
    state_.resize(num_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_mean_.resize(num_state, 1);
    state_var_.resize(num_state, 1);

    // Discriminator Network
    disc_net_w0_.resize(num_disc_hidden1, num_disc_state);
    disc_net_b0_.resize(num_disc_hidden1, 1);
    disc_net_w2_.resize(num_disc_hidden2, num_disc_hidden1);
    disc_net_b2_.resize(num_disc_hidden2, 1);    
    disc_net_w_.resize(1, num_disc_hidden2);
    disc_net_b_.resize(1, 1);

    disc_hidden_layer1_.resize(num_disc_hidden1, 1);
    disc_hidden_layer2_.resize(num_disc_hidden2, 1);

    disc_state_buffer_.resize(num_disc_cur_state*2, 1);
    disc_state_cur_.resize(num_disc_cur_state, 1);
    disc_state_.resize(num_disc_state, 1);
    disc_state_mean_.resize(num_disc_state, 1);
    disc_state_var_.resize(num_disc_state, 1);

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
                    
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;

    kp_.setZero();
    kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    // kp_.diagonal() /= 9.0;
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    // kv_.diagonal() /= 3.0;
}

Eigen::Vector3d CustomController::mat2euler(Eigen::Matrix3d mat)
{
    Eigen::Vector3d euler;

    double cy = std::sqrt(mat(2, 2) * mat(2, 2) + mat(1, 2) * mat(1, 2));
    if (cy > std::numeric_limits<double>::epsilon())
    {
        euler(2) = -atan2(mat(0, 1), mat(0, 0));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = -atan2(mat(1, 2), mat(2, 2));
    }
    else
    {
        euler(2) = -atan2(-mat(1, 0), mat(1, 1));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = 0.0;
    }
    return euler;
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    if (is_on_robot_)
    {
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_dot_lpf_ = q_dot_lpf_;
        }
    }
    else
    {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        }
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_vel_noise_ = q_vel_noise_;
            q_dot_lpf_ = q_dot_lpf_;
        }
        q_noise_pre_ = q_noise_;
    }
    time_pre_ = time_cur_;
}

void CustomController::processObservation()
{
    /*
    **obs 
    **  1) root_h: root height (z)                  (1)     0
    **  3) root_rot: root rotation                  (3)     2:8
    **  4) root_vel: root linear velocity           (3)     8:11
    **  5) root_ang_vel: root angular velocity      (3)     11:14
    **  6) commands: x, y, yaw                      (3)     14:17
    **  7) dof_pos: dof position                    (12)    17:29
    **  8) dof_vel: dof velocity                    (12)    29:41
    **  10) action: action                           (12)    47:59
    */
    int data_idx = 0;

    //** 1) root_h: root height (z)                  (1)     0 
    //** state_cur_(data_idx) = rd_cc_.q_virtual_(2);
    //** data_idx++;

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    
    //** 3) root_rot: root rotation                  (3)     2:8
    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    state_cur_(data_idx) = euler_angle_(0); //rui 1
    data_idx++;

    state_cur_(data_idx) = euler_angle_(1); //rui 1
    data_idx++;

    state_cur_(data_idx) = euler_angle_(2); //rui 1
    data_idx++;

    //** 4) root_vel: root linear velocity           (3)     8:11
    //** 5) root_ang_vel: root angular velocity      (3)     11:14    
    
    local_lin_vel_ = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0,3));
    for (int i=0; i<3; i++)
    {
        state_cur_(data_idx) = local_lin_vel_(i);
        data_idx++;
    }    
    for (int i=0; i<3; i++)
    {
        state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i+3);
        data_idx++;
    }

    //** 6) commands: x, y, yaw                      (3)     14:17
    if (rd_cc_.control_time_us_ < start_time_ + 10e6) {
        desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 5e6, 0.0, target_vel_x_yaml_, 0.0, 0.0);
        // desired_vel_yaw = 0.0;
    }
    else if (rd_cc_.control_time_us_ < start_time_ + 20e6) {
        // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 14e6, start_time_ + 19e6, 0.3, 0.0, 0.0, 0.0);
        desired_vel_x = target_vel_x_yaml_;
        // desired_vel_yaw = 0.0;
    }
    else if (rd_cc_.control_time_us_ < start_time_ + 30e6) {
        desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 24e6, start_time_ + 29e6, target_vel_x_yaml_, 0.0, 0.0, 0.0);
        // desired_vel_yaw = 0.0;
    }
    // else if (rd_cc_.control_time_us_ < start_time_ + 40e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 30e6, start_time_ + 35e6, 0.0, 0.5, 0.0, 0.0);
    //     // desired_vel_yaw = 0.0;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 50e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 44e6, start_time_ + 49e6, 0.5, 0.0, 0.0, 0.0);
    //     // desired_vel_yaw = -0.4;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 60e6) {
    //     // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 15e6, start_time_ + 20e6, 0.0, 0.5, 0.0, 0.0);
    //     desired_vel_x = 0.0;
    //     // desired_vel_yaw = 0.4;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 70e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 60e6, start_time_ + 65e6, 0.0, 0.7, 0.0, 0.0);
    //     // desired_vel_yaw = -0.4;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 80e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 74e6, start_time_ + 79e6, 0.7, 0.0, 0.0, 0.0);
    //     // desired_vel_yaw = 0.4;
    // }
    else {
        desired_vel_x = 0.0;
        // desired_vel_yaw = 0.0;
    }

    // if (rd_cc_.control_time_us_ < start_time_ + 20e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 20e6, 0.0, 0.8, 0.8/20e6, 0.8/20e6);
    //     desired_vel_yaw = 0.0;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 48e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_+20e6, start_time_ + 48e6, 0.8, -0.4, 1.2/28e6, 1.2/28e6);
    //     desired_vel_yaw = 0.0;
    // }
    

    // state_cur_(data_idx) = desired_vel_x;
    // data_idx++;
    // state_cur_(data_idx) = 0.0;
    // data_idx++;
    // state_cur_(data_idx) = desired_vel_yaw;
    // data_idx++;

    // state_cur_(data_idx) = target_vel_x_yaml_;//?

    // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 20e6, 0.0, 1.5, 1.5/20e6, 1.5/20e6);
    // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 20e6, 0.0, 1.4, 1.4/20e6, 1.4/20e6);
    // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 30e6, 0.0, 1.5, 1.5/30e6, 1.5/30e6);
    state_cur_(data_idx) = desired_vel_x;
    data_idx++;
    // state_cur_(data_idx) = target_vel_y_;
    state_cur_(data_idx) = 0.0;
    data_idx++;
    // state_cur_(data_idx) = target_vel_yaw_;
    state_cur_(data_idx) = 0.0;
    // state_cur_(data_idx) = 0.0;
    data_idx++;

    //** 7) dof_pos: dof position                    (12)    17:29
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_(data_idx) = q_noise_(i); //rui 12
        data_idx++; 
    }
    //** 8) dof_vel: dof velocity                    (12)    29:41
    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rui 12
        }
        else
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6);
        }
        data_idx++;
    }
    
    // float squat_duration = 1.7995;
    // phase_ = std::fmod((rd_cc_.control_time_us_-start_time_)/1e6 + action_dt_accumulate_, squat_duration) / squat_duration;

    // state_cur_(data_idx) = sin(2*M_PI*phase_); //rui 1
    // data_idx++;
    // state_cur_(data_idx) = cos(2*M_PI*phase_); //rui 1
    // data_idx++;
    
    // state_cur_(data_idx) = target_vel_x_yaml_;//target_vel_x_; //rui 1
    // data_idx++;

    // state_cur_(data_idx) = target_vel_y_yaml_;//target_vel_y_; //rui 1
    // data_idx++;

    // for (int i=0; i<6; i++)
    // {
    //     state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i); //rui 6 base_lin_vel base_ang_vel
    //     data_idx++;
    // }

    // // state_cur_(data_idx) = -rd_cc_.LF_FT(2);
    // // data_idx++;

    // // state_cur_(data_idx) = -rd_cc_.RF_FT(2);
    // // data_idx++;

    // // state_cur_(data_idx) = rd_cc_.LF_FT(3);
    // // data_idx++;

    // // state_cur_(data_idx) = rd_cc_.RF_FT(3);
    // // data_idx++;

    // // state_cur_(data_idx) = rd_cc_.LF_FT(4);
    // // data_idx++;

    // // state_cur_(data_idx) = rd_cc_.RF_FT(4);
    // // data_idx++;

    //** 10) action: action                           (12)    47:59
    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);  //rui 12
        data_idx++;
    }
    // state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(num_actuator_action), 0.0, 1.0); //rui 1(phase)
    // data_idx++;
    
//? orig {
    // state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1);
    // state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();

    // // Internal State First
    // for (int i = 0; i < num_state_hist; i++)
    // {
    //     state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1);
    // }
    // // Action History Second
    // for (int i = 0; i < num_state_hist-1; i++)
    // {
    //     state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)) + num_cur_internal_state, 0, num_action, 1);
    // }
//? orig }
    state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1);
    state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = state_cur_;

    // Internal State First
    for (int i = 0; i < num_state_hist; i++)
    {
        state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1);
    }
    // Action History Second
    for (int i = 0; i < num_state_hist-1; i++)
    {
        state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)) + num_cur_internal_state, 0, num_action, 1);
    }

    // Normalization of State
    // state_ = (state_ - state_mean_).array() / state_var_.cwiseSqrt().array();
    state_ = (state_ - state_mean_).array() / (state_var_.array() + 1e-05).sqrt();

}

void CustomController::processDiscriminator()
{
    /*
    **1) root_h
    **2) base euler
    **3) q pos
    **4) q vel
    **5) local key pos
    */
    int disc_data_idx = 0;

    // 1) root_h
    disc_state_cur_(disc_data_idx) = rd_cc_.q_virtual_(2);
    disc_data_idx++;

    // 2) base euler
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);
    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    for (int i=0; i<3; i++)
    {
        disc_state_cur_(disc_data_idx) = euler_angle_(i);
        disc_data_idx++;
    }

    // 3) q pos
    for (int i = 0; i < 12; i++)
    {
        disc_state_cur_(disc_data_idx) = q_noise_(i);
        disc_data_idx++;
    }    

    // 4) q vel
    for (int i = 0; i < 12; i++)
    {
        disc_state_cur_(disc_data_idx) = q_vel_noise_(i);
        disc_data_idx++;
    }

    // 5) local key pos
    Vector3d global_lfoot_pos = rd_cc_.link_[Left_Foot].xpos;
    Vector3d global_rfoot_pos = rd_cc_.link_[Right_Foot].xpos;

    Vector3d local_lfoot_pos = quatRotateInverse(q, global_lfoot_pos - rd_cc_.q_virtual_.head(3));
    Vector3d local_rfoot_pos = quatRotateInverse(q, global_rfoot_pos - rd_cc_.q_virtual_.head(3));

    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = local_lfoot_pos(i);
        disc_data_idx++;
    }
    for (int i = 0; i < 3; i++)
    {
        disc_state_cur_(disc_data_idx) = local_rfoot_pos(i);
        disc_data_idx++;
    }

    disc_state_buffer_.block(num_disc_cur_state, 0, num_disc_cur_state,1) = disc_state_buffer_.block(0, 0, num_disc_cur_state,1); 
    disc_state_buffer_.block(0, 0, num_disc_cur_state,1) = disc_state_cur_;

    disc_state_ = (disc_state_buffer_ - disc_state_mean_).array() / (disc_state_var_.array() + 1e-05).sqrt();   
}

void CustomController::feedforwardPolicy()
{
    hidden_layer1_ = policy_net_w0_ * state_ + policy_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (hidden_layer1_(i) < 0)
            hidden_layer1_(i) = 0.0;
    }

    hidden_layer2_ = policy_net_w2_ * hidden_layer1_ + policy_net_b2_;
    for (int i = 0; i < num_hidden2; i++) 
    {
        if (hidden_layer2_(i) < 0)
            hidden_layer2_(i) = 0.0;
    }

    rl_action_ = action_net_w_ * hidden_layer2_ + action_net_b_;

    value_hidden_layer1_ = value_net_w0_ * state_ + value_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (value_hidden_layer1_(i) < 0)
            value_hidden_layer1_(i) = 0.0;
    }

    value_hidden_layer2_ = value_net_w2_ * value_hidden_layer1_ + value_net_b2_;
    for (int i = 0; i < num_hidden2; i++) 
    {
        if (value_hidden_layer2_(i) < 0)
            value_hidden_layer2_(i) = 0.0;
    }

    value_ = (value_net_w_ * value_hidden_layer2_ + value_net_b_)(0);
    
    // Discriminator
    disc_hidden_layer1_ = disc_net_w0_ * disc_state_ + disc_net_b0_;
    for (int i = 0; i < num_disc_hidden1; i++) 
    {
        if (disc_hidden_layer1_(i) < 0)
            disc_hidden_layer1_(i) = 0.0;
    }

    disc_hidden_layer2_ = disc_net_w2_ * disc_hidden_layer1_ + disc_net_b2_;
    for (int i = 0; i < num_disc_hidden2; i++) 
    {
        if (disc_hidden_layer2_(i) < 0)
            disc_hidden_layer2_(i) = 0.0;
    }

    disc_value_ = (disc_net_w_ * disc_hidden_layer2_ + disc_net_b_)(0);
}

void CustomController::computeSlow()
{
    copyRobotData(rd_);
    if (rd_cc_.tc_.mode == 7)
    {
//SECTION - init 
        if (rd_cc_.tc_init)
        {
            //Initialize settings for Task Control! 
            start_time_ = rd_cc_.control_time_us_;
            q_noise_pre_ = q_noise_ = q_init_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1/249.9)*1e6;

            rd_.tc_init = false;
            std::cout<<"cc mode 7"<<std::endl;
            torque_init_ = rd_cc_.torque_desired;

            processNoise();
            processObservation();
            feedforwardPolicy();
            for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            {
                // state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
                state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1).setZero();
            }
            disc_state_buffer_.block(num_disc_cur_state, 0, num_disc_cur_state,1).setZero();
        }
//!SECTION - init

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/250.0 - 1/10000.0) 
//SECTION - feedforwardPolicy
        // if (policy_step >= frameskip_)
        {
            processObservation(); //? orig 
            processDiscriminator();
            feedforwardPolicy();
            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/250.0, 0.0, 5/250.0); //? orig 

            if (value_ < -10.0)
            {
                cout << "Value: " << value_ << endl;
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function" << std::endl;
                }
            }
            cout << "Value: " << value_ << " Disc: " << disc_value_ << " target x vel : " <<  desired_vel_x << " time : "<< rd_cc_.control_time_us_ << endl;
            // checkTouchDown();

            if (is_write_file_)
            {
                    // double reward = computeReward();
                    writeFile << (rd_cc_.control_time_us_ - time_inference_pre_)/1e6 << "\t";
                    writeFile << phase_ << "\t";
                    writeFile << DyrosMath::minmax_cut(rl_action_(num_action-1)*1/100.0, 0.0, 1/100.0) << "\t";
                    writeFile << rd_cc_.LF_FT.transpose() << "\t";
                    writeFile << rd_cc_.RF_FT.transpose() << "\t";
                    writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
                    writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";
                    writeFile << rd_cc_.torque_desired.transpose()  << "\t";
                    writeFile << q_noise_.transpose() << "\t";
                    writeFile << q_dot_lpf_.transpose() << "\t";
                    writeFile << rd_cc_.q_dot_virtual_.transpose() << "\t";
                    writeFile << rd_cc_.q_virtual_.transpose() << "\t";
                    writeFile << value_ << "\t" << stop_by_value_thres_ <<"\t" ;
                    // writeFile << value_ << "\t" << stop_by_value_thres_ <<"\t" << reward;
                    writeFile << target_vel_x_ << "\t";
                    writeFile << desired_vel_x << "\t";       
                    writeFile << local_lin_vel_(0) << "\t";

                    writeFile << target_vel_yaw_ << "\t";
                    writeFile << desired_vel_yaw << "\t";
                    writeFile << rd_cc_.q_dot_virtual_(5) << "\t";
                    
                    writeFile << std::endl;
                    time_write_pre_ = rd_cc_.control_time_us_;
            }
            // std::cout << policy_step << " " << rd_cc_.control_time_us_ - time_inference_pre_ << " " << value_ << " " << stop_by_value_thres_ << std::endl;
            time_inference_pre_ = rd_cc_.control_time_us_;
            policy_step = 0;
        }
//!SECTION - feedforwardPolicy
        policy_step++;
        
        action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*freq_scaler_, 0.0, freq_scaler_); 
        // time_inputTorque_pre_ = rd_cc_.control_time_us_;

//? 500Hz act delay
        // //** put action into buffer **//num_action*(num_state_skip*frameskip_custom*num_state_hist)
        // action_buffer_simfreq_.block(0, 0, num_action*(num_state_skip*frameskip_*num_state_hist-1),1) = action_buffer_simfreq_.block(num_action, 0, num_action*(num_state_skip*frameskip_*num_state_hist-1),1); //rui 0~13x(2*5*8-1) = 13~13x(2*5*8), 13개만큼 끌어오고
        // action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1), 0, num_action,1) = rl_action_; //rui 새로운 action로 채워주기
        // //** apply action delay **//
        // if( action_buffer_length  <= action_delay_){
        //     rl_action_simfreq_ = action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1 - action_buffer_length), 0, num_action, 1);
        //     action_buffer_length++;
        // }
        // else{
        //     rl_action_simfreq_ = action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1 - action_delay_), 0, num_action, 1);
        // }
//? 500Hz act delay


        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
        }
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);
        }
        
        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
            rd_.torque_desired = torque_rl_;
        }

        if (stop_by_value_thres_)
        {
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
        }


    }
    LF_CF_FT_pre = rd_cc_.LF_CF_FT;
    RF_CF_FT_pre = rd_cc_.RF_CF_FT;
}

void CustomController::computeFast()
{
    // if (tc.mode == 10)
    // {
    // }
    // else if (tc.mode == 11)
    // {
    // }
}

void CustomController::computePlanner()
{
}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::joyCallback(const tocabi_msgs::WalkingCommand::ConstPtr& joy)
{
    // target_vel_x_ = DyrosMath::minmax_cut(joy->axes[0], -0.5, 1.0);
    target_vel_x_ = DyrosMath::minmax_cut(joy->step_length_x, -0.4, 0.8);
    target_vel_y_ = 0.0; // DyrosMath::minmax_cut(joy->axes[1], -0.0, 0.0);
    target_vel_yaw_ = -DyrosMath::minmax_cut(joy->step_length_y, -0.3, 0.3);
}

void CustomController::xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1], -0.5, 1.0);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0], -0.0, 0.0);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3], -0.4, 0.4);
}

void CustomController::quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal) {
    // Reference direction and normal vectors
    Eigen::Vector3d refDirection(1, 0, 0); // Tangent vector reference
    Eigen::Vector3d refNormal(0, 1, 0);    // Normal vector reference

    // Rotate the reference vectors
    tangent = quaternion * refDirection;
    normal = quaternion * refNormal;

    // Normalize the vectors
    tangent.normalize();
    normal.normalize();
}

void CustomController::checkTouchDown() {
    // Check if the foot is in contact with the ground
    if (LF_CF_FT_pre(2) < 10.0 && rd_cc_.LF_CF_FT(2) > 10.0) {
        std::cout << "Left Foot Touch Down" << std::endl;
    }
    if (RF_CF_FT_pre(2) < 10.0 && rd_cc_.RF_CF_FT(2) > 10.0) {
        std::cout << "Right Foot Touch Down" << std::endl;
    }
}

Eigen::Vector3d CustomController::quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {

    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Eigen::Vector3d c = 2.0 * q_vec * q_vec.dot(v);

    return a - b + c;
}

double CustomController::computeReward()
{
    Eigen::Quaterniond quat_cur;
    quat_cur.x() = rd_cc_.q_virtual_(3);
    quat_cur.y() = rd_cc_.q_virtual_(4);
    quat_cur.z() = rd_cc_.q_virtual_(5);
    quat_cur.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    
    double angle = quat_cur.angularDistance(Eigen::Quaterniond::Identity()) * 2;
    double mimic_body_orientation_reward = 0.3 * std::exp(-13.2 * std::abs(angle)); 

    
    Eigen::Matrix<double, MODEL_DOF, 1> joint_position_target;
    Eigen::Matrix<double, 2, 1> force_target;
    double cur_time = std::fmod((rd_cc_.control_time_us_-start_time_)/1e6 + action_dt_accumulate_, 1.7995);
    int mocap_data_idx = int(cur_time / 0.0005) % 3600;
    int next_idx = mocap_data_idx + 1;
    for (int i = 0; i <MODEL_DOF; i++)
    {
        joint_position_target(i) = DyrosMath::cubic(cur_time, mocap_data(mocap_data_idx,0), mocap_data(next_idx,0), 
                                        mocap_data(mocap_data_idx,i+1), mocap_data(next_idx,i+1), 0.0, 0.0);
    }
    for (int i = 0; i < 2; i++)
    {
        force_target(i) = DyrosMath::cubic(cur_time, mocap_data(mocap_data_idx,0), mocap_data(next_idx,0), 
                                        mocap_data(mocap_data_idx,i+33), mocap_data(next_idx,i+33), 0.0, 0.0);
    }
    double qpos_regulation = 0.35 * std::exp(-2.0 * pow((joint_position_target - q_noise_).norm(),2));
    double qvel_regulation = 0.05 * std::exp(-0.01 * pow((q_vel_noise_).norm(),2));

    double contact_force_diff_regulation = 0.2 * std::exp(-0.01*((rd_cc_.LF_FT-LF_FT_pre_).norm() + (rd_cc_.RF_FT-LF_FT_pre_).norm()));
    double torque_regulation = 0.05 * std::exp(-0.01 * (rl_action_.block(0,0,12,1)*333).norm());
    double torque_diff_regulation = 0.6 * std::exp(-0.01 * ((rl_action_.block(0,0,12,1)-rl_action_pre_.block(0,0,12,1))*333).norm());
    double qacc_regulation = 0.05 * std::exp(-20.0*pow((q_vel_noise_-q_vel_noise_pre_).norm(),2));
    Eigen::Vector2d target_vel;
    Eigen::Vector2d cur_vel;
    target_vel << 0.4, 0.0;
    cur_vel << rd_cc_.q_dot_virtual_(0), rd_cc_.q_dot_virtual_(1);
    double body_vel_reward = 0.3 * std::exp(-3.0 * pow((target_vel - cur_vel).norm(),2));
    
    double foot_contact_reward = 0.0;
    if ((3300 <= mocap_data_idx) & (mocap_data_idx < 3600) || (mocap_data_idx < 300) || ((1500 <= mocap_data_idx) & ( mocap_data_idx < 2100)))
    {
        if (abs(rd_cc_.LF_FT(2)) > 100 && abs(rd_cc_.RF_FT(2)) > 100)
            foot_contact_reward = 0.2;
    }
    else if ((300 <= mocap_data_idx) & (mocap_data_idx < 1500))
    {
        if (abs(rd_cc_.LF_FT(2)) < 100 && abs(rd_cc_.RF_FT(2)) > 100)
            foot_contact_reward = 0.2;
    }    
    else if ((2100 <= mocap_data_idx) & (mocap_data_idx < 3300))
    {
        if (abs(rd_cc_.LF_FT(2)) > 100 && abs(rd_cc_.RF_FT(2)) < 100)
            foot_contact_reward = 0.2;
    }

    double force_thres_penalty = 0.0;
    if (abs(rd_cc_.LF_FT(2)) > 1.4*9.81*100 || abs(rd_cc_.RF_FT(2)) > 1.4*9.81*100)
    {
        force_thres_penalty = -0.2;
    }
    double contact_force_penalty = 0.1;
    if (abs(rd_cc_.LF_FT(2)) > 1.4*9.81*100 || abs(rd_cc_.RF_FT(2)) > 1.4*9.81*100)
    {
        contact_force_penalty = 0.1*(1-std::exp(-0.007*((min(abs(rd_cc_.LF_FT(2)) - 1.4*9.81*100, 0.0)) \
                                                            + (min(abs(rd_cc_.RF_FT(2)) - 1.4*9.81*100, 0.0)))));
    }
    double force_diff_thres_penalty = 0.0;
    if (abs(rd_cc_.LF_FT(2)-LF_FT_pre_(2)) > 0.2*9.81*100 || abs(rd_cc_.RF_FT(2)-RF_FT_pre_(2)) > 1.4*9.81*100)
    {
        force_diff_thres_penalty = -0.05;
    }
    double force_ref_reward = 0.1*std::exp(-0.001*(abs(rd_cc_.LF_FT(2)+force_target(0)))) + 0.1*std::exp(-0.001*(abs(rd_cc_.RF_FT(2)+force_target(1))));
    
    LF_FT_pre_ = rd_cc_.LF_FT;
    RF_FT_pre_ = rd_cc_.RF_FT;
    rl_action_pre_ = rl_action_;
    q_vel_noise_pre_ = q_vel_noise_;

    double total_reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + 
        torque_regulation + torque_diff_regulation + body_vel_reward + qacc_regulation + foot_contact_reward + 
        contact_force_diff_regulation  + force_thres_penalty + force_diff_thres_penalty + force_ref_reward;

    return total_reward;
}