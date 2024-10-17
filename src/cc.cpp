#include "cc.h"

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

//? yaml {
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
    ros::param::get("/tocabi_controller/vel_cubic_scaler", vel_cubic_scaler_); 
    std::cout << "freq_scaler : " << vel_cubic_scaler_ << std::endl;
    ros::param::get("/tocabi_controller/is_on_robot", is_on_robot_); 
    std::cout << "is_on_robot : " << is_on_robot_ << std::endl;

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
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/dyros/tocabi_ws/src/tocabi_cc/weight/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/weight/";
    }
    std::ifstream file[15];
    file[0].open(cur_path+"a2c_network_actor_mlp_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"a2c_network_actor_mlp_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"a2c_network_actor_mlp_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"a2c_network_actor_mlp_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"a2c_network_mu_weight.txt", std::ios::in);
    file[5].open(cur_path+"a2c_network_mu_bias.txt", std::ios::in);
    file[6].open(cur_path+"obs_mean_fixed.txt", std::ios::in);
    file[7].open(cur_path+"obs_variance_fixed.txt", std::ios::in);
    file[8].open(cur_path+"a2c_network_critic_mlp_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"a2c_network_critic_mlp_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"a2c_network_critic_mlp_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"a2c_network_critic_mlp_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"a2c_network_value_weight.txt", std::ios::in);
    file[13].open(cur_path+"a2c_network_value_bias.txt", std::ios::in);
    file[14].open(cur_path+"processed_data_tocabi_walk.txt", std::ios::in);

    if (!file[0].is_open()) {
        std::cerr << "Error: Cannot open weight file a2c_network_actor_mlp_0_weight.txt" << std::endl;
        return;
    }

    auto loadMatrix = [&](std::ifstream &file, Eigen::MatrixXd &matrix) {
        int row = 0, col = 0;
        float temp;
        while (file >> temp && row < matrix.rows()) {
            matrix(row, col++) = temp;
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
    loadMatrix(file[14], mocap_data);

}

void CustomController::initVariable()
{    
    policy_net_w0_.resize(num_hidden, num_state);
    policy_net_b0_.resize(num_hidden, 1);
    policy_net_w2_.resize(num_hidden, num_hidden);
    policy_net_b2_.resize(num_hidden, 1);
    action_net_w_.resize(num_action, num_hidden);
    action_net_b_.resize(num_action, 1);
    hidden_layer1_.resize(num_hidden, 1);
    hidden_layer2_.resize(num_hidden, 1);
    rl_action_.resize(num_action, 1);
    rl_action_pre_.resize(num_action, 1);
    mocap_data.resize(3600,36);

    value_net_w0_.resize(num_hidden, num_state);
    value_net_b0_.resize(num_hidden, 1);
    value_net_w2_.resize(num_hidden, num_hidden);
    value_net_b2_.resize(num_hidden, 1);
    value_net_w_.resize(1, num_hidden);
    value_net_b_.resize(1, 1);
    value_hidden_layer1_.resize(num_hidden, 1);
    value_hidden_layer2_.resize(num_hidden, 1);
    
    state_cur_.resize(num_cur_state, 1);
    state_.resize(num_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_mean_.resize(num_cur_state, 1);
    state_var_.resize(num_cur_state, 1);

    q_dot_lpf_.setZero();

    rl_action_simfreq_.resize(num_action, 1);
    state_temp_.resize(num_cur_state*num_state_hist, 1);
    state_buffer_simfreq_.resize(num_cur_state*(num_state_skip*num_state_hist*frameskip_), 1);
    action_buffer_simfreq_.resize(num_action*(num_state_skip*frameskip_*num_state_hist), 1);

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
    kp_.diagonal() /= 9.0;
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    kv_.diagonal() /= 3.0;
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
    ;obs
    ;1) root_rot: root rotation                  (3)     0:2
    ;2) dof_pos: dof position                    (12)    3:14
    ;3) dof_vel: dof velocity                    (12)    15:26 
    ;4) phase: sin cos                           (2)     27:29
    ;5) commands: x, y                           (2)     30:32
    ;6) root_vel: root linear velocity           (3)     31:33
    ;7) root_ang_vel: root angular velocity      (3)     34:36
    ;8) action: actuator + phase                 (13)        
    */

    int data_idx = 0;

    //;1) root_rot: root rotation                  (3)     0:2
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);    

    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    state_cur_(data_idx) = euler_angle_(0); //rui 1
    data_idx++;

    state_cur_(data_idx) = euler_angle_(1); //rui 1
    data_idx++;

    state_cur_(data_idx) = euler_angle_(2); //rui 1
    data_idx++;

    //;2) dof_pos: dof position                    (12)    3:14
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_(data_idx) = q_noise_(i); //rui 12
        data_idx++;
    }

    //;3) dof_vel: dof velocity                    (12)    15:26 
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

    float squat_duration = 1.7995;
    phase_ = std::fmod((rd_cc_.control_time_us_-start_time_)/1e6 + action_dt_accumulate_, squat_duration) / squat_duration;
    //;4) phase: sin cos                           (2)     27:29
    state_cur_(data_idx) = sin(2*M_PI*phase_); //rui 1
    data_idx++;
    state_cur_(data_idx) = cos(2*M_PI*phase_); //rui 1
    data_idx++;
    
    //;5) commands: x, y                           (2)     30:32
    // if (rd_cc_.control_time_us_ < start_time_ + 10e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + vel_cubic_scaler_, 0.0, target_vel_x_yaml_, 0.0, 0.0);
    //     // desired_vel_yaw = 0.0;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 20e6) {
    //     // desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 14e6, start_time_ + 19e6, 0.3, 0.0, 0.0, 0.0);
    //     desired_vel_x = target_vel_x_yaml_;
    //     // desired_vel_yaw = 0.0;
    // }
    // else if (rd_cc_.control_time_us_ < start_time_ + 30e6) {
    //     desired_vel_x = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_ + 26e6, start_time_ + 26e6 + vel_cubic_scaler_, target_vel_x_yaml_, 0.0, 0.0, 0.0);
    //     // desired_vel_yaw = 0.0;
    // }
    // else {
    //     desired_vel_x = 0.0;
    //     // desired_vel_yaw = 0.0;
    // }
    
    
    // state_cur_(data_idx) = desired_vel_x;//target_vel_x_; //rui 1
    state_cur_(data_idx) = target_vel_x_yaml_;//target_vel_x_; //rui 1
    data_idx++;

    // state_cur_(data_idx) = target_vel_y_yaml_;//target_vel_y_; //rui 1
    state_cur_(data_idx) = 0.0;//target_vel_y_; //rui 1
    data_idx++;

    //;6) root_vel: root linear velocity           (3)     31:33
    //;7) root_ang_vel: root angular velocity      (3)     34:36

    for (int i=0; i<6; i++)
    {
        state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i); //rui 6 base_lin_vel base_ang_vel
        data_idx++;
    }

    // state_cur_(data_idx) = -rd_cc_.LF_FT(2);
    // data_idx++;

    // state_cur_(data_idx) = -rd_cc_.RF_FT(2);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(3);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(3);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(4);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(4);
    // data_idx++;

    //;8) action: actuator + phase                 (13)        
    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);  //rui 12
        data_idx++;
    }
    state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(num_actuator_action), 0.0, 1.0); //rui 1(phase)
    data_idx++;
    
//orig
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
//orig
//SECTION - //? 500Hz
    // ** buffer size should be changed regarding to the policy frequency ** //
    
    state_buffer_simfreq_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_-1), 1) = state_buffer_simfreq_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_-1),1); //rui 0~50*(2*10*frameskip-1) = 50~50*(2*10*frameskip), 50개만큼 끌어오고
    state_buffer_simfreq_.block(num_cur_state*(num_state_skip*num_state_hist*frameskip_-1), 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array(); //rui 50*(2*10*frameskip-1)~50*(2*10*frameskip) 50개 새로 채워주기
    
    // ** giving delay ** //
    // Internal State First
    for (int i = 0; i < num_state_hist; i++) //rui num_state_hist --> 10 num_cur_internal_state --> 37 (base_ori, q_noise, q_vel_noise, phase_sin, phase_cos, target_vel_x, target_vel_y, root_lin_vel, root_ang_vel)
    {
        // state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_temp_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1); //rui (37xi) ~ (37xi)+37 = (50xi) ~ (50xi)+37 
        state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_simfreq_.block(num_cur_state*(num_state_skip*frameskip_*(i+1)-1 - observation_delay_), 0, num_cur_internal_state, 1);
    }
    
    // Action History Second
    for (int i = 0; i < num_state_hist-1; i++)
    {
        // state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_temp_.block(num_cur_state*(i) + num_cur_internal_state, 0, num_action, 1); //rui (10x37+13xi) ~ (10x37+13xi)+13 = ((50xi)+37) ~ ((50xi)+37)+13
        state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_simfreq_.block(num_cur_state*(num_state_skip*frameskip_*(i+1) - observation_delay_) + num_cur_internal_state, 0, num_action, 1); //rui (10x37+13xi) ~ (10x37+13xi)+13 = ((50xi)+37) ~ ((50xi)+37)+13
    }
    
//!SECTION - //? 500Hz

}

void CustomController::feedforwardPolicy()
{
    hidden_layer1_ = policy_net_w0_ * state_ + policy_net_b0_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (hidden_layer1_(i) < 0)
            hidden_layer1_(i) = 0.0;
    }

    hidden_layer2_ = policy_net_w2_ * hidden_layer1_ + policy_net_b2_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (hidden_layer2_(i) < 0)
            hidden_layer2_(i) = 0.0;
    }

    rl_action_ = action_net_w_ * hidden_layer2_ + action_net_b_;

    value_hidden_layer1_ = value_net_w0_ * state_ + value_net_b0_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (value_hidden_layer1_(i) < 0)
            value_hidden_layer1_(i) = 0.0;
    }

    value_hidden_layer2_ = value_net_w2_ * value_hidden_layer1_ + value_net_b2_;
    for (int i = 0; i < num_hidden; i++) 
    {
        if (value_hidden_layer2_(i) < 0)
            value_hidden_layer2_(i) = 0.0;
    }

    value_ = (value_net_w_ * value_hidden_layer2_ + value_net_b_)(0);
    
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
            
// orig {
            // for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            // {
            //     state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
            //     // state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1).setZero();
            // }
// orig }
//SECTION - //? 500Hz
            for (int i = 0; i < num_state_skip*frameskip_*num_state_hist; i++) //rui 0~2*8*5
            {
                state_buffer_simfreq_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
            }
            
//!SECTION - //? 500Hz
        }
//!SECTION - init

        processNoise();
//SECTION - //? 500Hz
        processObservation();
        
        // // ** buffer size should be changed regarding to the policy frequency ** //
        // state_buffer_simfreq_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_-1),1) = state_buffer_simfreq_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_-1),1); //rui 0~44*(2*5*8-1) = 44~44*(2*5*8), 44개만큼 끌어오고
        // state_buffer_simfreq_.block(num_cur_state*(num_state_skip*num_state_hist*frameskip_-1), 0, num_cur_state,1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array(); //rui 0~44*(2*5*8) 44개 새로 채워주기
        // // ** giving delay ** //
        // for (int i = 0; i < num_state_hist; i++){//rui num_state_hist --> 5 개의 0~44 
        //     state_temp_.block(num_cur_state*(i), 0, num_cur_state, 1) = state_buffer_simfreq_.block(num_cur_state*(num_state_skip*frameskip_*(i+1)-1 - observation_delay_), 0, num_cur_state, 1); //rui 5 개의 44개 에다가 2000Hz에서 딜레이가 적용된 44크기의 state input
        // }
        // // Internal State First
        // for (int i = 0; i < num_state_hist; i++) //rui num_state_hist --> 5 num_cur_internal_state --> 31 (base_ori, q_noise, q_vel_noise, phase_sin, phase_cos, target_vel_x, target_vel_y)
        // {
        //     state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_temp_.block(num_cur_state*(i), 0, num_cur_internal_state, 1); //rui (31xi) ~ (31xi)+31 = (44x(2x(i+1)-1)) ~ (44x(2x(i+1)-1))+31 
        // }
        // // Action History Second
        // for (int i = 0; i < num_state_hist-1; i++)
        // {
        //     state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_temp_.block(num_cur_state*(i+1) + num_cur_internal_state, 0, num_action, 1); //rui (5x31+13xi) ~ (5x31+13xi)+13 = ((44x2x(i+1))+31) ~ ((44x2x(i+1))+31)+13
        // }
//!SECTION - //? 500Hz
        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= freq_scaler_ - 1/10000.0) // orig
//SECTION - feedforwardPolicy
        // if (policy_step >= frameskip_)
        {
            // processObservation(); // orig 
            feedforwardPolicy();
            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/250.0, 0.0, 5/250.0); // orig 

            if (value_ < 50.0)
            {
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function" << std::endl;
                }
            }
            std::cout << "Value : " << value_ << std::endl;

            if (is_write_file_)
            {
                    double reward = computeReward();
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
                    writeFile << value_ << "\t" << stop_by_value_thres_ <<"\t" << reward;
                    writeFile << std::endl;
                    time_write_pre_ = rd_cc_.control_time_us_;
            }
            // std::cout << policy_step << " " << rd_cc_.control_time_us_ - time_inference_pre_ << " " << value_ << " " << stop_by_value_thres_ << std::endl;
            time_inference_pre_ = rd_cc_.control_time_us_;
            // policy_step = 0;
        }
//!SECTION - feedforwardPolicy
        // policy_step++;
        
        action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5*0.0005, 0.0, 5*0.0005); 
        // time_inputTorque_pre_ = rd_cc_.control_time_us_;

//SECTION - //? 500Hz act delay
        //** put action into buffer **//num_action*(num_state_skip*frameskip_custom*num_state_hist)
        action_buffer_simfreq_.block(0, 0, num_action*(num_state_skip*frameskip_*num_state_hist-1),1) = action_buffer_simfreq_.block(num_action, 0, num_action*(num_state_skip*frameskip_*num_state_hist-1),1); //rui 0~13x(2*5*8-1) = 13~13x(2*5*8), 13개만큼 끌어오고
        action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1), 0, num_action,1) = rl_action_; //rui 새로운 action로 채워주기
        //** apply action delay **//
        if( action_buffer_length  <= action_delay_){
            rl_action_simfreq_ = action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1 - action_buffer_length), 0, num_action, 1);
            action_buffer_length++;
        }
        else{
            rl_action_simfreq_ = action_buffer_simfreq_.block(num_action*(num_state_skip*frameskip_*num_state_hist-1 - action_delay_), 0, num_action, 1);
        }
//SECTION - //? 500Hz act delay


        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = DyrosMath::minmax_cut(rl_action_simfreq_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
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

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(0.5 * joy->axes[1], -0.2, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(0.5 * joy->axes[0], -0.2, 0.2);
}