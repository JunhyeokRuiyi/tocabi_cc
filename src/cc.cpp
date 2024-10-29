#include "cc.h"
#include <rbdl/Kinematics.h>
#include <yaml-cpp/yaml.h>

using namespace TOCABI;

CustomController::CustomController(RobotData &rd) : rd_(rd) //, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

    if(is_on_robot_) {
        try {
            YAML::Node node = YAML::LoadFile("/home/dyros/catkin_ws/src/tocabi_cc/include/delay_config.yaml");
            // auto delay = node["delay"];
            auto data_path_ = node["result"]["real"].as<std::string>();
            auto target_vel_x_yaml_ = node["target_vel"]["x"].as<double>();
            target_vel_x_yaml = target_vel_x_yaml_;
            data_path = data_path_;


        }
        catch(const YAML::BadFile& e) {
            std::cerr << e.msg << std::endl;
        }
        catch (YAML::ParserException &e){
            std::cerr << e.msg << std::endl;
        }
    }
    else{
        try {
            YAML::Node node = YAML::LoadFile("/home/dyros/tocabi_ws/src/tocabi_cc/include/delay_config.yaml");
            // auto delay = node["delay"];
            auto data_path_ = node["result"]["sim"].as<std::string>();
            auto target_vel_x_yaml_ = node["target_vel"]["x"].as<double>();
            data_path = data_path_;
            target_vel_x_yaml = target_vel_x_yaml_;

        }
        catch(const YAML::BadFile& e) {
            std::cerr << e.msg << std::endl;
        }
        catch (YAML::ParserException &e){
            std::cerr << e.msg << std::endl;
        }
    }

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open(data_path, std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open(data_path, std::ofstream::out | std::ofstream::app);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    ros::param::get("/tocabi_controller/ext_time",param_ext_force_time_);
    ros::param::get("/tocabi_controller/extforce_x",ext_force_x_);
    ros::param::get("/tocabi_controller/extforce_y",ext_force_y_);
    initVariable();
    loadNetwork();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this);
    mujoco_ext_force_apply_pub = nh_.advertise<std_msgs::Float32MultiArray>("/tocabi_avatar/applied_ext_force", 10);
    mujoco_applied_ext_force_.data.resize(7);
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

void CustomController::loadNetwork() //rui weight 불러오기 weight TocabiRL 파일 저장된 12개 파일 저장해주면 됨
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/dyros/tocabi_ws/src/tocabi_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }
    std::ifstream file[14];
    file[0].open(cur_path+"weight/mlp_extractor_policy_net_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"weight/mlp_extractor_policy_net_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"weight/mlp_extractor_policy_net_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"weight/mlp_extractor_policy_net_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"weight/action_net_weight.txt", std::ios::in);
    file[5].open(cur_path+"weight/action_net_bias.txt", std::ios::in);
    file[6].open(cur_path+"weight/obs_mean_fixed.txt", std::ios::in);
    file[7].open(cur_path+"weight/obs_variance_fixed.txt", std::ios::in);
    file[8].open(cur_path+"weight/mlp_extractor_value_net_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"weight/mlp_extractor_value_net_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"weight/mlp_extractor_value_net_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"weight/mlp_extractor_value_net_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"weight/value_net_weight.txt", std::ios::in);
    file[13].open(cur_path+"weight/value_net_bias.txt", std::ios::in);
    file[14].open(cur_path+"processed_data_tocabi_walk.txt", std::ios::in);


    if(!file[0].is_open())
    {
        std::cout<<"Can not find the weight file"<<std::endl;
    }

    float temp;
    int row = 0;
    int col = 0;

    while(!file[0].eof() && row != policy_net_w0_.rows())
    {
        file[0] >> temp;
        if(temp != '\n')
        {
            policy_net_w0_(row, col) = temp;
            col ++;
            if (col == policy_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[1].eof() && row != policy_net_b0_.rows())
    {
        file[1] >> temp;
        if(temp != '\n')
        {
            policy_net_b0_(row, col) = temp;
            col ++;
            if (col == policy_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[2].eof() && row != policy_net_w2_.rows())
    {
        file[2] >> temp;
        if(temp != '\n')
        {
            policy_net_w2_(row, col) = temp;
            col ++;
            if (col == policy_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[3].eof() && row != policy_net_b2_.rows())
    {
        file[3] >> temp;
        if(temp != '\n')
        {
            policy_net_b2_(row, col) = temp;
            col ++;
            if (col == policy_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[4].eof() && row != action_net_w_.rows())
    {
        file[4] >> temp;
        if(temp != '\n')
        {
            action_net_w_(row, col) = temp;
            col ++;
            if (col == action_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[5].eof() && row != action_net_b_.rows())
    {
        file[5] >> temp;
        if(temp != '\n')
        {
            action_net_b_(row, col) = temp;
            col ++;
            if (col == action_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[6].eof() && row != state_mean_.rows())
    {
        file[6] >> temp;
        if(temp != '\n')
        {
            state_mean_(row, col) = temp;
            col ++;
            if (col == state_mean_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[7].eof() && row != state_var_.rows())
    {
        file[7] >> temp;
        if(temp != '\n')
        {
            state_var_(row, col) = temp;
            col ++;
            if (col == state_var_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[8].eof() && row != value_net_w0_.rows())
    {
        file[8] >> temp;
        if(temp != '\n')
        {
            value_net_w0_(row, col) = temp;
            col ++;
            if (col == value_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[9].eof() && row != value_net_b0_.rows())
    {
        file[9] >> temp;
        if(temp != '\n')
        {
            value_net_b0_(row, col) = temp;
            col ++;
            if (col == value_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[10].eof() && row != value_net_w2_.rows())
    {
        file[10] >> temp;
        if(temp != '\n')
        {
            value_net_w2_(row, col) = temp;
            col ++;
            if (col == value_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[11].eof() && row != value_net_b2_.rows())
    {
        file[11] >> temp;
        if(temp != '\n')
        {
            value_net_b2_(row, col) = temp;
            col ++;
            if (col == value_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[12].eof() && row != value_net_w_.rows())
    {
        file[12] >> temp;
        if(temp != '\n')
        {
            value_net_w_(row, col) = temp;
            col ++;
            if (col == value_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[13].eof() && row != value_net_b_.rows())
    {
        file[13] >> temp;
        if(temp != '\n')
        {
            value_net_b_(row, col) = temp;
            col ++;
            if (col == value_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[14].eof() && row != mocap_data.rows())
    {
        file[14] >> temp;
        if(temp != '\n')
        {
            mocap_data(row, col) = temp;
            col ++;
            if (col == mocap_data.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
}

void CustomController::initVariable() //rui 변수 초기화
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
    rl_action_2000_.resize(num_action, 1);
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
    
    action_buffer_2000_.resize(num_action*(num_state_skip*frameskip_custom*num_state_hist), 1); //rui
    state_cur_.resize(num_cur_state, 1);
    state_.resize(num_state, 1);
    state_buffer_2000_.resize(num_cur_state*(num_state_skip*frameskip_custom*num_state_hist), 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_temp_.resize(num_cur_state*num_state_hist, 1);
    state_mean_.resize(num_cur_state, 1);
    state_var_.resize(num_cur_state, 1);

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
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
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

void CustomController::processNoise() //rui noise 만들어주기
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
            q_noise_(i) = rd_cc_.q_virtual_(6+i); // + dis(gen);
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

void CustomController::processObservation() //rui observation 만들어주기 
{
    int data_idx = 0;

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


    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_(data_idx) = q_noise_(i); //rui 12
        data_idx++;
    }

    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rui 12
        }
        else
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6); //q_vel_noise_(i);
        }
        data_idx++;
    }

    float squat_duration = 1.7995;
    phase_ = std::fmod((rd_cc_.control_time_us_-start_time_)/1e6 + action_dt_accumulate_, squat_duration) / squat_duration;
    
    state_cur_(data_idx) = sin(2*M_PI*phase_); //rui 1
    data_idx++;
    state_cur_(data_idx) = cos(2*M_PI*phase_); //rui 1
    data_idx++;
    state_cur_(data_idx) = target_vel_x_yaml;//target_vel_x_; //rui 1
    data_idx++;

    state_cur_(data_idx) = target_vel_y_; //rui 1
    data_idx++;

    // state_cur_(data_idx) = rd_cc_.LF_FT(2);
    // data_idx++;

    // state_cur_(data_idx) = rd_cc_.RF_FT(2);
    // data_idx++;

    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);  //rui 12
        data_idx++;
    }
    state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(num_actuator_action), 0.0, 1.0); //rui 1?
    data_idx++;
    

//!
    // // ** buffer size should be changed regarding to the policy frequency ** //
    // state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1); //rui 0~396 = 44~440, 44개만큼 끌어오고
    // state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array(); //rui 0~440 채워주기

    // // Internal State First
    // for (int i = 0; i < num_state_hist; i++) //rui num_state_hist --> 5 num_cur_internal_state --> 31 (base_ori, q_noise, q_vel_noise, phase_sin, phase_cos, target_vel_x, target_vel_y)
    // {
    //     state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1); //rui (31xi) ~ (31xi)+31 = (44x(2x(i+1)-1)) ~ (44x(2x(i+1)-1))+31 
    // }
    // // Action History Second
    // for (int i = 0; i < num_state_hist-1; i++)
    // {
    //     state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)) + num_cur_internal_state, 0, num_action, 1); //rui (5x31+13xi) ~ (5x31+13xi)+13 = ((44x2x(i+1))+31) ~ ((44x2x(i+1))+31)+13
    // }
//!
    // // std::cout << "----------------state_cur----------------" << std::endl; //rui 44x1
    // // std::cout << "shape: " << state_cur_.rows() << "x" << state_cur_.cols() << std::endl;
    // // std::cout << state_cur_ << std::endl;
    // // std::cout << "----------------state_buffer----------------" << std::endl; //rui 440x1
    // // std::cout << "shape: " << state_buffer_.rows() << "x" << state_buffer_.cols() << std::endl;
    // // std::cout << state_buffer_<< std::endl;
    // // std::cout << "----------------state----------------" << std::endl; //rui 207x1
    // // std::cout << "shape: " << state_.rows() << "x" << state_.cols() << std::endl;
    // // std::cout << state_<< std::endl;

    
}

void CustomController::feedforwardPolicy() //rui mlp feedforward
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

void CustomController::computeSlow() //rui main
{
    copyRobotData(rd_);
    if (rd_cc_.tc_.mode == 7)
    {
        if (rd_cc_.tc_init)
        {
            //Initialize settings for Task Control! 
            start_time_ = rd_cc_.control_time_us_;
            q_noise_pre_ = q_noise_ = q_init_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;

            rd_.tc_init = false;
            std::cout<<"cc mode 7"<<std::endl;
            torque_init_ = rd_cc_.torque_desired;

            processNoise();
            processObservation();





            // for (int i = 0; i < num_state_skip*num_state_hist; i++) 
            // {
            //     state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
            // }

            

            // //! 2000Hz
            for (int i = 0; i < num_state_skip*frameskip_custom*num_state_hist; i++) //rui 0~2*8*5
            {
                state_buffer_2000_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
            }
            // //! 2000Hz
        }

        processNoise();


        // // if ((rd_cc_.control_time_us_ - time_inputTorque_pre_)/1.0e6 > freq_tester_2000HZ){
        // //     cout << "(rd_cc_.control_time_us_ - time_inputTorque_pre_)/1.0e6 2000Hz: " << (rd_cc_.control_time_us_ - time_inputTorque_pre_)/1.0e6 << endl;
        // //     cout << "(rd_cc_.control_time_us_ - time_inputTorque_pre_) 2000Hz: " << (rd_cc_.control_time_us_ - time_inputTorque_pre_) << endl;
        // // }
        // //! 2000Hz
        processObservation(); //rui observation in 2000 44개를 받아옴

        
        if(is_on_robot_) {
            try {
                YAML::Node node = YAML::LoadFile("/home/dyros/catkin_ws/src/tocabi_cc/include/delay_config.yaml");
                // auto delay = node["delay"];
                auto action_delay_ = node["delay"]["action"].as<int>();
                auto observation_delay_ = node["delay"]["observation"].as<int>();

                action_delay = action_delay_;
                observation_delay = observation_delay_;

            }
            catch(const YAML::BadFile& e) {
                std::cerr << e.msg << std::endl;
            }
            catch (YAML::ParserException &e){
                std::cerr << e.msg << std::endl;
            }
        }
        else{
            try {
                YAML::Node node = YAML::LoadFile("/home/dyros/tocabi_ws/src/tocabi_cc/include/delay_config.yaml");
                // auto delay = node["delay"];
                auto action_delay_ = node["delay"]["action"].as<int>();
                auto observation_delay_ = node["delay"]["observation"].as<int>();

                action_delay = action_delay_;
                observation_delay = observation_delay_;

            }
            catch(const YAML::BadFile& e) {
                std::cerr << e.msg << std::endl;
            }
            catch (YAML::ParserException &e){
                std::cerr << e.msg << std::endl;
            }
        }
        // //! 2000Hz

        // cout << "a " << action_delay << "o " << observation_delay << " " << endl;
        
        // //! 2000Hz obs delay
        // ** buffer size should be changed regarding to the policy frequency ** //
        state_buffer_2000_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_custom-1),1) = state_buffer_2000_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist*frameskip_custom-1),1); //rui 0~44*(2*5*8-1) = 44~44*(2*5*8), 44개만큼 끌어오고
        state_buffer_2000_.block(num_cur_state*(num_state_skip*num_state_hist*frameskip_custom-1), 0, num_cur_state,1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array(); //rui 0~44*(2*5*8) 44개 새로 채워주기
        // ** giving delay ** //
        for (int i = 0; i < num_state_hist; i++){//rui num_state_hist --> 5 개의 0~44 
            state_temp_.block(num_cur_state*(i), 0, num_cur_state, 1) = state_buffer_2000_.block(num_cur_state*(num_state_skip*frameskip_custom*(i+1)-1 - observation_delay), 0, num_cur_state, 1); //rui 5 개의 44개 에다가 2000Hz에서 딜레이가 적용된 44크기의 state input
        }

        // Internal State First
        for (int i = 0; i < num_state_hist; i++) //rui num_state_hist --> 5 num_cur_internal_state --> 31 (base_ori, q_noise, q_vel_noise, phase_sin, phase_cos, target_vel_x, target_vel_y)
        {
            state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_temp_.block(num_cur_state*(i), 0, num_cur_internal_state, 1); //rui (31xi) ~ (31xi)+31 = (44x(2x(i+1)-1)) ~ (44x(2x(i+1)-1))+31 
        }
        // Action History Second
        for (int i = 0; i < num_state_hist-1; i++)
        {
            state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_temp_.block(num_cur_state*(i+1) + num_cur_internal_state, 0, num_action, 1); //rui (5x31+13xi) ~ (5x31+13xi)+13 = ((44x2x(i+1))+31) ~ ((44x2x(i+1))+31)+13
        }
        
        // //! 2000Hz obs delay
        
        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us 
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 > freq_scaler_) //rui 250hz 변수만들어서 바꿔주기 default 1/250.0 (control time - inference_time_pre)/ 이 250Hz, 0.004 초 지날때마다 inference
        {
            
            // processObservation(); //rui observation in 2000 44개를 받아옴
            feedforwardPolicy();

            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*freq_scaler_, 0.0, freq_scaler_);
            if (is_write_file_) //rui 파일 write
            {       
                    double reward = computeReward();
                    writeFile << (rd_cc_.control_time_us_ - time_inference_pre_)/1e6 << "\t";
                    writeFile << phase_ << "\t";
                    // writeFile << DyrosMath::minmax_cut(rl_action_(num_action-1)*freq_scaler_, 0.0, freq_scaler_) << "\t";
                    //! 2000Hz
                    writeFile << DyrosMath::minmax_cut(rl_action_2000_(num_action-1)*freq_scaler_, 0.0, freq_scaler_) << "\t";
                    //! 2000Hz

                    writeFile << rd_cc_.LF_FT.transpose() << "\t";
                    writeFile << rd_cc_.RF_FT.transpose() << "\t";
                    writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
                    writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";

                    writeFile << rd_cc_.torque_desired.transpose()  << "\t";
                    writeFile << q_noise_.transpose() << "\t";
                    writeFile << q_dot_lpf_.transpose() << "\t";
                    writeFile << rd_cc_.q_dot_virtual_.transpose() << "\t";
                    writeFile << rd_cc_.q_virtual_.transpose() << "\t";

                    writeFile << value_ << "\t";
                    writeFile << stop_by_value_thres_ <<"\t" ;
                    writeFile << reward ;
                    
                    writeFile << std::endl;
                    
            }
            time_inference_pre_ = rd_cc_.control_time_us_;

        }
        LF_FT_pre_ = rd_cc_.LF_FT;
        RF_FT_pre_ = rd_cc_.RF_FT;
        rl_action_pre_ = rl_action_;
        q_vel_noise_pre_ = q_vel_noise_;

        action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*freq_tester_2000HZ, 0.0, freq_tester_2000HZ); 
        // time_inputTorque_pre_ = rd_cc_.control_time_us_;
        // //! 2000Hz act delay
        //** put action into buffer **//num_action*(num_state_skip*frameskip_custom*num_state_hist)
        action_buffer_2000_.block(0, 0, num_action*(num_state_skip*frameskip_custom*num_state_hist-1),1) = action_buffer_2000_.block(num_action, 0, num_action*(num_state_skip*frameskip_custom*num_state_hist-1),1); //rui 0~13x(2*5*8-1) = 13~13x(2*5*8), 13개만큼 끌어오고
        action_buffer_2000_.block(num_action*(num_state_skip*frameskip_custom*num_state_hist-1), 0, num_action,1) = rl_action_; //rui 새로운 action로 채워주기
        //** apply action delay **//
        if( action_buffer_length  <= action_delay){
            rl_action_2000_ = action_buffer_2000_.block(num_action*(num_state_skip*frameskip_custom*num_state_hist-1 - action_buffer_length), 0, num_action, 1);
            action_buffer_length++;
        }
        else{
            rl_action_2000_ = action_buffer_2000_.block(num_action*(num_state_skip*frameskip_custom*num_state_hist-1 - action_delay), 0, num_action, 1);
        }
        // //! 2000Hz act delay

        // for (int i = 0; i < num_actuator_action; i++)
        // {
        //     torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
        // }
        //! 2000Hz
        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = DyrosMath::minmax_cut(rl_action_2000_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
        }
        //! 2000Hz
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i,i) * (q_init_(i) - q_noise_(i)) - kv_(i,i)*q_vel_noise_(i);
        }
        
        //** torque 는 2000Hz 로 들어감 **//
        if (rd_cc_.control_time_us_ < start_time_ + 0.2e6) //rui torque 쏴주는것
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.2e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
             rd_.torque_desired = torque_rl_;
        }
        
        // if (value_ < 50.0)
        // {
        //     if (stop_by_value_thres_ == false)
        //     {
        //         stop_by_value_thres_ = true;
        //         stop_start_time_ = rd_cc_.control_time_us_;
        //         q_stop_ = q_noise_;
        //         std::cout << "Stop by Value Function" << std::endl;
        //     }
        // }

        if (abs(state_cur_(0)) > 10.0*M_PI/180.0 || abs(state_cur_(1)) > 10.0*M_PI/180.0)
        {
            if (stop_by_value_thres_ == false)
            {
                stop_by_value_thres_ = true;
                stop_start_time_ = rd_cc_.control_time_us_;
                q_stop_ = q_noise_;
                std::cout << "Stop by Value Function" << std::endl;
            }
        }
        if (stop_by_value_thres_)
        {
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
        }

        walking_tick_mj++;
        if((walking_tick_mj >= param_ext_force_time_*hz_)  && (walking_tick_mj < (param_ext_force_time_ + 0.2)*hz_))
        //if((current_step_num_ == 6 && (walking_tick_mj >= t_start_ + t_total_ - 0.1*hz_)) || ((current_step_num_ == 7)  && (walking_tick_mj < t_start_ + 0.1*hz_)))
        { 
            std::cout << "ext_force x : " << ext_force_x_ <<  std::endl;
            std::cout << "ext_force y : " << ext_force_y_ <<  std::endl;
            mujoco_applied_ext_force_.data[0] = ext_force_x_;
            mujoco_applied_ext_force_.data[1] = ext_force_y_;
            mujoco_applied_ext_force_.data[2] =  0.0; //z-axis linear force
            mujoco_applied_ext_force_.data[3] =  0.0; //x-axis angular moment
            mujoco_applied_ext_force_.data[4] =  0.0; //y-axis angular moment
            mujoco_applied_ext_force_.data[5] =  0.0; //z-axis angular moment

            mujoco_applied_ext_force_.data[6] = 1; //link idx; 1:pelvis

            mujoco_ext_force_apply_pub.publish(mujoco_applied_ext_force_);                    
        } 
        else
        {
            mujoco_applied_ext_force_.data[0] = 0; //x-axis linear force
            mujoco_applied_ext_force_.data[1] = 0; //y-axis linear force
            mujoco_applied_ext_force_.data[2] = 0; //z-axis linear force
            mujoco_applied_ext_force_.data[3] = 0; //x-axis angular moment
            mujoco_applied_ext_force_.data[4] = 0; //y-axis angular moment
            mujoco_applied_ext_force_.data[5] = 0; //z-axis angular moment
            mujoco_applied_ext_force_.data[6] = 1; //link idx; 1:pelvis

            mujoco_ext_force_apply_pub.publish(mujoco_applied_ext_force_);
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

    double contact_force_diff_regulation = 0.2 * std::exp(-0.01/250.0*(((rd_cc_.LF_FT-LF_FT_pre_)*(2000.0/frameskip_custom)).norm() + ((rd_cc_.RF_FT-LF_FT_pre_)*(2000.0/frameskip_custom)).norm()));
    // std::cout << "2000.0/frameskip_custom : " << 2000.0/frameskip_custom << " " << typeid(2000.0/frameskip_custom).name()<< std::endl;
    double torque_regulation = 0.05 * std::exp(-0.01 * (rl_action_.block(0,0,12,1)*333).norm());
    double torque_diff_regulation = 0.6 * std::exp(-0.01/250.0* ((rl_action_.block(0,0,12,1)-rl_action_pre_.block(0,0,12,1))*(2000.0/frameskip_custom)*333).norm());
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
    
    // LF_FT_pre_ = rd_cc_.LF_FT;
    // RF_FT_pre_ = rd_cc_.RF_FT;
    // rl_action_pre_ = rl_action_;
    // q_vel_noise_pre_ = q_vel_noise_;

    double total_reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + 
        torque_regulation + torque_diff_regulation + body_vel_reward + qacc_regulation + foot_contact_reward + 
        contact_force_diff_regulation  + force_thres_penalty + force_diff_thres_penalty + force_ref_reward;

    return total_reward;
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(0.5*joy->axes[1], -0.2, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(0.5*joy->axes[0], -0.2, 0.2);
}