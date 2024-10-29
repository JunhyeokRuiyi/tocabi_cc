d1 = load('data.csv');
d2 = load('data_low_frictionloss.csv');

figure();
for i=1:12
    subplot(2,6,i);
    plot(d1(:,i));
    hold on
    plot(d2(:,i));
end

figure();
for i=13:33
    subplot(4,6,i-12);
    plot(d1(:,i));
    hold on
    plot(d2(:,i));
end

%%
clc; clearvars;
% d3 = load('./iserdata/data_nodelay.csv');
% 
% figure();
% % observation 62.5Hz
% plot(d3(2000:3000,1),d3(2000:3000,6))
% hold on
% plot(d3(2000:3000,1),d3(2000:3000,12))
% 
% d4 = load('./iserdata/data_orig.csv');
% 
% figure();
% % observation 2000Hz
% plot(d4(2000:3000,1),d4(2000:3000,6))
% hold on
% plot(d4(2000:3000,1),d4(2000:3000,12))
% 
% d5 = load('./iserdata/data_nodelay_w_buffer.csv');
% 
% figure();
% % observation 2000Hz with buffer
% plot(d5(2000:3000,1),d5(2000:3000,6))
% hold on
% plot(d5(2000:3000,1),d5(2000:3000,12))


d6 = load('./iserdata/data_orig.csv');
% observation 2000Hz with buffer + action_dt_accumulate_ EDIT
figure();
% observation 2000Hz with buffer
plot(d6(2000:3000,1),d6(2000:3000,6))
hold on
plot(d6(2000:3000,1),d6(2000:3000,12))

d7 = load('./iserdata/data_3.csv');
% observation 2000Hz with buffer + action_dt_accumulate_ EDIT
figure();
% observation 2000Hz with buffer
plot(d7(2000:3000,1),d7(2000:3000,6))
hold on
plot(d7(2000:3000,1),d7(2000:3000,12))


%% Value Function
% 1                writeFile << (rd_cc_.control_time_us_ - start_time_)/1e6 << "\t";
% 2                writeFile << phase_ << "\t";
% 3                writeFile << DyrosMath::minmax_cut(rl_action_(num_action-1)*1/250.0, 0.0, 1/250.0) << "\t";
% 
% 4-9                  writeFile << rd_cc_.LF_FT.transpose() << "\t";
% 10-15                writeFile << rd_cc_.RF_FT.transpose() << "\t";
% 16-21                writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
% 22-27                writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";
% 
% 28-60                writeFile << rd_cc_.torque_desired.transpose()  << "\t";
% 61-93                writeFile << q_noise_.transpose() << "\t";
% 94-126                writeFile << q_dot_lpf_.transpose() << "\t";
% 127-165                writeFile << rd_cc_.q_dot_virtual_.transpose() << "\t";
% 166-205                writeFile << rd_cc_.q_virtual_.transpose() << "\t";

% 206 207  208              writeFile << value_ << "\t" << stop_by_value_thres_ << reward;
clear d
d = load('data.csv');

figure()
yyaxis left
plot(d(:,1),d(:,206), 'LineWidth', 7)
ylabel('Value','FontSize', 40, 'FontWeight','bold')
yyaxis right
plot(d(:,1),d(:,end), 'LineWidth', 7)
ylabel('IsStopped','FontSize', 40, 'FontWeight','bold')

set(gca,'FontSize',20, 'FontWeight','bold')
title('Emergent Stop Using Value Function','FontSize', 50)
xlabel('Time(s)','FontSize', 14, 'FontWeight','bold')
legend('Value','Stopped','FontSize', 50, 'FontWeight','bold')
grid on
ax = gca;

ax.GridColor = [0 0 0];
ax.GridLineStyle = '-';
ax.GridAlpha = 0.5;

figure()
plot(d(:,167))

figure()
plot(d(:,[6,12]),'DisplayName','d(:,[6,12])')
%%
figure()
for i=1:12
    subplot(2,6,i)
    plot(d(1:1000,1),d(1:1000,132+i))
    hold on
    plot(d2(1:1000,1),d2(1:1000,132+i))
end

figure()
plot(d(1:1000,1),d(1:1000,6))
hold on
plot(d(1:1000,1),d(1:1000,12))
plot(d2(1:1000,1),d2(1:1000,6))
plot(d2(1:1000,1),d2(1:1000,12))


%%
clc; clearvars;

% Read the CSV file starting from the second row to exclude headers
filename = 'data_250.csv';
data = readmatrix(filename, 'NumHeaderLines', 1);

% Extracting columns from the data
time_diff = data(:, 1); % Column 1: Time differences
phase = data(:, 2); % Column 2: Gait phase
reward = data(:, 208); % Column 208: Reward
reward(isnan(reward)) = 0; % Replace NaN values in reward with zeros
LF_FT = data(:, 4:9); % Columns 4-9: Left Foot Force
RF_FT = data(:, 10:15); % Columns 10-15: Right Foot Force
LF_CF_FT_Z = data(:, 18); % Column 18: Left Foot Clearance (Z)
RF_CF_FT_Z = data(:, 24); % Column 24: Right Foot Clearance (Z)
torque_desired = data(:, 28:60); % Columns 28-60: Joint Torque Desired
q_dot_virtual = data(:, 127:165); % Columns 127-165: Joint Velocities

% Create cumulative time series from the differences
time = cumsum(time_diff); % Now, 'time' can be used as the x-axis

% %% Plot Joint Torque Action Over Time
% figure('Name', 'Joint Torque Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, torque_desired, 'LineWidth', 1.5);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Joint Torque (Nm)', 'FontWeight', 'bold', 'FontSize', 12);
% title('Joint Torque Action Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Foot Force Trajectories Over Time
% figure('Name', 'Foot Forces Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, LF_FT, 'LineWidth', 1.5);
% hold on;
% plot(time, RF_FT, 'LineWidth', 1.5);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Force (N)', 'FontWeight', 'bold', 'FontSize', 12);
% title('Left and Right Foot Forces Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% legend('LF X', 'LF Y', 'LF Z', 'LF Mx', 'LF My', 'LF Mz', 'RF X', 'RF Y', 'RF Z', 'RF Mx', 'RF My', 'RF Mz');
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Gait Phase Over Time
% figure('Name', 'Gait Phase Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% stairs(time, phase, 'LineWidth', 1.5);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Gait Phase', 'FontWeight', 'bold', 'FontSize', 12);
% title('Gait Phase Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Control Effort Over Time
control_effort = sum(torque_desired.^2, 2); % Sum of squared torques
% figure('Name', 'Control Effort Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, control_effort, 'LineWidth', 2);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Control Effort (Sum of Squared Torques)', 'FontWeight', 'bold', 'FontSize', 12);
% title('Control Effort Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Accumulated Reward Over Time
% accumulated_reward = cumsum(reward);
% figure('Name', 'Accumulated Reward Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, accumulated_reward, 'Color', [0.1, 0.7, 0.2], 'LineWidth', 2);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Accumulated Reward', 'FontWeight', 'bold', 'FontSize', 12);
% title('Accumulated Reward Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Foot Clearance Over Time
% figure('Name', 'Foot Clearance Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, LF_CF_FT_Z, 'LineWidth', 1.5);
% hold on;
% plot(time, RF_CF_FT_Z, 'LineWidth', 1.5);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Foot Clearance (m)', 'FontWeight', 'bold', 'FontSize', 12);
% title('Foot Clearance Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% legend('Left Foot', 'Right Foot');
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);
% 
% %% Plot Joint Velocities Over Time
% figure('Name', 'Joint Velocities Over Time', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
% plot(time, q_dot_virtual, 'LineWidth', 1.5);
% xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 12);
% ylabel('Joint Velocities (rad/s)', 'FontWeight', 'bold', 'FontSize', 12);
% title('Joint Velocities Over Time', 'FontWeight', 'bold', 'FontSize', 14);
% grid on;
% set(gca, 'GridAlpha', 0.3, 'FontSize', 12);

% Plot Reward vs. Control Effort (Scatter Plot)
figure('Name', 'Reward vs Control Effort', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
scatter(control_effort, reward, 50, 'filled', 'MarkerEdgeColor', [0, 0.5, 0.5], 'MarkerFaceColor', [0, 0.7, 0.7], 'MarkerFaceAlpha', 0.6);
xlabel('Control Effort (Sum of Squared Torque)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('Reward', 'FontWeight', 'bold', 'FontSize', 14);
title('Reward vs Control Effort', 'FontWeight', 'bold', 'FontSize', 16);
grid on;
set(gca, 'GridAlpha', 0.4, 'FontSize', 14);
box on;
set(gca, 'LineWidth', 1.2);

