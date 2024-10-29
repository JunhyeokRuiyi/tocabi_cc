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

% Specify the CSV file name
filename = 'data_250.csv';

% Detect import options for reading data starting from the second row
opts = detectImportOptions(filename);
opts.DataLines = [2 Inf]; % Read from the second row to the end
data = readmatrix(filename, opts);

% Extract each column or set of columns as specified
elapsed_time_sec = data(:, 1); % Column 1: Elapsed time in seconds
phase = data(:, 2); % Column 2: Phase

% Column 3: Min-max cut value
minmax_cut_value = data(:, 3);

% Columns 4-9: Left Foot Force-Torque data
LF_FT = data(:, 4:9);

% Columns 10-15: Right Foot Force-Torque data
RF_FT = data(:, 10:15);

% Columns 16-21: Left Foot Compensated Force-Torque data
LF_CF_FT = data(:, 16:21);

% Columns 22-27: Right Foot Compensated Force-Torque data
RF_CF_FT = data(:, 22:27);

% Columns 28-60: Desired Torque
torque_desired = data(:, 28:60);

% Columns 61-93: Joint Position Noise Values
q_noise = data(:, 61:93);

% Columns 94-126: Low-pass Filtered Joint Velocities
q_dot_lpf = data(:, 94:126);

% Columns 127-165: Virtual Joint Velocities
q_dot_virtual = data(:, 127:165);

% Columns 166-205: Virtual Joint Positions
q_virtual = data(:, 166:205);

% Columns 206-208: Value, Stop Threshold, and Reward
value = data(:, 206);
stop_by_value_thres = data(:, 207);
reward = data(:, 208);

% Calculate Accumulated Reward up to 32 Seconds
% Find indices where elapsed time is less than or equal to 32 seconds
indices_32 = find(elapsed_time_sec <= 32);
accumulated_reward_32 = sum(reward(indices_32));

% Display accumulated reward at 32 seconds in the command window
fprintf('Accumulated Reward at 32 Seconds: %.4f\n', accumulated_reward_32);

% Plot settings
figure;
set(gcf, 'Position', [100, 100, 1400, 900]); % Set figure size

% Plot 1: Elapsed Time vs Phase
subplot(3, 2, 1);
plot(elapsed_time_sec, phase, '-o', 'LineWidth', 1.5, 'MarkerIndices', 1:200:length(phase));
title('Elapsed Time vs Phase', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Phase', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 2: Min-Max Cut Value
subplot(3, 2, 2);
plot(elapsed_time_sec, minmax_cut_value, '-s', 'LineWidth', 1.5, 'MarkerIndices', 1:200:length(minmax_cut_value), 'Color', [0.85 0.33 0.1]);
title('Min-Max Cut Value over Time', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Min-Max Cut Value', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 3: Left Foot and Right Foot Force-Torque Data
subplot(3, 2, 3);
plot(elapsed_time_sec, LF_FT(:, 1), '-b', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(LF_FT)); hold on;
plot(elapsed_time_sec, RF_FT(:, 1), '-r', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(RF_FT));
title('Left vs Right Foot Force-Torque Data (X-axis)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Force (N)', 'FontSize', 12);
legend({'Left Foot (LF)', 'Right Foot (RF)'}, 'FontSize', 10, 'Location', 'best');
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 4: Compensated Force-Torque Data
subplot(3, 2, 4);
plot(elapsed_time_sec, LF_CF_FT(:, 1), '-g', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(LF_CF_FT)); hold on;
plot(elapsed_time_sec, RF_CF_FT(:, 1), '-m', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(RF_CF_FT));
title('Compensated Force-Torque (LF vs RF) - X-axis', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Compensated Force (N)', 'FontSize', 12);
legend({'Compensated LF', 'Compensated RF'}, 'FontSize', 10, 'Location', 'best');
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 5: Desired Torque
subplot(3, 2, 5);
plot(elapsed_time_sec, torque_desired(:, 1), '-b', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(torque_desired));
title('Desired Torque over Time', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Torque (Nm)', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 6: Joint Position Noise
subplot(3, 2, 6);
plot(elapsed_time_sec, q_noise(:, 1), '-c', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(q_noise));
title('Joint Position Noise (First Joint) over Time', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Noise', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 7: Low-pass Filtered Joint Velocities (Single Joint)
figure;
plot(elapsed_time_sec, q_dot_lpf(:, 1), '-k', 'LineWidth', 1.5, 'MarkerIndices', 1:200:length(q_dot_lpf));
title('Low-pass Filtered Joint Velocities over Time (Single Joint)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Velocity (rad/s)', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 8: Virtual Joint Velocities
figure;
plot(elapsed_time_sec, q_dot_virtual(:, 1), '-m', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(q_dot_virtual));
title('Virtual Joint Velocities over Time (First Joint)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Virtual Joint Velocity (m/s)', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 9: Virtual Joint Positions
figure;
plot(elapsed_time_sec, q_virtual(:, 1), '-g', 'LineWidth', 1.2, 'MarkerIndices', 1:200:length(q_virtual));
title('Virtual Joint Positions over Time (First Joint)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Joint Position (m)', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Plot 10: Reward over Time
figure;
plot(elapsed_time_sec, reward, '-r', 'LineWidth', 1.5, 'MarkerIndices', 1:200:length(reward));
title('Reward Value over Time', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Elapsed Time (s)', 'FontSize', 12);
ylabel('Reward', 'FontSize', 12);
grid on; grid minor;
set(gca, 'FontSize', 12);

% Calculate and display the mean value of the reward
mean_reward = mean(reward);

% Display the mean value on the command window
fprintf('Mean Value of Reward: %.4f\n', mean_reward);

% Display the mean value on the plot as text
hold on;
text(elapsed_time_sec(end) * 0.5, max(reward) * 0.85, ['Mean Reward: ', num2str(mean_reward, '%.4f')], 'FontSize', 14, 'Color', 'b', 'FontWeight', 'bold');

% Display the accumulated reward at 32 seconds on the plot
text(32, max(reward) * 0.75, ['Accumulated Reward @ 32s: ', num2str(accumulated_reward_32, '%.4f')], 'FontSize', 14, 'Color', 'k', 'FontWeight', 'bold');
