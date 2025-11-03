%% OFDM Complete System with Detailed Transmitter/Receiver Visualization
% CORRECTED: Proper Rician channel and extended SNR for waterfall curves

clear all; close all; clc;
tic; % Start timer

%% ========================================================================
%% PART 0: DETAILED OFDM TRANSMITTER AND RECEIVER VISUALIZATION
%% ========================================================================
fprintf('========================================\n');
fprintf('OFDM COMPLETE SYSTEM DEMONSTRATION\n');
fprintf('========================================\n\n');

%% System Parameters
Nsc = 64;               % Number of subcarriers
CP_len = 16;            % Cyclic Prefix length
SNR_dB_demo = 20;       % SNR for demonstration
fc = 2e9;               % Carrier frequency (2 GHz)
Fs = 20e6;              % Sampling frequency
Ts = 1/Fs;              % Sampling period
T_sym = (Nsc + CP_len) * Ts;  % OFDM symbol duration
K_rician = 5;           % Rician K-factor
fd_demo = 100;          % Doppler frequency for demo

%% Generate Random Codeword
codeword_input = randi([0, 1], 384, 1);
fprintf('Generated random codeword of 384 bits\n');
fprintf('Codeword first 32 bits: %s...\n\n', sprintf('%d', codeword_input(1:32)));

modulation_schemes = {'QPSK', '16QAM', '64QAM'};
M_schemes = [4, 16, 64];

%% Storage for all three modulations
tx_data_all = cell(3, 1);
tx_symbols_all = cell(3, 1);
tx_freq_all = cell(3, 1);
tx_time_all = cell(3, 1);
tx_with_cp_all = cell(3, 1);
rx_with_noise_all = cell(3, 1);
rx_no_cp_all = cell(3, 1);
rx_freq_all = cell(3, 1);
rx_equalized_all = cell(3, 1);
rx_symbols_all = cell(3, 1);
rx_bits_all = cell(3, 1);

%% Process Each Modulation Scheme
for mod_idx = 1:3
    M = M_schemes(mod_idx);
    k = log2(M);
    mod_name = modulation_schemes{mod_idx};
    
    fprintf('----- Processing %s -----\n', mod_name);
    
    %% TRANSMITTER
    n_bits_needed = Nsc * k;
    tx_bits = codeword_input(1:n_bits_needed);
    tx_data_all{mod_idx} = tx_bits;
    fprintf('  [TX] Input: %d bits\n', n_bits_needed);
    
    tx_symbols = qammod(tx_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
    tx_symbols_all{mod_idx} = tx_symbols;
    fprintf('  [TX] Modulation: %d %s symbols\n', Nsc, mod_name);
    
    tx_freq = tx_symbols;
    tx_freq_all{mod_idx} = tx_freq;
    fprintf('  [TX] Frequency Domain: %d subcarriers\n', Nsc);
    
    tx_time = ifft(tx_freq, Nsc);
    tx_time_all{mod_idx} = tx_time;
    fprintf('  [TX] IFFT: %d samples\n', Nsc);
    
    tx_with_cp = [tx_time(Nsc - CP_len + 1:Nsc); tx_time];
    tx_with_cp_all{mod_idx} = tx_with_cp;
    fprintf('  [TX] Add CP: %d samples\n', Nsc + CP_len);
    
    %% CHANNEL - CORRECTED RICIAN
    h_channel = (randn + 1j*randn) / sqrt(2);
    
    t_vec = (0:Nsc+CP_len-1) * Ts;
    doppler_phase = exp(1j * 2 * pi * fd_demo * t_vec).';
    
    rx_signal = tx_with_cp * h_channel .* doppler_phase;
    
    SNR_lin = 10^(SNR_dB_demo/10);
    noise_power = 1 / SNR_lin;
    noise = sqrt(noise_power/2) * (randn(size(rx_signal)) + 1j*randn(size(rx_signal)));
    rx_with_noise = rx_signal + noise;
    rx_with_noise_all{mod_idx} = rx_with_noise;
    fprintf('  [CH] Rayleigh + Doppler + AWGN\n');
    
    %% RECEIVER
    rx_no_cp = rx_with_noise(CP_len+1:end);
    rx_no_cp_all{mod_idx} = rx_no_cp;
    fprintf('  [RX] Remove CP\n');
    
    rx_freq = fft(rx_no_cp, Nsc);
    rx_freq_all{mod_idx} = rx_freq;
    fprintf('  [RX] FFT\n');
    
    H_channel = h_channel * ones(Nsc, 1);
    ICI_factor = (2 * pi * fd_demo * T_sym);
    ICI_power = ICI_factor^2 / 6;
    effective_noise_power = noise_power + ICI_power * abs(H_channel).^2;
    
    H_conj = conj(H_channel);
    H_mag_sq = abs(H_channel).^2;
    mmse_weights = H_conj ./ (H_mag_sq + effective_noise_power);
    
    rx_equalized = rx_freq .* mmse_weights;
    rx_equalized_all{mod_idx} = rx_equalized;
    fprintf('  [RX] MMSE Equalization\n');
    
    rx_symbols = rx_equalized;
    rx_symbols_all{mod_idx} = rx_symbols;
    
    rx_bits = qamdemod(rx_symbols, M, 'OutputType', 'bit', 'UnitAveragePower', true);
    rx_bits_all{mod_idx} = rx_bits;
    fprintf('  [RX] Demodulation\n');
    
    bit_errors = sum(tx_bits ~= rx_bits);
    ber = bit_errors / n_bits_needed;
    fprintf('  [OUTPUT] BER = %.4f (%d/%d errors)\n\n', ber, bit_errors, n_bits_needed);
end

%% ========================================================================
%% VISUALIZATION: TRANSMITTER STAGES
%% ========================================================================
fprintf('Generating Transmitter Visualization...\n');

figure('Name', 'OFDM Transmitter - All Stages', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1400, 900]);

for mod_idx = 1:3
    mod_name = modulation_schemes{mod_idx};
    row_start = (mod_idx-1)*4 + 1;
    
    t_no_cp = (0:Nsc-1) * Ts * 1e6;
    t_with_cp = (0:Nsc+CP_len-1) * Ts * 1e6;
    
    subplot(3, 4, row_start);
    plot(real(tx_symbols_all{mod_idx}), imag(tx_symbols_all{mod_idx}), 'bo', ...
         'MarkerSize', 6, 'LineWidth', 1.5);
    xlabel('In-Phase', 'FontSize', 9);
    ylabel('Quadrature', 'FontSize', 9);
    title(sprintf('%s TX Constellation', mod_name), 'FontSize', 10, 'FontWeight', 'bold');
    grid on; axis equal; xlim([-2 2]); ylim([-2 2]);
    
    subplot(3, 4, row_start + 1);
    stem(0:Nsc-1, abs(tx_freq_all{mod_idx}), 'b', 'filled', 'LineWidth', 1);
    xlabel('Subcarrier Index', 'FontSize', 9);
    ylabel('Magnitude', 'FontSize', 9);
    title('TX Frequency Domain', 'FontSize', 10, 'FontWeight', 'bold');
    grid on; xlim([0 Nsc-1]);
    
    subplot(3, 4, row_start + 2);
    plot(t_no_cp, real(tx_time_all{mod_idx}), 'b-', 'LineWidth', 1);
    hold on;
    plot(t_no_cp, imag(tx_time_all{mod_idx}), 'r--', 'LineWidth', 1);
    xlabel('Time (?s)', 'FontSize', 9);
    ylabel('Amplitude', 'FontSize', 9);
    title('TX Time (No CP)', 'FontSize', 10, 'FontWeight', 'bold');
    legend('Real', 'Imag', 'FontSize', 7);
    grid on;
    
    subplot(3, 4, row_start + 3);
    plot(t_with_cp, abs(tx_with_cp_all{mod_idx}), 'b-', 'LineWidth', 1.5);
    xline(CP_len * Ts * 1e6, 'r--', 'LineWidth', 2);
    xlabel('Time (?s)', 'FontSize', 9);
    ylabel('Magnitude', 'FontSize', 9);
    title('TX Envelope (With CP)', 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
end
sgtitle('OFDM Transmitter Processing Stages', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% VISUALIZATION: RECEIVER STAGES
%% ========================================================================
fprintf('Generating Receiver Visualization...\n');

figure('Name', 'OFDM Receiver - All Stages', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1400, 900]);

for mod_idx = 1:3
    mod_name = modulation_schemes{mod_idx};
    row_start = (mod_idx-1)*4 + 1;
    
    t_with_cp = (0:Nsc+CP_len-1) * Ts * 1e6;
    t_no_cp = (0:Nsc-1) * Ts * 1e6;
    
    subplot(3, 4, row_start);
    plot(t_with_cp, abs(rx_with_noise_all{mod_idx}), 'b-', 'LineWidth', 1);
    xline(CP_len * Ts * 1e6, 'r--', 'LineWidth', 1.5);
    xlabel('Time (?s)', 'FontSize', 9);
    ylabel('Magnitude', 'FontSize', 9);
    title(sprintf('%s RX + Noise', mod_name), 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
    
    subplot(3, 4, row_start + 1);
    plot(t_no_cp, real(rx_no_cp_all{mod_idx}), 'b-', 'LineWidth', 1);
    hold on;
    plot(t_no_cp, imag(rx_no_cp_all{mod_idx}), 'r--', 'LineWidth', 1);
    xlabel('Time (?s)', 'FontSize', 9);
    ylabel('Amplitude', 'FontSize', 9);
    title('RX After CP Removal', 'FontSize', 10, 'FontWeight', 'bold');
    legend('Real', 'Imag', 'FontSize', 7);
    grid on;
    
    subplot(3, 4, row_start + 2);
    stem(0:Nsc-1, abs(rx_freq_all{mod_idx}), 'r', 'filled', 'LineWidth', 1);
    xlabel('Subcarrier Index', 'FontSize', 9);
    ylabel('Magnitude', 'FontSize', 9);
    title('RX Freq (Before EQ)', 'FontSize', 10, 'FontWeight', 'bold');
    grid on; xlim([0 Nsc-1]);
    
    subplot(3, 4, row_start + 3);
    plot(real(rx_symbols_all{mod_idx}), imag(rx_symbols_all{mod_idx}), 'ro', ...
         'MarkerSize', 6, 'LineWidth', 1);
    hold on;
    ideal_points = qammod(0:M_schemes(mod_idx)-1, M_schemes(mod_idx), 'UnitAveragePower', true);
    plot(real(ideal_points), imag(ideal_points), 'k+', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('In-Phase', 'FontSize', 9);
    ylabel('Quadrature', 'FontSize', 9);
    title('RX Constellation (After EQ)', 'FontSize', 10, 'FontWeight', 'bold');
    legend('RX', 'Ideal', 'FontSize', 7);
    grid on; axis equal; xlim([-2 2]); ylim([-2 2]);
end
sgtitle('OFDM Receiver Processing Stages', 'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% VISUALIZATION: TX vs RX COMPARISON
%% ========================================================================
fprintf('Generating TX vs RX Comparison...\n');

figure('Name', 'OFDM TX vs RX Comparison', 'NumberTitle', 'off', ...
       'Position', [150, 150, 1200, 800]);

for mod_idx = 1:3
    mod_name = modulation_schemes{mod_idx};
    
    subplot(3, 3, (mod_idx-1)*3 + 1);
    plot(real(tx_symbols_all{mod_idx}), imag(tx_symbols_all{mod_idx}), 'bo', ...
         'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'TX');
    hold on;
    plot(real(rx_symbols_all{mod_idx}), imag(rx_symbols_all{mod_idx}), 'rx', ...
         'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'RX');
    xlabel('In-Phase', 'FontSize', 9, 'FontWeight', 'bold');
    ylabel('Quadrature', 'FontSize', 9, 'FontWeight', 'bold');
    title(sprintf('%s: TX vs RX', mod_name), 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on; axis equal; xlim([-2 2]); ylim([-2 2]);
    
    subplot(3, 3, (mod_idx-1)*3 + 2);
    plot(0:Nsc-1, abs(tx_freq_all{mod_idx}), 'b-', 'LineWidth', 2, 'DisplayName', 'TX');
    hold on;
    plot(0:Nsc-1, abs(rx_equalized_all{mod_idx}), 'r--', 'LineWidth', 1.5, 'DisplayName', 'RX (EQ)');
    xlabel('Subcarrier Index', 'FontSize', 9, 'FontWeight', 'bold');
    ylabel('Magnitude', 'FontSize', 9, 'FontWeight', 'bold');
    title('Frequency Domain', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on; xlim([0 Nsc-1]);
    
    subplot(3, 3, (mod_idx-1)*3 + 3);
    t_no_cp = (0:Nsc-1) * Ts * 1e6;
    plot(t_no_cp, abs(tx_time_all{mod_idx}), 'b-', 'LineWidth', 2, 'DisplayName', 'TX');
    hold on;
    plot(t_no_cp, abs(rx_no_cp_all{mod_idx}), 'r--', 'LineWidth', 1.5, 'DisplayName', 'RX');
    xlabel('Time (?s)', 'FontSize', 9, 'FontWeight', 'bold');
    ylabel('Magnitude', 'FontSize', 9, 'FontWeight', 'bold');
    title('Time Domain Envelope', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 8);
    grid on;
end
sgtitle('TX vs RX: Constellation, Frequency, and Time Domain', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% ========================================================================
%% MAIN SIMULATION - CORRECTED FOR WATERFALL CURVES
%% ========================================================================
fprintf('\nStarting BER Simulations...\n');

doppler_vec = 0:25:300;
n_doppler = length(doppler_vec);
SNR_vec = 0:2:50;           % EXTENDED to 50 dB for complete waterfall
n_snr = length(SNR_vec);
SNR_dB_doppler = 20;
fd_fixed = 100;
Nsym_per_run = 500;
Nbits_sim = 100000;         % INCREASED for better accuracy at high SNR

modulations = {'QPSK', '16QAM', '64QAM'};
M_values = [4, 16, 64];

BER_Rayleigh = zeros(3, n_doppler);
BER_Rician = zeros(3, n_doppler);
BER_SNR_Rayleigh = zeros(3, n_snr);
BER_SNR_Rician = zeros(3, n_snr);
BER_SNR_AWGN = zeros(3, n_snr);

%% BER vs Doppler
fprintf('BER vs Doppler: ');
SNR_lin = 10^(SNR_dB_doppler/10);

for mod_idx = 1:3
    M = M_values(mod_idx);
    k = log2(M);
    fprintf('%s.', modulations{mod_idx});
    
    for dop_idx = 1:n_doppler
        fd = doppler_vec(dop_idx);
        n_runs = ceil(Nbits_sim / (Nsc * k * Nsym_per_run));
        err_ray = 0; err_ric = 0; tot = 0;
        
        for run = 1:n_runs
            tx_b = randi([0, 1], Nsc * k * Nsym_per_run, 1);
            tx_s = qammod(tx_b, M, 'InputType', 'bit', 'UnitAveragePower', true);
            tx_sm = reshape(tx_s, Nsc, Nsym_per_run);
            
            tx_t = zeros(Nsc, Nsym_per_run);
            for s = 1:Nsym_per_run
                tx_t(:, s) = ifft(tx_sm(:, s), Nsc);
            end
            
            tx_cp = zeros(Nsc + CP_len, Nsym_per_run);
            for s = 1:Nsym_per_run
                tx_cp(:, s) = [tx_t(Nsc-CP_len+1:Nsc, s); tx_t(:, s)];
            end
            
            % CORRECTED Rayleigh channel
            h_r = (randn(1, Nsym_per_run) + 1j*randn(1, Nsym_per_run)) / sqrt(2);
            
            % CORRECTED Rician channel with proper normalization
            h_los = ones(1, Nsym_per_run);  % LOS component
            h_nlos = (randn(1, Nsym_per_run) + 1j*randn(1, Nsym_per_run)) / sqrt(2);
            h_ric = sqrt(K_rician/(K_rician+1)) * h_los + ...
                    sqrt(1/(K_rician+1)) * h_nlos;
            
            rx_r = zeros(size(tx_cp));
            rx_ric = zeros(size(tx_cp));
            
            for s = 1:Nsym_per_run
                t_s = (s-1)*(Nsc+CP_len)*Ts + (0:Nsc+CP_len-1)*Ts;
                dop = exp(1j*2*pi*fd*t_s);
                rx_r(:,s) = tx_cp(:,s)*h_r(s).*dop.';
                rx_ric(:,s) = tx_cp(:,s)*h_ric(s).*dop.';
            end
            
            np = 1/SNR_lin;
            n = sqrt(np/2)*(randn(size(rx_r(:)))+1j*randn(size(rx_r(:))));
            rx_r = reshape(rx_r(:)+n, Nsc+CP_len, Nsym_per_run);
            rx_ric = reshape(rx_ric(:)+n, Nsc+CP_len, Nsym_per_run);
            
            [rb_r] = ofdm_rx(rx_r, h_r, fd, T_sym, np, Nsc, CP_len, M, Nsym_per_run);
            [rb_ric] = ofdm_rx(rx_ric, h_ric, fd, T_sym, np, Nsc, CP_len, M, Nsym_per_run);
            
            err_ray = err_ray + sum(tx_b ~= rb_r);
            err_ric = err_ric + sum(tx_b ~= rb_ric);
            tot = tot + length(tx_b);
        end
        
        BER_Rayleigh(mod_idx, dop_idx) = max(err_ray/tot, 1e-7);
        BER_Rician(mod_idx, dop_idx) = max(err_ric/tot, 1e-7);
    end
end
fprintf(' Done\n');

%% BER vs SNR - CORRECTED FOR WATERFALL
fprintf('BER vs SNR: ');

for mod_idx = 1:3
    M = M_values(mod_idx);
    k = log2(M);
    fprintf('%s.', modulations{mod_idx});
    
    for snr_idx = 1:n_snr
        SNR_c = 10^(SNR_vec(snr_idx)/10);
        n_runs = ceil(Nbits_sim / (Nsc * k * Nsym_per_run));
        err_ray = 0; err_ric = 0; err_awgn = 0; tot = 0;
        
        for run = 1:n_runs
            tx_b = randi([0, 1], Nsc * k * Nsym_per_run, 1);
            tx_s = qammod(tx_b, M, 'InputType', 'bit', 'UnitAveragePower', true);
            tx_sm = reshape(tx_s, Nsc, Nsym_per_run);
            
            tx_t = zeros(Nsc, Nsym_per_run);
            for s = 1:Nsym_per_run
                tx_t(:, s) = ifft(tx_sm(:, s), Nsc);
            end
            
            tx_cp = zeros(Nsc + CP_len, Nsym_per_run);
            for s = 1:Nsym_per_run
                tx_cp(:, s) = [tx_t(Nsc-CP_len+1:Nsc, s); tx_t(:, s)];
            end
            
            % CORRECTED Rayleigh
            h_r = (randn(1, Nsym_per_run) + 1j*randn(1, Nsym_per_run)) / sqrt(2);
            
            % CORRECTED Rician with proper normalization
            h_los = ones(1, Nsym_per_run);
            h_nlos = (randn(1, Nsym_per_run) + 1j*randn(1, Nsym_per_run)) / sqrt(2);
            h_ric = sqrt(K_rician/(K_rician+1)) * h_los + ...
                    sqrt(1/(K_rician+1)) * h_nlos;
            
            rx_r = zeros(size(tx_cp));
            rx_ric = zeros(size(tx_cp));
            rx_awgn = tx_cp;
            
            for s = 1:Nsym_per_run
                t_s = (s-1)*(Nsc+CP_len)*Ts + (0:Nsc+CP_len-1)*Ts;
                dop = exp(1j*2*pi*fd_fixed*t_s);
                rx_r(:,s) = tx_cp(:,s)*h_r(s).*dop.';
                rx_ric(:,s) = tx_cp(:,s)*h_ric(s).*dop.';
            end
            
            np = 1/SNR_c;
            n = sqrt(np/2)*(randn(Nsc+CP_len, Nsym_per_run)+1j*randn(Nsc+CP_len, Nsym_per_run));
            rx_r = rx_r + n;
            rx_ric = rx_ric + n;
            rx_awgn = rx_awgn + n;
            
            [rb_r] = ofdm_rx(rx_r, h_r, fd_fixed, T_sym, np, Nsc, CP_len, M, Nsym_per_run);
            [rb_ric] = ofdm_rx(rx_ric, h_ric, fd_fixed, T_sym, np, Nsc, CP_len, M, Nsym_per_run);
            [rb_awgn] = ofdm_rx(rx_awgn, ones(1,Nsym_per_run), 0, T_sym, np, Nsc, CP_len, M, Nsym_per_run);
            
            err_ray = err_ray + sum(tx_b ~= rb_r);
            err_ric = err_ric + sum(tx_b ~= rb_ric);
            err_awgn = err_awgn + sum(tx_b ~= rb_awgn);
            tot = tot + length(tx_b);
        end
        
        BER_SNR_Rayleigh(mod_idx, snr_idx) = max(err_ray/tot, 1e-7);
        BER_SNR_Rician(mod_idx, snr_idx) = max(err_ric/tot, 1e-7);
        BER_SNR_AWGN(mod_idx, snr_idx) = max(err_awgn/tot, 1e-7);
    end
end
fprintf(' Done\n');

%% OFDM Receiver Function
function [rx_bits] = ofdm_rx(rx_sig, h_sym, fd, Ts, np, Nsc, CP, M, Ns)
    rx_ncp = rx_sig(CP+1:end, :);
    rx_eq = zeros(Nsc, Ns);
    ICI = (2*pi*fd*Ts)^2/6;
    
    for s = 1:Ns
        rf = fft(rx_ncp(:, s), Nsc);
        H = h_sym(s) * ones(Nsc, 1);
        enp = np + ICI * abs(H).^2;
        w = conj(H) ./ (abs(H).^2 + enp);
        rx_eq(:, s) = rf .* w;
    end
    
    rx_bits = qamdemod(rx_eq(:), M, 'OutputType', 'bit', 'UnitAveragePower', true);
end

%% Performance Plots
colors = {'b', 'r', 'g'};
markers = {'o', 's', '^'};

%% BER vs Doppler
figure('Name', 'BER vs Doppler', 'NumberTitle', 'off', 'Position', [200, 200, 1000, 400]);
subplot(1, 2, 1);
for i = 1:3
    semilogy(doppler_vec, BER_Rayleigh(i, :), [colors{i} markers{i} '-'], ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', colors{i}, ...
             'DisplayName', modulations{i}); 
    hold on;
end
xlabel('Doppler (Hz)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 11, 'FontWeight', 'bold');
title('Rayleigh', 'FontSize', 12, 'FontWeight', 'bold');
legend; grid on; ylim([1e-7, 1]);

subplot(1, 2, 2);
for i = 1:3
    semilogy(doppler_vec, BER_Rician(i, :), [colors{i} markers{i} '--'], ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'w', ...
             'DisplayName', modulations{i}); 
    hold on;
end
xlabel('Doppler (Hz)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 11, 'FontWeight', 'bold');
title('Rician (K=5)', 'FontSize', 12, 'FontWeight', 'bold');
legend; grid on; ylim([1e-7, 1]);
sgtitle('BER vs Doppler (SNR=20dB)', 'FontSize', 14, 'FontWeight', 'bold');

%% BER vs SNR - WITH WATERFALL
figure('Name', 'BER vs SNR - Waterfall Curves', 'NumberTitle', 'off', 'Position', [300, 300, 1100, 750]);
for i = 1:3
    % AWGN
    semilogy(SNR_vec, BER_SNR_AWGN(i, :), [colors{i} 'o-'], ...
             'LineWidth', 2.5, 'MarkerSize', 5, 'MarkerFaceColor', colors{i}, ...
             'DisplayName', [modulations{i} ' (AWGN)']); 
    hold on;
    % Rayleigh
    semilogy(SNR_vec, BER_SNR_Rayleigh(i, :), [colors{i} 's--'], ...
             'LineWidth', 2.5, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
             'DisplayName', [modulations{i} ' (Rayleigh)']); 
    % Rician
    semilogy(SNR_vec, BER_SNR_Rician(i, :), [colors{i} '^:'], ...
             'LineWidth', 2.5, 'MarkerSize', 5, 'MarkerFaceColor', colors{i}, ...
             'DisplayName', [modulations{i} ' (Rician K=5)']); 
end
xlabel('SNR at Receiver (dB)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontSize', 13, 'FontWeight', 'bold');
title('OFDM BER vs SNR with Complete Waterfall Curves (Doppler=100Hz)', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9, 'NumColumns', 3);
grid on; 
ylim([1e-7, 1]);
xlim([0, 50]);

% Add annotations
text(15, 5e-2, 'Fading Margin', 'FontSize', 10, 'FontWeight', 'bold', ...
     'BackgroundColor', 'yellow', 'EdgeColor', 'black');
text(35, 5e-6, 'Waterfall Region', 'FontSize', 10, 'FontWeight', 'bold', ...
     'BackgroundColor', 'cyan', 'EdgeColor', 'black');

fprintf('\n========== COMPLETE ==========\n');
fprintf('Time: %.1f sec\n', toc);
fprintf('Generated 5 figures with proper waterfall curves\n');
