import numpy as np
import torch
import os
from tqdm import tqdm
from scipy import signal
from types import SimpleNamespace
from matplotlib import pyplot as plt
from pathlib import Path
from plot_utils import (
    generate_heatmaps_F3D,
)
LIGHT_SPEED = 299792458  # speed of light in m/s

def heatmap_setup(config):
    num_antennas = config["heatmap_setting"]["num_antenna"]
    num_subcarriers = config["heatmap_setting"]["num_subcarriers"]
    antenna_idx_spacing = config["heatmap_setting"]["antenna_idx_spacing"]
    subcarrier_idx_spacing = config["heatmap_setting"]["subcarrier_idx_spacing"]
    sample_idx_spacing = config["heatmap_setting"]["sample_idx_spacing"]
    f_center = config["heatmap_setting"]["center_frequency"]
    antenna_spacing = config["heatmap_setting"]["antenna_spacing"]
    fs = config["heatmap_setting"]["fs"]
    vector_size = tuple(config["heatmap_setting"]["steering_vector_size"])

    delta_f = 78125 * subcarrier_idx_spacing

    phi_start, phi_end, phi_step = config["heatmap_setting"]["phi_deg_axis_param"]
    phi_deg_axis = np.arange(phi_start, phi_end, phi_step)
    theta_start, theta_end, theta_step = config["heatmap_setting"]["theta_deg_axis_param"]
    theta_deg_axis = np.arange(theta_start, theta_end, theta_step)
    tau_start, tau_end, tau_step = config["heatmap_setting"]["tau_axis_param"]
    tau_axis = np.arange(tau_start, tau_end, tau_step) / LIGHT_SPEED
    f_start, f_end, f_step = config["heatmap_setting"]["f_axis_param"]
    fd_axis = np.arange(f_start, f_end, f_step)

    setting = SimpleNamespace(
        num_antennas			= num_antennas,
        num_subcarriers			= num_subcarriers,
        antenna_idx_spacing		= antenna_idx_spacing,
        vector_size				= vector_size,  # (M, N, T) not equl total subcarrier, antenna and Time window,你可以根據各參數spacing的大小去想要取多少才合理,目前以*spacing後剛好等於目標的一半為主ex:M*sub_spacing <= total subcarriers/2
        subcarrier_idx_spacing	= subcarrier_idx_spacing,
        f_center				= f_center,
        delta_f					= delta_f,
        antenna_spacing			= antenna_spacing,
        sample_idx_spacing		= sample_idx_spacing,
        fs 						= fs,
        phi_deg_axis			= phi_deg_axis,
        theta_deg_axis			= theta_deg_axis,
        tau_axis				= tau_axis,
        fd_axis					= fd_axis
    )
    return setting

def CSI_preprocessing(config,CSI,heatmap_setting):
    # type 1 TWH
    if config["preprocessing_setting"]["type"] == "TWH":
        print("Preprocessing...type:TWH")
        # Phase Sanitization
        x_abs = np.abs(CSI)
        x_abs[np.where(x_abs == 0)] = 1
        sanitized_x= CSI * np.conj(CSI) / x_abs # equl np.abs(x_abs)
        # sanitized_x= x_abs
        
        # # Moving average(remove background)
        # CSI_mean = np.empty_like(sanitized_x)
        # for i in range(sanitized_x.shape[0]):
        # 	# calculate and subtract by the background of the csi sample by moving window
        # 	background_ws = config["preprocessing_setting"]["window_size"] #24
        # 	background_ws_half = (10)//2 # //10
        # 	background_start_idx = max(0, i - background_ws_half)
        # 	background_end_idx = min(sanitized_x.shape[0], i + background_ws_half)
        # 	CSI_mean[i,:,:,:] = np.mean(sanitized_x[background_start_idx:background_end_idx], axis=0, keepdims=True)

        CSI_mean = sanitized_x.astype(np.float64) # need # if remove Moving average above

        # Filtering(bandpass filter)
        cutoffs = [2, 15] # 2,15 for exp 5,27 for sim
        order = 3
        wn = [float(2*cutoff) / config["heatmap_setting"]["fs"] for cutoff in cutoffs]
        b, a = signal.butter(order, wn, 'bandpass', analog = True)
        CSI_filtered = signal.filtfilt(b, a, np.real(CSI_mean), axis=0)

        CSI_filtered = CSI_mean # need # if using Filtering above

        # Moving average(Static Path Elimination)
        CSI_mov = np.zeros_like(CSI_filtered, dtype=np.float64)
        for i in range(CSI_filtered.shape[0]):
            # calculate and subtract by the background of the csi sample by moving window
            background_ws = config["preprocessing_setting"]["window_size"]
            background_ws_half = 50//2
            background_start_idx = max(0, i - background_ws_half)
            background_end_idx = min(CSI_filtered.shape[0], i + background_ws_half + 1)
            CSI_avg = np.mean(CSI_filtered[background_start_idx:background_end_idx], axis=0, keepdims=True)
            CSI_mov[i, :, :, :] = CSI_filtered[i, :, :, :] - CSI_avg

        # Only use tx_0
        CSI_mov = CSI_mov[:,0:1,:,:]

    # type 2 GRY
    elif config["preprocessing_setting"]["type"] == "GRY":
        print("Preprocessing...type:GRY")
        # Compute the power of the CSI
        CSI = np.abs(CSI) ** 2
        CSI = CSI.astype(np.float64)
        # Remove the background signals
        csi_subtracted = np.zeros_like(CSI, dtype=np.float64)
        for i in range(CSI.shape[0]):
            # calculate and subtract by the background of the csi sample by moving window
            background_ws = config["preprocessing_setting"]["window_size"]
            background_ws_half = background_ws//2 #=12
            background_start_idx = max(0, i - background_ws_half)
            background_end_idx = min(CSI.shape[0], i + background_ws_half + 1)
            csi_background = np.mean(CSI[background_start_idx:background_end_idx], axis=0, keepdims=True)
            csi_subtracted[i, :, :, :] = CSI[i, :, :, :] - csi_background
        CSI_mov = csi_subtracted

    else:
        print(f"Unknown Preprocessing type")

    return CSI_mov

def smoothed_CSI(heatmap_type, heatmap_setting, csi_data):
    # csi_data: (timestamps, tx, rx, subcarriers)
    #   for example, (3000, 2, 8, 2025) for 160 MHz or (3000, 2, 8, 489) for 40MHz
    # ant_spacing: antenna spacing (idx)
    # sub_spacing: subcarrier spacing (idx)
    
    # output: (timestamps, tx, num_AoA_heatmap, num_ToF_heatmap, vector_size[0], vector_size[1])
    # NOTE that the selection of num_AoA_heatmap and num_ToF_heatmap are not unique
    # and can be changed based on the scenario

    # here, N and M are the dimensions of the steering vector
    # and are not necessarily the same as the number of antennas and subcarriers
    ant_idx_spacing = heatmap_setting.antenna_idx_spacing
    sub_idx_spacing = heatmap_setting.subcarrier_idx_spacing
    sample_idx_spacing = heatmap_setting.sample_idx_spacing

    M, N, T = heatmap_setting.vector_size
    num_antennas = csi_data.shape[2]  # number of receiver antennas
    num_subcarriers = csi_data.shape[3]  # number of subcarriers
    num_samples = T

    # verify vector_size, ant_spacing, and sub_spacing
    if (N - 1) * ant_idx_spacing + 1 > num_antennas:
        raise ValueError("The number of antennas is not enough for the steering vector size.")
    if (M - 1) * sub_idx_spacing + 1 > num_subcarriers:
        raise ValueError("The number of subcarriers is not enough for the steering vector size.")
    # Try sample spacing in the future

    if 'ToF-Doppler' == heatmap_type:
        T = int((T * (5/2))/sample_idx_spacing)  # adjust T for ToF-Doppler
        # 對時間軸進行邊界值填補 (axis=0), each side has (T*sample_idx_spacing-1)//2 samples
        csi_padded = np.pad(csi_data, 
                       pad_width=(((T * sample_idx_spacing - 1) // 2, T * sample_idx_spacing - 1 - (T * sample_idx_spacing - 1) // 2), (0, 0), (0, 0), (0, 0)),
                       mode='edge')
        csi_smoothed = np.lib.stride_tricks.sliding_window_view(
            csi_padded, 
            window_shape=(M * sub_idx_spacing,2 * T * sample_idx_spacing),
            axis=(3, 0)
        )
        csi_smoothed = np.lib.stride_tricks.sliding_window_view(
            csi_smoothed, 
            window_shape=(T * sample_idx_spacing),
            axis=(-1)
        )
        csi_smoothed = csi_smoothed[:, :, ::ant_idx_spacing, :, ::sub_idx_spacing, :,::sample_idx_spacing]
    elif 'AoA-ToF-Doppler' == heatmap_type:
        T = int((T * (5/2))/sample_idx_spacing)  # Test!!!! adjust T for ToF-Doppler
        # 對時間軸進行邊界值填補 (axis=0), each side has (T*sample_idx_spacing-1)//2 samples
        csi_padded = np.pad(csi_data, 
                       pad_width=(((T * sample_idx_spacing - 1) // 2, T * sample_idx_spacing - 1 - (T * sample_idx_spacing - 1) // 2), (0, 0), (0, 0), (0, 0)),
                       mode='edge')
        csi_smoothed = np.lib.stride_tricks.sliding_window_view(
            csi_padded, 
            window_shape=(N * ant_idx_spacing, M * sub_idx_spacing, T * sample_idx_spacing),
            axis=(2, 3, 0)
        )
        csi_smoothed = csi_smoothed[:, :, :, :, ::ant_idx_spacing, ::sub_idx_spacing, ::sample_idx_spacing]
        # TODO 測試雙層smooth達到time average效果如何
    
    return csi_smoothed

def compute_steering_vector_ToF_Doppler(st_vector_size, tau, fd, delta_f, fs):
    """
    Create a single vector for any specific theta and fd

    Params:
    - vector_size : [M,T]
    - theta_deg : AoA(deg)
    - fd : Doppler shift(Hz)
    - delta_f : subcarrier spacing (Hz)
    - fs : sample frequency (Hz)
    """
    
    subcarrier_indices = np.arange(st_vector_size[0]) # (0, subcarrier_window_size)
    sample_indices = np.arange(st_vector_size[1]) # (0, time_window_size)

    
    # Phase due to ToF (subcarrier)
    phase_tof = 2 * np.pi * delta_f * subcarrier_indices[:, None] * tau  # (M , 1)

    # Phase due to Doppler shift (temperal)
    phase_Doppler = -2 * np.pi * fd * sample_indices[None, :] / fs   # (1, T)

    total_phase = phase_tof + phase_Doppler  # broadcast to (M, T)

    # From GRY
    steering_vector = np.exp(-1j * total_phase) / np.sqrt(st_vector_size[0] * st_vector_size[1])  # normalize


    # (M, T) -> (M * T, 1)
    steering_vector = steering_vector.reshape(-1, 1)  # (M * T, 1)
    return steering_vector.astype(np.complex64)

def create_steering_matrix_ToF_Doppler(heatmap_setting):
    """
    Create 2D ToF-Doppler steering matrix for all (tau, fd) pairs.

    Params:
    - st_vector_size: (M, T), not necessarily the same as the number of antennas and subcarriers
    - tau_axis: array of tau (sec), shape (num_tau,)
    - fd_axis: array of doppler shift (Hz), shape (num_fd,)
    """
    # steering vectors
    tau = heatmap_setting.tau_axis
    fd = heatmap_setting.fd_axis
    
    delta_f = heatmap_setting.delta_f
    fs = heatmap_setting.fs
    M, N, T = heatmap_setting.vector_size
    T = int((T * (5/2))/heatmap_setting.sample_idx_spacing)  # adjust T for ToF-Doppler

    steering_matrix = np.empty((tau.size, fd.size, M * T), dtype=np.complex64)

    # computing steering vectors
    for i in range(len(tau)):
        for j in range(len(fd)):		
            # compute steering vector and reshape it (M, 1)
            steering_matrix[i, j, :] = compute_steering_vector_ToF_Doppler((M, T), tau = tau[i], fd = fd[j], delta_f = delta_f, fs = fs)[:, 0]
    
    steering_matrix = steering_matrix.reshape(tau.size * fd.size, -1)
    return steering_matrix

def compute_steering_vector_F3D(st_vector_size, theta_deg, tau, fd, f_center, fs, delta_f, d):
    """
    Create a single vector for any specific theta, tau and fd

    Params:
    - vector_size : [M,N,T]
    - theta_deg : AoA(deg)
    - tau : ToF(sec)
    - fd : Doppler shift(Hz)
    - d : antenna spacing (m)
    - f_c : center frequency(Hz)
    - delta_f : subcarrier spacing (Hz)
    - fs : sample frequency (Hz)
    """
    
    antenna_indices = np.arange(st_vector_size[1]) # (0, stream_window_size)
    subcarrier_indices = np.arange(st_vector_size[0]) # (0, subcarrier_window_size)
    sample_indices = np.arange(st_vector_size[2]) # (0, time_window_size)
    
    theta_rad = np.deg2rad(theta_deg)

    
    # Phase due to AoA (spatial)
    phase_aoa = 2 * np.pi * f_center * d * antenna_indices[:, None, None] * np.sin(theta_rad) / LIGHT_SPEED  # (N , 1 , 1)

    # Phase due to ToF (subcarrier)
    phase_tof = 2 * np.pi * delta_f * subcarrier_indices[None, :, None] * tau  # (1, M, 1)

    # Phase due to Doppler shift (temperal)
    phase_Doppler = -2 * np.pi * fd * sample_indices[None, None, :] / fs   # (1, 1, T)

    total_phase = phase_aoa + phase_tof + phase_Doppler  # broadcast to (N, T)

    # From GRY
    steering_vector = np.exp(-1j * total_phase) / np.sqrt(st_vector_size[0] * st_vector_size[1] * st_vector_size[2])  # normalize

    # (N, M, T) -> (N * M * T, 1)
    steering_vector = steering_vector.reshape(-1, 1)  # (N * M * T, 1)
    return steering_vector.astype(np.complex64)

def create_steering_matrix_F3D(heatmap_setting):
    """
    Create 3D AoA-ToF-FD steering matrix for all (theta, tau, fd) pairs.

    Params:
    - st_vector_size: (N, T), not necessarily the same as the number of antennas and subcarriers
    - theta_deg_axis: array of theta (degrees), shape (num_theta,)
    - fd_axis: array of doppler shift (Hz), shape (num_fd,)
    """
    # steering vectors
    theta_deg = heatmap_setting.theta_deg_axis
    tau = heatmap_setting.tau_axis
    fd = heatmap_setting.fd_axis
    
    f_center = heatmap_setting.f_center
    fs = heatmap_setting.fs
    delta_f = heatmap_setting.delta_f
    d = heatmap_setting.antenna_spacing
    M, N, T = heatmap_setting.vector_size

    T = int((T * (5/2)))  # adjust T for ToF-Doppler

    steering_matrix = np.empty((theta_deg.size, tau.size, fd.size, N * M * T), dtype=np.complex64)

    # computing steering vectors
    for i in range(len(theta_deg)):
        for j in range(len(tau)):
            for k in range(len(fd)):
                # compute steering vector and reshape it (M, 1)
                steering_matrix[i, j, k, :] = compute_steering_vector_F3D((M, N, T), theta_deg = theta_deg[i], tau = tau[j], fd = fd[k], f_center = f_center, fs = fs, delta_f = delta_f, d = d)[:, 0]
    
    steering_matrix = steering_matrix.reshape(theta_deg.size * tau.size * fd.size, -1)
    return steering_matrix

def calculate_correlation_matrix(csi_smoothed, s_idx=None, heatmap_type = None):
    # csi_smoothed: (timestamps, tx, num_AoA_heatmap, num_ToF_heatmap, N, M)

    if heatmap_type == "ToF-Doppler":
        print("ToF_Doppler average over streams")
        csi_smoothed = np.transpose(csi_smoothed,(0,1,2,3,5,4,6))
        num_heatmaps = csi_smoothed.shape[1] * csi_smoothed.shape[2] * csi_smoothed.shape[3] * csi_smoothed.shape[4]
        if csi_smoothed.dtype != np.complex64:
            # After phase sanitization
            R = np.zeros((csi_smoothed.shape[0],csi_smoothed.shape[-2]*csi_smoothed.shape[-1],csi_smoothed.shape[-2]*csi_smoothed.shape[-1]))
        else:
            # Test for presurve phase
            R = np.zeros((csi_smoothed.shape[0],csi_smoothed.shape[-2]*csi_smoothed.shape[-1],csi_smoothed.shape[-2]*csi_smoothed.shape[-1]),dtype=np.complex64)
        for idx in tqdm(range(csi_smoothed.shape[0])):
            csi_smoothed_rt = csi_smoothed[idx].reshape( # remove timestamps
                csi_smoothed.shape[1] * csi_smoothed.shape[2] * csi_smoothed.shape[3] * csi_smoothed.shape[4], 	# tx * num_streams * num_subcarriers_smoothed * num_time_smoothed = num_heatmaps
                csi_smoothed.shape[5] * csi_smoothed.shape[6]							# M * T (vector_size)
            )
            # seg(range of average. you can try other number) already count in vector_size and smoothed
            R_i = np.einsum('sj,sk->jk', csi_smoothed_rt, np.conj(csi_smoothed_rt), optimize=True)
            R[idx] = R_i / num_heatmaps  # normalize
    else:
        print("No avalible correlation type")

    # # GRY type correlation (No matter what type of heatmap)
    # num_heatmaps = csi_smoothed.shape[1] * csi_smoothed.shape[2] * csi_smoothed.shape[3]
    # csi_smoothed = csi_smoothed.reshape(
    # 	csi_smoothed.shape[0], 													# timestamps
    # 	csi_smoothed.shape[1] * csi_smoothed.shape[2] * csi_smoothed.shape[3], 	# tx * num_AoA_heatmap * num_ToF_heatmap
    # 	csi_smoothed.shape[4] * csi_smoothed.shape[5]							# N * M
    # )
    # if s_idx is None:
    # 	# average over tx, num_AoA_heatmap, and num_ToF_heatmap
    # 	R = np.einsum('tsj,tsk->tjk', csi_smoothed, np.conj(csi_smoothed), optimize=True)
    # 	R = R / num_heatmaps  # normalize
    # else:
    # 	# don't average over tx, num_AoA_heatmap, and num_ToF_heatmap
    # 	# instead, get the s_idx'th one
    # 	R = np.einsum('tj,tk->tjk', csi_smoothed[:, s_idx, :], np.conj(csi_smoothed[:, s_idx, :]), optimize=True)
    return R

def calculate_correlation_matrix_F3D(csi_smoothed, s_idx=None):
    # csi_smoothed: (tx, num_AoA_heatmap, num_ToF_heatmap, N, M)
    # calculate correlation matrix without weighting
    num_heatmaps = csi_smoothed.shape[0] * csi_smoothed.shape[1] * csi_smoothed.shape[2]
    csi_smoothed = csi_smoothed.reshape(
        csi_smoothed.shape[0] * csi_smoothed.shape[1] * csi_smoothed.shape[2], 	# tx * num_AoA_heatmap * num_ToF_heatmap
        csi_smoothed.shape[3] * csi_smoothed.shape[4] * csi_smoothed.shape[5]							# N * M
    )

    # GRY type correlation
    if s_idx is None:
        # average over tx, num_AoA_heatmap, and num_ToF_heatmap
        R = np.einsum('sj,sk->jk', csi_smoothed, np.conj(csi_smoothed), optimize=True)
        R = R / num_heatmaps  # normalize
    # else:
    # 	# don't average over tx, num_AoA_heatmap, and num_ToF_heatmap
    # 	# instead, get the s_idx'th one
    # 	R = np.einsum('tj,tk->tjk', csi_smoothed[:, s_idx, :], np.conj(csi_smoothed[:, s_idx, :]), optimize=True)
    return R

def run_music_algorithm(R, steering_matrix,heatmap_type = None):
    # R: (..., N, N)
    # steering_matrix: (num_axis, N)
    # calculate eigenvalues and eigenvectors

    epsilon = 1e-5  # small value to avoid division by zero
    # fixed number of signal subspace TODO  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_signal_subspace = 20  # fixed number of signal subspace
    if heatmap_type == "AoA-ToF-Doppler":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        R_gpu = torch.as_tensor(np.ascontiguousarray(R, dtype=np.complex64)).pin_memory().to(device, non_blocking=True)
        eigenvalues_gpu, eigenvectors_gpu = torch.linalg.eigh(R_gpu)

        Es_gpu = eigenvectors_gpu.flip(-1)[..., :20]  

        # steering matrix 
        sm_gpu = torch.as_tensor(
            np.ascontiguousarray(steering_matrix, dtype=np.complex64)
        ).pin_memory().to(device, non_blocking=True)

        aE = torch.einsum('ai,...ij->...aj', sm_gpu.conj(), Es_gpu)
        aEEa = torch.einsum('...aj,...aj->...a', aE, aE.conj()).real

        aEEa = torch.clamp(aEEa, -1, 1)
        P = 10 * torch.log10(1 / (1 - aEEa + epsilon))
        P = P.cpu().numpy()
    # Original No GPU version
    # elif heatmap_type == "F3D":
    #     # calculate eigenvalues and eigenvectors
    #     eigenvalues, eigenvectors = np.linalg.eigh(R)
    #     # eigenvalues, eigenvectors = np.linalg.eigh(R)
    #     # sort eigenvalues and eigenvectors (along the last dimention)
    #     idx = np.argsort(eigenvalues, axis=-1)[..., ::-1]  # sort in descending order
    #     eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
    #     eigenvectors = np.take_along_axis(eigenvectors, np.expand_dims(idx, axis=-2), axis=-1)
    #     # where eigenvector[..., :, i] is the i-th eigenvector of the i-th eigenvalue eigenvalues[..., i]

    #     # fixed number of signal subspace TODO  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     num_signal_subspace = 20  # fixed number of signal subspace
    #     Es = eigenvectors[..., :num_signal_subspace]  # (samples, N, num_signal_subspace)

    #     # instead of calculate inner product between the steering vector and the noise subspace
    #     # we calculate the inner product between the steering vector and the signal subspace
    #     # and then subtract from 1
    #     epsilon = 1e-5  # small value to avoid division by zero
    #     aE = np.einsum('ai,...ij->...aj', steering_matrix.conj(), Es, optimize=True)
    #     aEEa = np.einsum('...aj,...aj->...a', aE, aE.conj(), optimize=True)  # (samples, num_axis)
    #     aEEa = np.real(aEEa) # (..., num_axis)

    #     # prevent numerical error due to floating point precision, should <= 1
    #     aEEa = np.clip(aEEa, -1, 1)

    #     # calculate the power spectrum
    #     P = 1 / (1 - aEEa + epsilon)
    #     P = 10 * np.log10(P)  # convert to dB

    else:
        batch_size = 25 # For saving memories
        P = np.empty((R.shape[0],steering_matrix.shape[0]))
        print("Calculating MUSIC algorithm...")
        for start in tqdm(range(0,R.shape[0],batch_size)):
            
            end = min(start + batch_size, R.shape[0])
            # calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(R[start:end])
            # sort eigenvalues and eigenvectors (along the last dimention)
            idx = np.argsort(eigenvalues, axis=-1)[..., ::-1]  # sort in descending order
            eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)		
            eigenvectors = np.take_along_axis(eigenvectors, np.expand_dims(idx, axis=-2), axis=-1)
            # where eigenvector[..., :, i] is the i-th eigenvector of the i-th eigenvalue eigenvalues[..., i]
            Es = eigenvectors[..., :num_signal_subspace]  # (samples, N, num_signal_subspace)
            # instead of calculate inner product between the steering vector and the noise subspace
            # we calculate the inner product between the steering vector and the signal subspace
            # and then subtract from 1
            aE = np.einsum('ai,...ij->...aj', steering_matrix.conj(), Es, optimize=True)
            aEEa = np.einsum('...aj,...aj->...a', aE, aE.conj(), optimize=True)  # (samples, num_axis)
            aEEa = np.real(aEEa) # (..., num_axis)

            # prevent numerical error due to floating point precision, should <= 1
            aEEa = np.clip(aEEa, -1, 1)

            # calculate the power spectrum
            P[start:end,:] = 1 / (1 - aEEa + epsilon)
        P = 10 * np.log10(P)  # convert to dB
    
    return P

def pipeline_3D(exp_name,args, CSI_smoothed, steering_matrix_3D, heatmap_setting, output_folder = "/home/tonic/guan125/1exp_data/20250512/heatmap_output/figure", filename_prefix = "3D_spectrum",
                 visualization_method="slice_avg_2d"# avalible:"plotly_3d" "matplotlib_3d" "slice_avg_2d" "contour_3d"
                 ,save_format="html",start_sample=None, end_sample=None, plot_gt=False):
        
    total_samples = CSI_smoothed.shape[0]

    if start_sample is None:
        start_sample = 0
    
    if start_sample >= total_samples:
        raise ValueError(f"start_sample ({start_sample}) out of bounds ({total_samples})")

    if end_sample is None:
        end_sample = total_samples
    else:
        end_sample = min(end_sample, total_samples)

    if start_sample >= end_sample:
        raise ValueError(f"start_sample ({start_sample}) must be less than end_sample ({end_sample})")

    type_folder = "F3D"
    
    if args.save_fig:
        fig_path = Path(args.save_path)/'heatmap_result'/'figures'/exp_name/type_folder/visualization_method
        if os.path.exists(fig_path):
            print(f"{fig_path} exist, skipping creation...")
        else:
            fig_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 fig ouput folder: {fig_path.absolute()}")
    if args.save_mat:
        mat_npz_path = Path(args.save_path)/'heatmap_result'/'mat'/exp_name/type_folder
        temp_dir = mat_npz_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        if os.path.exists(mat_npz_path):
            print(f"{mat_npz_path} exist, skipping creation...")
        else:
            mat_npz_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 mat ouput folder: {mat_npz_path.absolute()}")

    print(f"Processing: sample {start_sample} to {end_sample-1} (total {end_sample-start_sample} samples)")

    print("Generating 3D AoA-ToF-FD heatmap:")
    print(f"- Vector size:M = {heatmap_setting.vector_size[0]}, N = {heatmap_setting.vector_size[1]}, T = {heatmap_setting.vector_size[2]}")
    print(f"- subcarrier spacing:{heatmap_setting.subcarrier_idx_spacing}")
    print(f"- scanning range AoA:{heatmap_setting.theta_deg_axis[0]} to {heatmap_setting.theta_deg_axis[-1]} degrees")
    print(f"- scanning range ToF:{heatmap_setting.tau_axis[0]} to {heatmap_setting.tau_axis[-1]} seconds")
    print(f"- scanning range FD:{heatmap_setting.fd_axis[0]} to {heatmap_setting.fd_axis[-1]} Hz")

    for sample_idx in tqdm(range(start_sample, end_sample, 1)): 
        # get the correlation matrix R
        # output:(timestamp, N*M*T, N*M*T)
        R = calculate_correlation_matrix_F3D(CSI_smoothed[sample_idx,:,:,:,:,:])

        # get spectrums
        spectrums = run_music_algorithm(R, steering_matrix_3D, heatmap_type="AoA-ToF-Doppler")
        spectrums = spectrums.reshape(heatmap_setting.theta_deg_axis.shape[0], heatmap_setting.tau_axis.shape[0],heatmap_setting.fd_axis.shape[0])
        spectrums = spectrums.transpose(1,0,2) # keep (tau, theta, fd)
        # save as figure
        generate_heatmaps_F3D(sample_idx,spectrums=spectrums, heatmap_setting=heatmap_setting, 
                        filename_prefix = filename_prefix,save_format = save_format,
                        visualization_method = visualization_method,
                        threshold_percentile = 1, alpha = 0.05,
                        figsize = (12, 10),output_path = fig_path,plot_gt=plot_gt)
        if args.save_mat:
            temp_file = temp_dir / f"spectrum_{sample_idx:06d}.npy"
            np.save(temp_file, spectrums)
    
    if args.save_mat:
        
        temp_files = sorted(temp_dir.glob("spectrum_*.npy"))
        
        if len(temp_files) > 0:
            first_spectrum = np.load(temp_files[0])
            tau_size, theta_size, fd_size = first_spectrum.shape
            time_size = len(temp_files)
            
            # (time, tau, theta, fd)
            all_spectrums = np.zeros((time_size, tau_size, theta_size, fd_size), dtype=first_spectrum.dtype)
            
            for i, temp_file in enumerate(tqdm(temp_files, desc="Loading spectrums")):
                all_spectrums[i] = np.load(temp_file)
            
            # Save as .npz file
            npz_filename = mat_npz_path / f"{filename_prefix}_spectrums.npz"
            np.savez_compressed(npz_filename, spectrums=all_spectrums)
            print(f"✅ NPZ file saved: {npz_filename}")
            
            # # Save as .mat file
            # mat_filename = mat_npz_path / f"{filename_prefix}_spectrums.mat"
            # savemat(mat_filename, {'spectrums': all_spectrums})
            # print(f"✅ MAT file saved: {mat_filename}")
            
            for temp_file in temp_files:
                temp_file.unlink()
            temp_dir.rmdir()
            
            print(f"📊 final shape: {all_spectrums.shape} (time, tau, theta, fd)")
        else:
            print("⚠️  No temporary spectrum files found.")

    result_info = {
        'total_requested': end_sample - start_sample,
        'output_folder': str(fig_path.absolute()),
        'sample_range': (start_sample, end_sample-1),
        'file_format': save_format,
        'visualization_method': visualization_method,
    }

    print(f"\n✅ Completion!")
    print(f"   Output location: {fig_path.absolute()}")
    return result_info