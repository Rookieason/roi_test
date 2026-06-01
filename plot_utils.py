import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import os

def generate_heatmaps(exp_name,heatmap_type,spectrum, heatmap_setting,
                         start_sample=None, end_sample=None, 
                         output_folder="/home/tonic/guan125/1exp_data/20250512/heatmap_output/figure",
                         filename_prefix=None,
                         save_format="png",
                         figsize=(10, 8),
                         cmap='jet',
                         vmin=None, vmax=None,
                         transpose_data=False,plot_gt=False):

    if 'ToF-Doppler' == heatmap_type:
        spectrum = spectrum.reshape(spectrum.shape[0], heatmap_setting.tau_axis.shape[0], heatmap_setting.fd_axis.shape[0])
        x_axis_data = heatmap_setting.fd_axis
        x_axis_name = "Doppler Shift"
        x_axis_unit = "(Hz)"
        y_axis_data = heatmap_setting.tau_axis
        y_axis_name = "ToF"
        y_axis_unit = "(s)"
        default_prefix = "tof_Doppler_spectrum"
        title_template = "ToF_Doppler Spectrum - Sample {}"
        type_folder = "ToF-Doppler"
    
    if filename_prefix is None:
        filename_prefix = default_prefix
    
    total_samples = spectrum.shape[0]
    
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
    
    output_path = Path(output_folder)/"heatmap_result"/"figures"/exp_name/type_folder
    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output folder: {output_path.absolute()}")
    # print(f"🎯 Processing type: {args.heatmap_type}")
    print(f"🎯 Processing range: sample {start_sample} to {end_sample-1} (total {end_sample-start_sample} samples)")
    # print(f"📊 Data shape: {spectrum.shape}")
    print(f"📏 X-axis ({x_axis_name}): {len(x_axis_data)} points, range: {x_axis_data[0]:.2e} ~ {x_axis_data[-1]:.2e}")
    print(f"📏 Y-axis ({y_axis_name}): {len(y_axis_data)} points, range: {y_axis_data[0]:.1f}° ~ {y_axis_data[-1]:.1f}°")
    
    x_range = [x_axis_data[0], x_axis_data[-1]]
    y_range = [y_axis_data[0], y_axis_data[-1]]
    
    processed_count = 0
    failed_count = 0
    

    for sample_idx in tqdm(range(start_sample, end_sample , 1), #30fps match video
                          desc=f"生成進度", 
                          unit="samples"):
        try:
            plt.figure(figsize=figsize)
            
            if transpose_data:
                current_data = spectrum[sample_idx, :, :].T
            else:
                current_data = spectrum[sample_idx, :, :]
            
            im = plt.imshow(current_data, 
                           aspect='auto', 
                           extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                           origin='lower', 
                           cmap=cmap, 
                           vmin=vmin, 
                           vmax=vmax)
            
            plt.xlabel(f'{x_axis_name} {x_axis_unit}')
            plt.ylabel(f'{y_axis_name} {y_axis_unit}')
            plt.title(title_template.format(sample_idx))
            
            cbar = plt.colorbar(im, label='Power (dB)')
            
            

            # 針對不同類型調整顯示格式
            if plot_gt:
                if "ToF-Doppler" == heatmap_type:
                    doppler_range = abs(x_axis_data[-1] - x_axis_data[0])
                    if doppler_range > 1000:
                        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
                    # Plot each object (column) with different color
                    for obj_idx in range(num_obj):
                        # Get data for this object (column)
                        obj_data_x = fds_gt_data.iloc[sample_idx, obj_idx]
                        obj_data_y = tof_gt_data.iloc[sample_idx, obj_idx]
                        plt.plot(obj_data_x, 
                                obj_data_y, 
                                color=colors[obj_idx], 
                                linewidth=2, 
                                marker='*', 
                                markersize=5,
                                label=f'Ground Truth User {obj_idx+1}')
            

            filename = f"{sample_idx}.{save_format}"
            filepath = output_path / filename
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Sample {sample_idx} failed: {str(e)}")
            failed_count += 1
            plt.close()
    
    result_info = {
        'heatmap_type': heatmap_type,
        'total_requested': end_sample - start_sample,
        'processed': processed_count,
        'failed': failed_count,
        'output_folder': str(output_path.absolute()),
        'sample_range': (start_sample, end_sample-1),
        'file_format': save_format,
        'x_axis_info': {
            'name': x_axis_name,
            'unit': x_axis_unit,
            'range': x_range,
            'points': len(x_axis_data)
        },
        'y_axis_info': {
            'name': y_axis_name,
            'unit': y_axis_unit, 
            'range': y_range,
            'points': len(y_axis_data)
        }
    }
    
    print(f"\n✅ {heatmap_type} Completed!")
    print(f"   Success: {processed_count} samples")
    print(f"   Failed: {failed_count} samples")
    print(f"   Output Location: {output_path.absolute()}")
    
    return result_info

def generate_heatmaps_F3D(sample_idx, spectrums, heatmap_setting,output_path, plot_gt,aoa_gt=None,tof_gt=None,fds_gt=None,
                        filename_prefix="3D_spectrum",
                        save_format="html",
                        visualization_method="plotly_3d",
                        threshold_percentile=50,
                        alpha=0.6,
                        figsize=(12, 10)):
    tau_grid, theta_grid, fd_grid = np.meshgrid(
        heatmap_setting.tau_axis,
        heatmap_setting.theta_deg_axis, 
        heatmap_setting.fd_axis,
        indexing='ij'
    )
    
    processed_count = 0
    failed_count = 0

    try:
        current_data = spectrums
        
        if visualization_method == "plotly_3d":
            success = _create_plotly_3d(heatmap_setting,
                current_data, tau_grid, theta_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "matplotlib_3d":
            success = _create_matplotlib_3d(
                current_data, tau_grid, theta_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha, figsize
            )
        elif visualization_method == "slice_avg_2d":
            success = _create_slice_avg_2d(
                current_data, heatmap_setting,
                sample_idx, output_path, filename_prefix, 
                figsize,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "contour_3d":
            success = _create_contour_3d(
                current_data, tau_grid, theta_grid, fd_grid,
                sample_idx, output_path, filename_prefix, 
                save_format, figsize
            )
        else:
            raise ValueError(f"不支援的可視化方法: {visualization_method}")
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
            
    except Exception as e:
        print(f"❌ Sample {sample_idx} 處理失敗: {str(e)}")
        failed_count += 1
    
    return 

def generate_heatmaps_S3D(sample_idx, spectrums, heatmap_setting,output_path, plot_gt,aoa_gt=None,tof_gt=None,fds_gt=None,
                        filename_prefix="S3D_spectrum",
                        save_format="html",
                        visualization_method="plotly_3d",
                        threshold_percentile=50,
                        alpha=0.6,
                        figsize=(12, 10)):
    """
    批次生成 AoD-AoA-ToF 3D heatmap 並存檔
    
    Parameters:
    -----------
    spectrum : numpy.ndarray
        頻譜數據，形狀為 (samples, phi, theta, tau)
    heatmap_setting : object
        包含 tau_axis, theta_deg_axis, fd_axis 的設定物件
    start_sample : int, default=200
        開始的 sample 索引
    end_sample : int, optional
        結束的 sample 索引，若為 None 則處理到最後
    output_folder : str
        輸出資料夾路徑
    filename_prefix : str, default="aoa_tof_fd_spectrum"
        檔案名稱前綴
    save_format : str, default="html"
        儲存格式 ("html", "png", "pdf" for interactive; "png", "jpg", "pdf" for static)
    visualization_method : str, default="plotly_3d"
        可視化方法: "plotly_3d", "matplotlib_3d", "slice_2d", "contour_3d"
    threshold_percentile : float, default=50
        顯示數據的百分位閾值 (只顯示高於此百分位的點)
    alpha : float, default=0.6
        透明度 (0-1)
    figsize : tuple, default=(12, 10)
        圖片大小 (僅用於matplotlib)
    interactive : bool, default=True
        是否生成交互式圖表
    
    Returns:
    --------
    dict : 包含處理資訊的字典
    """
    # 建立座標網格
    phi_grid, theta_grid, tau_grid = np.meshgrid(
        heatmap_setting.phi_deg_axis,
        heatmap_setting.theta_deg_axis,
        heatmap_setting.tau_axis,
        indexing='ij'
    )
    
    processed_count = 0
    failed_count = 0

    try:
        current_data = spectrums
        
        if visualization_method == "plotly_3d":
            success = _create_plotly_3d(heatmap_setting,
                current_data, tau_grid, theta_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "matplotlib_3d":
            success = _create_matplotlib_3d(
                current_data, tau_grid, theta_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha, figsize
            )
        elif visualization_method == "slice_avg_2d":
            success = _create_slice_avg_2d_S3D(
                current_data, heatmap_setting,
                sample_idx, output_path, filename_prefix, 
                figsize,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "contour_3d":
            success = _create_contour_3d(
                current_data, tau_grid, theta_grid, fd_grid,
                sample_idx, output_path, filename_prefix, 
                save_format, figsize
            )
        else:
            raise ValueError(f"不支援的可視化方法: {visualization_method}")
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
            
    except Exception as e:
        print(f"❌ Sample {sample_idx} 處理失敗: {str(e)}")
        failed_count += 1
    
    return 

def generate_heatmaps_F4D(sample_idx, spectrums, heatmap_setting,output_path, plot_gt,aoa_gt=None,tof_gt=None,fds_gt=None,
                        filename_prefix="4D_spectrum",
                        save_format="html",
                        visualization_method="plotly_3d",
                        threshold_percentile=50,
                        alpha=0.6,
                        figsize=(12, 10)):
    """
    批次生成 AoD-AoA-ToF-fd 4D heatmap 並存檔
    
    Parameters:
    -----------
    spectrum : numpy.ndarray
        頻譜數據，形狀為 (phi, theta, tau, fd)
    heatmap_setting : object
        包含 tau_axis, theta_deg_axis, fd_axis 的設定物件
    output_folder : str
        輸出資料夾路徑
    filename_prefix : str, default="4D_spectrum"
        檔案名稱前綴
    save_format : str, default="html"
        儲存格式 ("html", "png", "pdf" for interactive; "png", "jpg", "pdf" for static)
    visualization_method : str, default="plotly_3d"
        可視化方法: "plotly_3d", "matplotlib_3d", "slice_2d", "contour_3d"
    threshold_percentile : float, default=50
        顯示數據的百分位閾值 (只顯示高於此百分位的點)
    alpha : float, default=0.6
        透明度 (0-1)
    figsize : tuple, default=(12, 10)
        圖片大小 (僅用於matplotlib)
    interactive : bool, default=True
        是否生成交互式圖表
    
    Returns:
    --------
    dict : 包含處理資訊的字典
    """
    # 建立座標網格
    phi_grid, theta_grid, tau_grid, fd_grid = np.meshgrid(
        heatmap_setting.phi_deg_axis,
        heatmap_setting.theta_deg_axis,
        heatmap_setting.tau_axis, 
        heatmap_setting.fd_axis,
        indexing='ij'
    )
    
    processed_count = 0
    failed_count = 0

    try:
        current_data = spectrums
        
        if visualization_method == "plotly_3d":
            success = _create_plotly_3d_F4D(heatmap_setting,
                current_data, phi_grid, theta_grid, tau_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "matplotlib_3d":
            success = _create_matplotlib_3d(
                current_data, tau_grid, theta_grid, fd_grid, 
                sample_idx, output_path, filename_prefix, 
                save_format, threshold_percentile, alpha, figsize
            )
        elif visualization_method == "slice_avg_2d":
            success = _create_slice_avg_2d_F4D(
                current_data, heatmap_setting,
                sample_idx, output_path, filename_prefix, 
                figsize,plot_gt,aoa_gt,tof_gt,fds_gt
            )
        elif visualization_method == "contour_3d":
            success = _create_contour_3d(
                current_data, tau_grid, theta_grid, fd_grid,
                sample_idx, output_path, filename_prefix, 
                save_format, figsize
            )
        else:
            raise ValueError(f"不支援的可視化方法: {visualization_method}")
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
            
    except Exception as e:
        print(f"❌ Sample {sample_idx} 處理失敗: {str(e)}")
        failed_count += 1
    
    
    
    return 

# def _create_plotly_3d(heatmap_setting,data, tau_grid, theta_grid, fd_grid, sample_idx, 
#                      output_path, filename_prefix, save_format, 
#                      threshold_percentile, alpha,plot_gt,aoa_gt,tof_gt,fds_gt):
#     """使用Plotly創建交互式3D散點圖"""
#     if plot_gt:
#         # Get number of objects (columns)
#         num_obj = aoa_gt.shape[0]
#         # Define colors for different objects
#         # Detected peaks use default blue, so use warm colors for ground truth
#         colors = plt.cm.Set1(np.linspace(0, 1, num_obj))
#     try:
#         # 計算閾值
#         threshold = np.percentile(data, threshold_percentile)
        
#         # 找出高於閾值的點
#         mask = data > threshold
        
#         if not np.any(mask):
#             print(f"⚠️ Sample {sample_idx}: 沒有數據點高於閾值")
#             return False
        
#         # 提取座標和數值
#         tau_points = tau_grid[mask]
#         theta_points = theta_grid[mask]
#         fd_points = fd_grid[mask]
#         values = data[mask]
        
#         # 創建3D散點圖
#         fig = go.Figure(data=go.Scatter3d(
#             x=tau_points,
#             y=theta_points, 
#             z=fd_points,
#             mode='markers',
#             marker=dict(
#                 size=3,
#                 color=values,
#                 colorscale='Viridis',
#                 opacity=alpha,
#                 colorbar=dict(title="Power (dB)")
#             ),
#             text=[f'ToF: {t:.2e}s<br>AoA: {a:.1f}°<br>FD: {f:.1f}Hz<br>Power: {v:.2f}dB' 
#                   for t, a, f, v in zip(tau_points, theta_points, fd_points, values)],
#             hovertemplate='%{text}<extra></extra>'
#         ))
        
#         # 設定固定的軸範圍
#         scene_config = dict(
#             xaxis_title='ToF (s)',
#             yaxis_title='AoA (degrees)',
#             zaxis_title='Doppler Shift (Hz)'
#         )
#         scene_config.update({
#                 'xaxis': dict(title='ToF (s)', range=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1]]),           # tau_axis_param
#                 'yaxis': dict(title='AoA (degrees)', range=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]]),    # theta_deg_axis_param  
#                 'zaxis': dict(title='Doppler Shift (Hz)', range=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1]])  # f_axis_param
#             })
#         fig.update_layout(
#             title=f'3D AoA-ToF-FD Spectrum - Sample {sample_idx}',
#             scene=scene_config,
#             width=1200,
#             height=800
#         )
        
#         # 儲存檔案
#         filename = f"{sample_idx}.{save_format}"
#         filepath = output_path / filename
        
#         if save_format == 'html':
#             fig.write_html(filepath)
#         else:
#             fig.write_image(filepath)
        
#         return True
        
#     except Exception as e:
#         print(f"Plotly 3D 生成失敗: {e}")
#         return False

def _create_plotly_3d(heatmap_setting, data, tau_grid, theta_grid, fd_grid, sample_idx, 
                     output_path, filename_prefix, save_format, 
                     threshold_percentile, alpha, plot_gt, aoa_gt, tof_gt, fds_gt):
    """使用Plotly創建交互式3D散點圖，包含ground truth顯示"""    
    try:
        # 計算閾值
        threshold = np.percentile(data, threshold_percentile)
        
        # 找出高於閾值的點
        mask = data > threshold
        
        if not np.any(mask):
            print(f"⚠️ Sample {sample_idx}: 沒有數據點高於閾值")
            return False
        
        # 提取座標和數值
        tau_points = tau_grid[mask]
        theta_points = theta_grid[mask]
        fd_points = fd_grid[mask]
        values = data[mask]
        
        # 創建3D散點圖 - 調整透明度使其更透明
        fig = go.Figure(data=go.Scatter3d(
            x=tau_points,
            y=theta_points, 
            z=fd_points,
            mode='markers',
            marker=dict(
                size=3,
                color=values,
                colorscale='Viridis',
                opacity=alpha,  # 大幅降低透明度，使散點更透明
                colorbar=dict(title="Power (dB)")
            ),
            text=[f'ToF: {t:.2e}s<br>AoA: {a:.1f}°<br>FD: {f:.1f}Hz<br>Power: {v:.2f}dB' 
                  for t, a, f, v in zip(tau_points, theta_points, fd_points, values)],
            hovertemplate='%{text}<extra></extra>',
            name='Detected Peaks'
        ))
        
        # 添加ground truth標記
        if plot_gt and aoa_gt is not None and tof_gt is not None and fds_gt is not None:
            # 獲取對象數量
            num_obj = len(aoa_gt)
            
            # 為不同對象定義顏色（使用更鮮明的顏色以便與散點區分）
            colors = plt.cm.Set1(np.linspace(0, 1, max(num_obj, 8)))  # 確保至少有8種顏色可選
            
            for obj_idx in range(num_obj):
                # 轉換顏色為RGB字符串
                color_rgb = colors[obj_idx % len(colors)]
                color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                
                # 添加ground truth標記（星號）
                fig.add_trace(go.Scatter3d(
                    x=[tof_gt[obj_idx]],       # x軸對應ToF
                    y=[aoa_gt[obj_idx]],       # y軸對應AoA
                    z=[fds_gt[obj_idx]],       # z軸對應Doppler shift
                    mode='markers',
                    marker=dict(
                        size=12,               # 較大的標記以便識別
                        color=color_str,
                        symbol='x',            # 使用x符號（星號的近似）
                        opacity=1.0,           # 完全不透明以突出顯示
                        line=dict(width=3, color='black')  # 添加黑色邊框增加可見性
                    ),
                    name=f'GT Object {obj_idx+1}',
                    text=f'GT Obj {obj_idx+1}<br>ToF: {tof_gt[obj_idx]:.2e}s<br>AoA: {aoa_gt[obj_idx]:.1f}°<br>FD: {fds_gt[obj_idx]:.1f}Hz',
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        # 設定固定的軸範圍
        scene_config = dict(
            xaxis_title='ToF (s)',
            yaxis_title='AoA (degrees)',
            zaxis_title='Doppler Shift (Hz)'
        )
        scene_config.update({
            'xaxis': dict(title='ToF (s)', range=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1]]),
            'yaxis': dict(title='AoA (degrees)', range=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]]),
            'zaxis': dict(title='Doppler Shift (Hz)', range=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1]])
        })
        
        # 更新布局
        title = f'3D AoA-ToF-FD Spectrum - Sample {sample_idx}'
        if plot_gt and aoa_gt is not None:
            title += f' (with {len(aoa_gt)} GT objects)'
            
        fig.update_layout(
            title=title,
            scene=scene_config,
            width=1200,
            height=800,
            showlegend=True,  # 顯示圖例以區分散點和GT
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        # 儲存檔案
        filename = f"{sample_idx}.{save_format}"
        filepath = output_path / filename
        
        if save_format == 'html':
            fig.write_html(filepath)
        else:
            fig.write_image(filepath)
        
        return True
        
    except Exception as e:
        print(f"Plotly 3D 生成失敗: {e}")
        return False


def _create_matplotlib_3d(data, tau_grid, theta_grid, fd_grid, sample_idx,
                         output_path, filename_prefix, save_format,
                         threshold_percentile, alpha, figsize):
    """使用Matplotlib創建3D散點圖"""
    try:
        # 計算閾值
        threshold = np.percentile(data, threshold_percentile)
        mask = data > threshold
        
        if not np.any(mask):
            print(f"⚠️ Sample {sample_idx}: 沒有數據點高於閾值")
            return False
        
        # 提取數據點
        tau_points = tau_grid[mask]
        theta_points = theta_grid[mask]
        fd_points = fd_grid[mask]
        values = data[mask]
        
        # 創建3D圖
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(tau_points, theta_points, fd_points, 
                           c=values, cmap='viridis', alpha=alpha, s=20)
        
        ax.set_xlabel('ToF (s)')
        ax.set_ylabel('AoA (degrees)')
        ax.set_zlabel('Doppler Shift (Hz)')
        ax.set_title(f'3D AoA-ToF-FD Spectrum - Sample {sample_idx}')
        
        plt.colorbar(scatter, label='Power (dB)', shrink=0.8)
        
        # 儲存檔案
        filename = f"{filename_prefix}_sample_{sample_idx:06d}.{save_format}"
        filepath = output_path / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Matplotlib 3D 生成失敗: {e}")
        plt.close()
        return False


# def _create_slice_avg_2d(data, heatmap_setting, sample_idx, output_path, 
#                    filename_prefix, figsize,plot_gt,aoa_gt,tof_gt,fds_gt):
#     """創建多個2D平均視圖"""
#     try:
#         n_tau, n_theta, n_fd = data.shape
        
#         # 創建子圖
#         fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*4, figsize[1]*2))
#         fig.suptitle(f'AoA-ToF-FD Spectrum Averages - Sample {sample_idx}', fontsize=16)
        
#         # 1. ToF-AoA 平均 (沿著FD軸取平均)
#         tof_aoa_avg = np.mean(data, axis=2)  # 沿著FD軸(axis=2)取平均
#         im1 = axes[0].imshow(tof_aoa_avg.T, 
#                               aspect='auto', 
#                               extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1], 
#                                      heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
#                               origin='lower', cmap='jet')
#         axes[0].set_xlabel('ToF (s)')
#         axes[0].set_ylabel('AoA (degrees)')
#         axes[0].set_title('ToF-AoA (averaged over Doppler Shift)')
#         plt.colorbar(im1, ax=axes[0])
        
#         # 2. ToF-FD 平均 (沿著AoA軸取平均)
#         tof_fd_avg = np.mean(data, axis=1)  # 沿著AoA軸(axis=1)取平均
#         im2 = axes[1].imshow(tof_fd_avg.T, 
#                               aspect='auto',
#                               extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
#                                      heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1]], 
#                               origin='lower', cmap='jet')
#         axes[1].set_xlabel('ToF (s)')
#         axes[1].set_ylabel('Doppler Shift (Hz)')
#         axes[1].set_title('ToF-FD (averaged over AoA)')
#         plt.colorbar(im2, ax=axes[1])
        
#         # 3. AoA-FD 平均 (沿著ToF軸取平均)
#         aoa_fd_avg = np.mean(data, axis=0)  # 沿著ToF軸(axis=0)取平均
#         im3 = axes[2].imshow(aoa_fd_avg.T, 
#                               aspect='auto',
#                               extent=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1],
#                                      heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1]], 
#                               origin='lower', cmap='jet')
#         axes[2].set_xlabel('AoA (degrees)')
#         axes[2].set_ylabel('Doppler Shift (Hz)')
#         axes[2].set_title('AoA-FD (averaged over ToF)')
#         plt.colorbar(im3, ax=axes[2])
        
#         # # 4. 最大投影 (保持原樣作為比較)
#         # max_proj = np.max(data, axis=2)  # 沿FD軸的最大投影
#         # im4 = axes[1,1].imshow(max_proj.T, 
#         #                       aspect='auto',
#         #                       extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
#         #                              heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
#         #                       origin='lower', cmap='viridis')
#         # axes[1,1].set_xlabel('ToF (s)')
#         # axes[1,1].set_ylabel('AoA (degrees)')
#         # axes[1,1].set_title('Max Projection (along FD)')
#         # plt.colorbar(im4, ax=axes[1,1])
        
#         if plot_gt:
#             # 獲取對象數量
#             num_obj = len(aoa_gt)
#             # 為不同對象定義顏色（使用更鮮明的顏色以便與散點區分）
#             # colors = plt.cm.Set1(np.linspace(0, 1, max(num_obj, 8)))  # 確保至少有8種顏色可選
#             colors = plt.cm.Paired(np.linspace(0, 1, max(num_obj, 8)))
#             for obj_idx in range(num_obj):
#                 # 轉換顏色為RGB字符串
#                 color_rgb = colors[obj_idx % len(colors)]
#                 color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                
#                 # 添加ground truth標記（星號）
#                 axes[0].plot(tof_gt[obj_idx], 
#                                 aoa_gt[obj_idx], 
#                                 color=colors[obj_idx], 
#                                 linewidth=5,                   
#                                 marker='X',                    # 使用大寫X (更粗)
#                                 markersize=18,                 
#                                 markeredgewidth=5,             
#                                 markeredgecolor='white',       # 白色邊框更突出
#                                 label=f'Ground Truth User {obj_idx+1}')

#                 # 添加ground truth標記
#                 axes[1].plot(tof_gt[obj_idx], 
#                                 fds_gt[obj_idx], 
#                                 color=colors[obj_idx], 
#                                 linewidth=5,                   
#                                 marker='X',                    # 使用大寫X (更粗)
#                                 markersize=18,                 
#                                 markeredgewidth=5,             
#                                 markeredgecolor='white',       # 白色邊框更突出
#                                 label=f'Ground Truth User {obj_idx+1}')

#                 # 添加ground truth標記（星號）
#                 axes[2].plot(aoa_gt[obj_idx], 
#                                 fds_gt[obj_idx], 
#                                 color=colors[obj_idx], 
#                                 linewidth=5,                   
#                                 marker='X',                    # 使用大寫X (更粗)
#                                 markersize=18,                 
#                                 markeredgewidth=5,             
#                                 markeredgecolor='white',       # 白色邊框更突出
#                                 label=f'Ground Truth User {obj_idx+1}')

#         plt.tight_layout()
        
#         filename = f"{sample_idx}.png"
#         filepath = output_path / filename
        
#         plt.savefig(filepath, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         return True
        
#     except Exception as e:
#         print(f"Error creating 2D average plots for sample {sample_idx}: {e}")
#         return False

def _create_slice_avg_2d(data, heatmap_setting, sample_idx, output_path, 
                   filename_prefix, figsize, plot_gt, aoa_gt, tof_gt, fds_gt):
    """創建多個2D平均視圖，並同時保存合併圖與個別視圖"""
    try:
        # data shape: (ToF, AoA, FD)
        n_tau, n_theta, n_fd = data.shape
        
        # --- 準備個別圖片的儲存路徑 ---
        parent_dir = output_path.parent
        dirs = {
            'AoA-ToF': parent_dir / 'AoA-ToF', # 對應要求 1
            'AoA-FD':  parent_dir / 'AoA-FD',  # 對應要求 2
            'ToF-FD':  parent_dir / 'ToF-FD'   # 對應要求 3
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # --- 計算平均值 ---
        # data原本是 (ToF, AoA, FD) -> indices (0, 1, 2)
        
        # 1. 沿著 FD (axis 2) 平均 -> 剩 (ToF, AoA)
        avg_over_fd = np.mean(data, axis=2) 
        
        # 2. 沿著 ToF (axis 0) 平均 -> 剩 (AoA, FD)
        avg_over_tof = np.mean(data, axis=0)
        
        # 3. 沿著 AoA (axis 1) 平均 -> 剩 (ToF, FD)
        avg_over_aoa = np.mean(data, axis=1)

        # --- 準備 Ground Truth 顏色 ---
        colors = []
        num_obj = 0
        if plot_gt:
            num_obj = len(aoa_gt)
            colors = plt.cm.Paired(np.linspace(0, 1, max(num_obj, 8)))

        # --- 定義繪圖函式 ---
        def save_single_heatmap(data_2d, extent, xlabel, ylabel, title, save_folder, gt_x, gt_y):
            fig_s, ax_s = plt.subplots(figsize=figsize)
            
            # 注意：這裡的 data_2d 傳入時就必須已經處理好轉置與否
            im = ax_s.imshow(data_2d, aspect='auto', extent=extent, 
                             origin='lower', cmap='jet')
            ax_s.set_xlabel(xlabel)
            ax_s.set_ylabel(ylabel)
            ax_s.set_title(title)
            plt.colorbar(im, ax=ax_s)
            
            if plot_gt:
                for obj_idx in range(num_obj):
                    ax_s.plot(gt_x[obj_idx], gt_y[obj_idx], 
                            color=colors[obj_idx], linewidth=5, marker='X', 
                            markersize=18, markeredgewidth=5, markeredgecolor='white',
                            label=f'GT User {obj_idx+1}')
            
            fig_s.tight_layout()
            filepath = save_folder / f"{sample_idx}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig_s)

        # ==========================================
        # 1. AoA (Y) - ToF (X)
        # ==========================================
        # 原始數據: (ToF, AoA)
        # 目標: Y=AoA, X=ToF
        # 操作: 需要轉置 .T 變成 (AoA, ToF)
        save_single_heatmap(
            data_2d=avg_over_fd.T,  # ★ 重要: 轉置
            extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],       # X軸範圍 (ToF)
                    heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], # Y軸範圍 (AoA)
            xlabel='ToF (s)', 
            ylabel='AoA (degrees)', 
            title='AoA-ToF',
            save_folder=dirs['AoA-ToF'],
            gt_x=tof_gt, gt_y=aoa_gt
        )

        # ==========================================
        # 2. AoA (Y) - Doppler (X)
        # ==========================================
        # 原始數據: (AoA, FD)
        # 目標: Y=AoA, X=Doppler
        # 操作: 不轉置，直接用 (AoA, FD)
        save_single_heatmap(
            data_2d=avg_over_tof,   # ★ 重要: 不轉置
            extent=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1],         # X軸範圍 (Doppler)
                    heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], # Y軸範圍 (AoA)
            xlabel='Doppler Shift (Hz)', 
            ylabel='AoA (degrees)', 
            title='AoA-Doppler',
            save_folder=dirs['AoA-FD'],
            gt_x=fds_gt, gt_y=aoa_gt # ★ GT 對應: X=FD, Y=AoA
        )

        # ==========================================
        # 3. ToF (Y) - Doppler (X)
        # ==========================================
        # 原始數據: (ToF, FD)
        # 目標: Y=ToF, X=Doppler
        # 操作: 不轉置，直接用 (ToF, FD)
        save_single_heatmap(
            data_2d=avg_over_aoa,   # ★ 重要: 不轉置
            extent=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1],   # X軸範圍 (Doppler)
                    heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1]], # Y軸範圍 (ToF)
            xlabel='Doppler Shift (Hz)', 
            ylabel='ToF (s)', 
            title='ToF-Doppler',
            save_folder=dirs['ToF-FD'],
            gt_x=fds_gt, gt_y=tof_gt # ★ GT 對應: X=FD, Y=ToF
        )

        # ==========================================
        # 合併圖代碼 (Combined Plot) - 同步更新邏輯
        # ==========================================
        
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*4, figsize[1]*2))
        fig.suptitle(f'AoA-ToF-FD Spectrum Averages - Sample {sample_idx}', fontsize=16)
        
        # 子圖 1: AoA(y)-ToF(x)
        im1 = axes[0].imshow(avg_over_fd.T, aspect='auto', 
                             extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1], 
                                     heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[0].set_xlabel('ToF (s)')
        axes[0].set_ylabel('AoA (degrees)')
        axes[0].set_title('AoA-ToF')
        plt.colorbar(im1, ax=axes[0])
        
        # 子圖 2: AoA(y)-Doppler(x)
        im2 = axes[1].imshow(avg_over_tof, aspect='auto', # 無 .T
                             extent=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1],
                                     heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[1].set_xlabel('Doppler Shift (Hz)')
        axes[1].set_ylabel('AoA (degrees)')
        axes[1].set_title('AoA-Doppler')
        plt.colorbar(im2, ax=axes[1])
        
        # 子圖 3: ToF(y)-Doppler(x)
        im3 = axes[2].imshow(avg_over_aoa, aspect='auto', # 無 .T
                             extent=[heatmap_setting.fd_axis[0], heatmap_setting.fd_axis[-1],
                                     heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[2].set_xlabel('Doppler Shift (Hz)')
        axes[2].set_ylabel('ToF (s)')
        axes[2].set_title('ToF-Doppler')
        plt.colorbar(im3, ax=axes[2])
        
        if plot_gt:
            for obj_idx in range(num_obj):
                # 1. AoA(y)-ToF(x) -> GT: x=ToF, y=AoA
                axes[0].plot(tof_gt[obj_idx], aoa_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

                # 2. AoA(y)-Doppler(x) -> GT: x=FD, y=AoA
                axes[1].plot(fds_gt[obj_idx], aoa_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

                # 3. ToF(y)-Doppler(x) -> GT: x=FD, y=ToF
                axes[2].plot(fds_gt[obj_idx], tof_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

        plt.tight_layout()
        filename = f"{sample_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error in creating slice avg 2d: {e}")
        import traceback
        traceback.print_exc()
        return False

def _create_contour_3d(data, tau_grid, theta_grid, fd_grid, sample_idx,
                      output_path, filename_prefix, save_format, figsize):
    """創建3D等值面圖"""
    try:
        from skimage import measure
        
        # 選擇等值面的數值 (例如最大值的50%, 70%, 90%)
        max_val = np.max(data)
        levels = [max_val * 0.5, max_val * 0.7, max_val * 0.9]
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['blue', 'green', 'red']
        alphas = [0.3, 0.5, 0.7]
        
        for i, (level, color, alpha_val) in enumerate(zip(levels, colors, alphas)):
            try:
                # 使用marching cubes算法提取等值面
                verts, faces, normals, values = measure.marching_cubes(data, level)
                
                # 轉換座標
                verts_scaled = np.zeros_like(verts)
                verts_scaled[:, 0] = np.interp(verts[:, 0], [0, data.shape[0]-1], 
                                             [tau_grid[0,0,0], tau_grid[-1,0,0]])
                verts_scaled[:, 1] = np.interp(verts[:, 1], [0, data.shape[1]-1], 
                                             [theta_grid[0,0,0], theta_grid[0,-1,0]])
                verts_scaled[:, 2] = np.interp(verts[:, 2], [0, data.shape[2]-1], 
                                             [fd_grid[0,0,0], fd_grid[0,0,-1]])
                
                # 繪製等值面
                ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:, 1], verts_scaled[:, 2],
                               triangles=faces, color=color, alpha=alpha_val,
                               label=f'Level {level:.2f}')
                
            except Exception as e:
                print(f"等值面 {level} 生成失敗: {e}")
                continue
        
        ax.set_xlabel('ToF (s)')
        ax.set_ylabel('AoA (degrees)')
        ax.set_zlabel('Doppler Shift (Hz)')
        ax.set_title(f'3D Isosurface - Sample {sample_idx}')
        ax.legend()
        
        # 儲存檔案
        filename = f"{filename_prefix}_contour_sample_{sample_idx:06d}.{save_format}"
        filepath = output_path / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"3D Contour 生成失敗: {e}")
        plt.close()
        return False
    
def _create_slice_avg_2d_S3D(data, heatmap_setting, sample_idx, output_path, 
                   filename_prefix, figsize, plot_gt, aoa_gt, tof_gt, fds_gt):
    """創建多個2D平均視圖，並同時保存合併圖與個別視圖"""
    try:
        
        # --- 準備個別圖片的儲存路徑 ---
        parent_dir = output_path.parent
        dirs = {
            'AoD-AoA': parent_dir / 'AoD-AoA', # 對應要求 1
            'AoA-ToF':  parent_dir / 'AoA-ToF',  # 對應要求 2
            'AoD-ToF':  parent_dir / 'AoD-ToF'   # 對應要求 3
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # --- 計算平均值 ---
        # data原本是 (AoD, AoA, ToF) -> indices (0, 1, 2)
        
        # 1. 沿著 ToF (axis 2) 平均 -> 剩 (AoD, AoA)
        avg_over_tof = np.mean(data, axis=2) 
        
        # 2. 沿著 AoD (axis 0) 平均 -> 剩 (AoA, ToF)
        avg_over_aod = np.mean(data, axis=0)
        
        # 3. 沿著 AoA (axis 1) 平均 -> 剩 (AoD, ToF)
        avg_over_aoa = np.mean(data, axis=1)

        # --- 準備 Ground Truth 顏色 ---
        colors = []
        num_obj = 0
        # if plot_gt:
        #     num_obj = len(aoa_gt)
        #     colors = plt.cm.Paired(np.linspace(0, 1, max(num_obj, 8)))

        # --- 定義繪圖函式 ---
        def save_single_heatmap(data_2d, extent, xlabel, ylabel, title, save_folder, gt_x=None, gt_y=None):
            fig_s, ax_s = plt.subplots(figsize=figsize)
            
            # 注意：這裡的 data_2d 傳入時就必須已經處理好轉置與否
            im = ax_s.imshow(data_2d, aspect='auto', extent=extent, 
                             origin='lower', cmap='jet')
            ax_s.set_xlabel(xlabel)
            ax_s.set_ylabel(ylabel)
            ax_s.set_title(title)
            plt.colorbar(im, ax=ax_s)
            
            # if plot_gt:
            #     for obj_idx in range(num_obj):
            #         ax_s.plot(gt_x[obj_idx], gt_y[obj_idx], 
            #                 color=colors[obj_idx], linewidth=5, marker='X', 
            #                 markersize=18, markeredgewidth=5, markeredgecolor='white',
            #                 label=f'GT User {obj_idx+1}')
            
            fig_s.tight_layout()
            filepath = save_folder / f"{sample_idx}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig_s)

        # ==========================================
        # 1. AoD (Y) - AoA (X)
        # ==========================================
        # 原始數據: (AoD, AoA)
        # 目標: Y=AoD, X=AoA
        # 操作: 不需要轉置
        save_single_heatmap(
            data_2d=avg_over_tof,
            extent=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1],       # X軸範圍 (AoA)
                    heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], # Y軸範圍 (AoD)
            xlabel='AoA (degrees)', 
            ylabel='AoD (degrees)', 
            title='AoD-AoA',
            save_folder=dirs['AoD-AoA'],
            # gt_x=aoa_gt, gt_y=phi_grid
        )

        # ==========================================
        # 2. AoA (Y) - ToF (X)
        # ==========================================
        # 原始數據: (AoA, ToF)
        # 目標: Y=AoA, X=ToF
        # 操作: 不轉置，直接用 (AoA, ToF)
        save_single_heatmap(
            data_2d=avg_over_aod,   # ★ 重要: 不轉置
            extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],         # X軸範圍 (ToF)
                    heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], # Y軸範圍 (AoA)
            xlabel='ToF (s)', 
            ylabel='AoA (degrees)', 
            title='AoA-ToF',
            save_folder=dirs['AoA-ToF'],
            # gt_x=fds_gt, gt_y=aoa_gt # ★ GT 對應: X=FD, Y=AoA
        )

        # ==========================================
        # 3. AoD (Y) - ToF (X)
        # ==========================================
        # 原始數據: (AoD, ToF)
        # 目標: Y=AoD, X=ToF
        # 操作: 不轉置，直接用 (AoD, ToF)
        save_single_heatmap(
            data_2d=avg_over_aoa,   # ★ 重要: 不轉置
            extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],   # X軸範圍 (ToF)
                    heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], # Y軸範圍 (AoD)
            xlabel='ToF (s)', 
            ylabel='AoD (degrees)', 
            title='AoD-ToF',
            save_folder=dirs['AoD-ToF'],
            # gt_x=fds_gt, gt_y=tof_gt # ★ GT 對應: X=FD, Y=ToF
        )

        # ==========================================
        # 合併圖代碼 (Combined Plot) - 同步更新邏輯
        # ==========================================
        
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*4, figsize[1]*2))
        fig.suptitle(f'AoD-AoA-ToF Spectrum Averages - Sample {sample_idx}', fontsize=16)
        
        # 子圖 1: AoD(y)-AoA(x)
        im1 = axes[0].imshow(avg_over_tof, aspect='auto', 
                             extent=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1], 
                                     heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[0].set_xlabel('AoA (degrees)')
        axes[0].set_ylabel('AoD (degrees)')
        axes[0].set_title('AoA-AoD')
        plt.colorbar(im1, ax=axes[0])
        
        # 子圖 2: AoA(y)-ToF(x)
        im2 = axes[1].imshow(avg_over_aod, aspect='auto',
                             extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
                                     heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[1].set_xlabel('ToF (s)')
        axes[1].set_ylabel('AoA (degrees)')
        axes[1].set_title('AoA-ToF')
        plt.colorbar(im2, ax=axes[1])
        
        # 子圖 3: AoD(y)-ToF(x)
        im3 = axes[2].imshow(avg_over_aoa, aspect='auto', # 無 .T
                             extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
                                     heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[2].set_xlabel('ToF (s)')
        axes[2].set_ylabel('AoD (degrees)')
        axes[2].set_title('AoD-ToF')
        plt.colorbar(im3, ax=axes[2])
        
        # if plot_gt:
        #     for obj_idx in range(num_obj):
        #         # 1. AoA(y)-ToF(x) -> GT: x=ToF, y=AoA
        #         axes[0].plot(tof_gt[obj_idx], aoa_gt[obj_idx], 
        #                      color=colors[obj_idx], linewidth=5, marker='X', 
        #                      markersize=18, markeredgewidth=5, markeredgecolor='white')

        #         # 2. AoA(y)-Doppler(x) -> GT: x=FD, y=AoA
        #         axes[1].plot(fds_gt[obj_idx], aoa_gt[obj_idx], 
        #                      color=colors[obj_idx], linewidth=5, marker='X', 
        #                      markersize=18, markeredgewidth=5, markeredgecolor='white')

        #         # 3. ToF(y)-Doppler(x) -> GT: x=FD, y=ToF
        #         axes[2].plot(fds_gt[obj_idx], tof_gt[obj_idx], 
        #                      color=colors[obj_idx], linewidth=5, marker='X', 
        #                      markersize=18, markeredgewidth=5, markeredgecolor='white')

        plt.tight_layout()
        filename = f"{sample_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error in creating slice avg 2d: {e}")
        import traceback
        traceback.print_exc()
        return False

def _create_plotly_3d_F4D(heatmap_setting, data, phi_grid, theta_grid, tau_grid, fd_grid, sample_idx, 
                     output_path, filename_prefix, save_format, 
                     threshold_percentile, alpha, plot_gt, aoa_gt, tof_gt, fds_gt):
    """使用Plotly創建交互式3D散點圖，包含ground truth顯示"""    
    try:
        # 計算閾值
        threshold = np.percentile(data, threshold_percentile)
        
        # 找出高於閾值的點
        mask = data > threshold
        
        if not np.any(mask):
            print(f"⚠️ Sample {sample_idx}: 沒有數據點高於閾值")
            return False
        
        # 提取座標和數值
        phi_points = phi_grid[mask]
        theta_points = theta_grid[mask]
        tau_points = tau_grid[mask]
        fd_points = fd_grid[mask]
        values = data[mask]
        
        # 創建3D散點圖 - 調整透明度使其更透明
        fig = go.Figure(data=go.Scatter3d(
            x=phi_points,
            y=theta_points, 
            z=tau_points,
            mode='markers',
            marker=dict(
                size=3,
                color=values,
                colorscale='Viridis',
                opacity=alpha,  # 大幅降低透明度，使散點更透明
                colorbar=dict(title="Power (dB)")
            ),
            text=[f'ToF: {t:.2e}s<br>AoA: {a:.1f}°<br>FD: {f:.1f}Hz<br>Power: {v:.2f}dB' 
                  for t, a, f, v in zip(phi_points, theta_points, tau_points, values)],
            hovertemplate='%{text}<extra></extra>',
            name='Detected Peaks'
        ))
        
        # # 添加ground truth標記
        # if plot_gt and aoa_gt is not None and tof_gt is not None and fds_gt is not None:
        #     # 獲取對象數量
        #     num_obj = len(aoa_gt)
            
        #     # 為不同對象定義顏色（使用更鮮明的顏色以便與散點區分）
        #     colors = plt.cm.Set1(np.linspace(0, 1, max(num_obj, 8)))  # 確保至少有8種顏色可選
            
        #     for obj_idx in range(num_obj):
        #         # 轉換顏色為RGB字符串
        #         color_rgb = colors[obj_idx % len(colors)]
        #         color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                
        #         # 添加ground truth標記（星號）
        #         fig.add_trace(go.Scatter3d(
        #             x=[tof_gt[obj_idx]],       # x軸對應ToF
        #             y=[aoa_gt[obj_idx]],       # y軸對應AoA
        #             z=[fds_gt[obj_idx]],       # z軸對應Doppler shift
        #             mode='markers',
        #             marker=dict(
        #                 size=12,               # 較大的標記以便識別
        #                 color=color_str,
        #                 symbol='x',            # 使用x符號（星號的近似）
        #                 opacity=1.0,           # 完全不透明以突出顯示
        #                 line=dict(width=3, color='black')  # 添加黑色邊框增加可見性
        #             ),
        #             name=f'GT Object {obj_idx+1}',
        #             text=f'GT Obj {obj_idx+1}<br>ToF: {tof_gt[obj_idx]:.2e}s<br>AoA: {aoa_gt[obj_idx]:.1f}°<br>FD: {fds_gt[obj_idx]:.1f}Hz',
        #             hovertemplate='%{text}<extra></extra>'
        #         ))
        
        # 設定固定的軸範圍
        scene_config = dict(
            xaxis_title='AoD (degrees)',
            yaxis_title='AoA (degrees)',
            zaxis_title='ToF (s)'
        )
        scene_config.update({
            'xaxis': dict(title='AoD (degrees)', range=[heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]]),
            'yaxis': dict(title='AoA (degrees)', range=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]]),
            'zaxis': dict(title='ToF (s)', range=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1]])
        })
        
        # 更新布局
        title = f'3D AoA-ToF-FD Spectrum - Sample {sample_idx}'
        # if plot_gt and aoa_gt is not None:
        #     title += f' (with {len(aoa_gt)} GT objects)'
            
        fig.update_layout(
            title=title,
            scene=scene_config,
            width=1200,
            height=800,
            showlegend=True,  # 顯示圖例以區分散點和GT
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        # 儲存檔案
        filename = f"{sample_idx}.{save_format}"
        filepath = output_path / filename
        
        if save_format == 'html':
            fig.write_html(filepath)
        else:
            fig.write_image(filepath)
        
        return True
        
    except Exception as e:
        print(f"Plotly 3D 生成失敗: {e}")
        return False
    
def _create_slice_avg_2d_F4D(data, heatmap_setting, sample_idx, output_path, 
                   filename_prefix, figsize, plot_gt, aoa_gt, tof_gt, fds_gt):
    """創建多個2D平均視圖，並同時保存合併圖與個別視圖"""
    try:
        # data shape: (AoD,AoA,ToF, FD)
        n_phi, n_theta, n_tau, n_fd = data.shape
        
        # --- 準備個別圖片的儲存路徑 ---
        parent_dir = output_path.parent
        dirs = {
            'AoA-AoD': parent_dir / 'AoA-AoD', # 對應要求 1
            'AoA-ToF':  parent_dir / 'AoA-ToF',  # 對應要求 2
            'AoD-ToF':  parent_dir / 'AoD-ToF'   # 對應要求 3
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # --- 計算平均值 ---
        # data原本是 (AoD,AoA,ToF,Fd) -> indices (0, 1, 2, 3)
        
        # 1. 沿著 ToF,FD (axis 2, 3) 平均 -> 剩 (AoD, AoA)
        avg_over_tof = np.mean(data, axis=(2,3)) 
        
        # 2. 沿著 AoD,FD (axis 0, 3) 平均 -> 剩 (AoA, ToF)
        avg_over_aod = np.mean(data, axis=(0,3))
        
        # 3. 沿著 AoA,FD (axis 1, 3) 平均 -> 剩 (AoD, ToF)
        avg_over_aoa = np.mean(data, axis=(1,3))

        # --- 準備 Ground Truth 顏色 ---
        colors = []
        num_obj = 0
        if plot_gt:
            num_obj = len(aoa_gt)
            colors = plt.cm.Paired(np.linspace(0, 1, max(num_obj, 8)))

        # --- 定義繪圖函式 ---
        def save_single_heatmap(data_2d, extent, xlabel, ylabel, title, save_folder, gt_x, gt_y):
            fig_s, ax_s = plt.subplots(figsize=figsize)
            
            # 注意：這裡的 data_2d 傳入時就必須已經處理好轉置與否
            im = ax_s.imshow(data_2d, aspect='auto', extent=extent, 
                             origin='lower', cmap='jet')
            ax_s.set_xlabel(xlabel)
            ax_s.set_ylabel(ylabel)
            ax_s.set_title(title)
            plt.colorbar(im, ax=ax_s)
            
            if plot_gt:
                for obj_idx in range(num_obj):
                    ax_s.plot(gt_x[obj_idx], gt_y[obj_idx], 
                            color=colors[obj_idx], linewidth=5, marker='X', 
                            markersize=18, markeredgewidth=5, markeredgecolor='white',
                            label=f'GT User {obj_idx+1}')
            
            fig_s.tight_layout()
            filepath = save_folder / f"{sample_idx}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig_s)

        # ==========================================
        # 1. AoD (Y) - AoA (X)
        # ==========================================
        # 原始數據: (AoD, AoA)
        # 目標: Y=AoD, X=AoA
        # 操作: 不需要轉置
        save_single_heatmap(
            data_2d=avg_over_tof,  # ★ 重要: 不轉置
            extent=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1],       # X軸範圍 (AoA)
                    heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], # Y軸範圍 (AoD)
            xlabel='AoA (degrees)', 
            ylabel='AoD (degrees)', 
            title='AoA-AoD',
            save_folder=dirs['AoA-AoD'],
            gt_x=aoa_gt, gt_y=aoa_gt
        )

        # ==========================================
        # 2. AoA (Y) - ToF (X)
        # ==========================================
        # 原始數據: (AoA, ToF)
        # 目標: Y=AoA, X=ToF
        # 操作: 不轉置，直接用 (AoA, ToF)
        save_single_heatmap(
            data_2d=avg_over_aod,   # ★ 重要: 不轉置
            extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],         # X軸範圍 (ToF)
                    heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], # Y軸範圍 (AoA)
            xlabel='ToF (s)', 
            ylabel='AoA (degrees)', 
            title='AoA-ToF',
            save_folder=dirs['AoA-ToF'],
            gt_x=tof_gt, gt_y=aoa_gt # ★ GT 對應: X=ToF, Y=AoA
        )

        # ==========================================
        # 3. AoD (Y) - ToF (X)
        # ==========================================
        # 原始數據: (AoD, ToF)
        # 目標: Y=AoD, X=ToF
        # 操作: 不轉置，直接用 (AoD, ToF)
        save_single_heatmap(
            data_2d=avg_over_aoa,   # ★ 重要: 不轉置
            extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],   # X軸範圍 (ToF)
                    heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], # Y軸範圍 (AoD)
            xlabel='ToF (s)', 
            ylabel='AoD (degrees)', 
            title='AoD-ToF',
            save_folder=dirs['AoD-ToF'],
            gt_x=tof_gt, gt_y=aoa_gt # ★ GT 對應: X=ToF, Y=AoD
        )

        # ==========================================
        # 合併圖代碼 (Combined Plot) - 同步更新邏輯
        # ==========================================
        
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*4, figsize[1]*2))
        fig.suptitle(f'AoD-AoA-ToF-FD Spectrum Averages - Sample {sample_idx}', fontsize=16)
        
        # 子圖 1: AoD(y)-AoA(x)
        im1 = axes[0].imshow(avg_over_tof, aspect='auto', 
                             extent=[heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1], 
                                     heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[0].set_xlabel('AoA (degrees)')
        axes[0].set_ylabel('AoD (degrees)')
        axes[0].set_title('AoA-AoD')
        plt.colorbar(im1, ax=axes[0])
        
        # 子圖 2: AoA(y)-ToF(x)
        im2 = axes[1].imshow(avg_over_aod, aspect='auto',
                             extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
                                     heatmap_setting.theta_deg_axis[0], heatmap_setting.theta_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[1].set_xlabel('ToF (s)')
        axes[1].set_ylabel('AoA (degrees)')
        axes[1].set_title('AoA-ToF')
        plt.colorbar(im2, ax=axes[1])
        
        # 子圖 3: AoD(y)-ToF(x)
        im3 = axes[2].imshow(avg_over_aoa, aspect='auto', # 無 .T
                             extent=[heatmap_setting.tau_axis[0], heatmap_setting.tau_axis[-1],
                                     heatmap_setting.phi_deg_axis[0], heatmap_setting.phi_deg_axis[-1]], 
                             origin='lower', cmap='jet')
        axes[2].set_xlabel('ToF (s)')
        axes[2].set_ylabel('AoD (degrees)')
        axes[2].set_title('AoD-ToF')
        plt.colorbar(im3, ax=axes[2])
        
        if plot_gt:
            for obj_idx in range(num_obj):
                # 1. AoD(y)-AoA(x) -> GT: x=AoA, y=AoD
                axes[0].plot(aoa_gt[obj_idx], aoa_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

                # 2. AoA(y)-ToF(x) -> GT: x=ToF, y=AoA
                axes[1].plot(tof_gt[obj_idx], aoa_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

                # 3. AoD(y)-ToF(x) -> GT: x=ToF, y=AoD
                axes[2].plot(tof_gt[obj_idx], aoa_gt[obj_idx], 
                             color=colors[obj_idx], linewidth=5, marker='X', 
                             markersize=18, markeredgewidth=5, markeredgecolor='white')

        plt.tight_layout()
        filename = f"{sample_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error in creating slice avg 2d: {e}")
        import traceback
        traceback.print_exc()
        return False