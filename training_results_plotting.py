import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from scipy.stats import linregress

def plot_model_performance(csv_path, ax_keys, legend_labels=None, trendline=False):
    """
    Plot model performance (Mean FRF Error) from a training results CSV.

    Args:
        csv_path (str): Path to the CSV file.
        x_keys (list): One or two column names to plot against performance.
        legend_labels (dict): Optional mapping from model class names to custom legend labels.
                            Example: {'RegressionModel1': 'Pure Convolutional CNN'}
        trendline (bool): Display trend line for 2D plots (for all model classes)
    """
    assert 2 <= len(ax_keys) <= 3, "Please provide two or three axis keys."

    df = pd.read_csv(csv_path)

    is_3d = len(ax_keys) == 3

    # Extract model class from Model ID
    df['Model Class'] = df['Model ID'].apply(lambda x: x.split('_')[-1])

    log_scale_cols = ['Trainable Parameters', 'Mean FRF Error', 'Training Time (s)']

    if is_3d:
        x_key = ax_keys[0]
        y_key = ax_keys[1]
        z_key = ax_keys[2]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        def log_tick_formatter(val, pos=None):
            # return f"$10^{{{int(val)}}}$"
            return f"$10^{{{val:g}}}$"
        
        for model_class in df['Model Class'].unique():
            sub_df = df[df['Model Class'] == model_class]
            label = legend_labels.get(model_class, model_class) if legend_labels else model_class

            # Get x, y values - apply log if needed
            x = np.log10(sub_df[x_key]) if x_key in log_scale_cols else sub_df[x_key]
            y = np.log10(sub_df[y_key]) if y_key in log_scale_cols else sub_df[y_key]
            z = np.log10(sub_df[z_key]) if z_key in log_scale_cols else sub_df[z_key]
            
            ax.scatter(x, y, z, label=label)

        # Set labels
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_zlabel(z_key)
        
        # Log format x and y axes if needed
        if x_key in log_scale_cols:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
            # True will only display 10^int gridlines
            # False will display many 10^non-int gridlines
        if y_key in log_scale_cols:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
            # True will only display 10^int gridlines
            # False will display many 10^non-int gridlines
        if z_key in log_scale_cols:
            ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            # ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
            # set_major_locator doesn't seem to work on z-axis
            # Probably another matplotlib z-axis bug/issue/missing-feature
    else:
        x_key = ax_keys[0]
        y_key = ax_keys[1]
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for model_class in df['Model Class'].unique():
            sub_df = df[df['Model Class'] == model_class]
            label = legend_labels.get(model_class, model_class) if legend_labels else model_class
            x = sub_df[x_key]
            y = sub_df[y_key]
            ax.scatter(x,y,label=label)

        if x_key in log_scale_cols:
            ax.set_xscale('log')
        if y_key in log_scale_cols:
            ax.set_yscale('log')
        if trendline:
            x_all = df[x_key]
            y_all = df[y_key]

            if x_key in log_scale_cols:
                x_all = np.log10(x_all)
            if y_key in log_scale_cols:
                y_all = np.log10(y_all)

            slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
            x_fit = np.linspace(x_all.min(), x_all.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit if x_key not in log_scale_cols else 10**x_fit,
                    y_fit if y_key not in log_scale_cols else 10**y_fit,
                    color='black', linestyle='--', label='Best Fit')

            if x_key in log_scale_cols and y_key in log_scale_cols:
                eq_label = f'log-log fit: log(y) = {slope:.3f} log(x) + {intercept:.3f}'
            elif x_key in log_scale_cols:
                eq_label = f'log-x fit: y = {slope:.3f} log(x) + {intercept:.3f}'
            elif y_key in log_scale_cols:
                eq_label = f'log-y fit: log(y) = {slope:.3f} x + {intercept:.3f}'
            else:
                eq_label = f'y = {slope:.3f}x + {intercept:.3f}'

            ax.text(0.05, 0.95, eq_label, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    plt.tight_layout()
    plt.show()


legend_mapping = {
    'RegressionModel1': 'Pure Convolutional CNN',
    'ResNet1': 'Res-Net style CNN',
    'DenseNet1': 'Dense-Net style CNN'
}

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Trainable Parameters', 'Mean FRF Error'],
    legend_labels=legend_mapping,
    trendline=True
)

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Receptive Field', 'Mean FRF Error'],
    legend_labels=legend_mapping
)

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Training Time (s)', 'Mean FRF Error'],
    legend_labels=legend_mapping,
    trendline=True,
)

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Trainable Parameters', 'Training Time (s)'],
    legend_labels=legend_mapping,
    trendline=True
)

# plot_model_performance(
#     r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
#     ['Trainable Parameters', 'Receptive Field'],
#     legend_labels=legend_mapping
# )

# plot_model_performance(
#     r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
#     ['Trainable Parameters', 'Training Time (s)', 'Mean FRF Error'],
#     legend_labels=legend_mapping
# )

# plot_model_performance(
#     r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
#     ['Trainable Parameters', 'Receptive Field', 'Mean FRF Error'],
#     legend_labels=legend_mapping
# )
