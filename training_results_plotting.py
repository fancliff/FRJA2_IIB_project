import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker

def plot_model_performance(csv_path, x_keys, legend_labels=None):
    """
    Plot model performance (Mean FRF Error) from a training results CSV.

    Args:
        csv_path (str): Path to the CSV file.
        x_keys (list): One or two column names to plot against performance.
        legend_labels (dict): Optional mapping from model class names to custom legend labels.
                            Example: {'RegressionModel1': 'Pure Convolutional CNN'}
    """
    assert 1 <= len(x_keys) <= 2, "Please provide one or two x-axis keys."

    df = pd.read_csv(csv_path)

    is_3d = len(x_keys) == 2

    # Extract model class from Model ID
    df['Model Class'] = df['Model ID'].apply(lambda x: x.split('_')[-1])

    log_scale_cols = ['Trainable Parameters', 'Mean FRF Error', 'Training Time (s)']

    if is_3d:
        # Apply log10 to Mean FRF Error (handling zeros/negative values)
        df['Mean FRF Error'] = df['Mean FRF Error'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
        df = df.dropna(subset=['Mean FRF Error'])
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        def log_tick_formatter(val, pos=None):
            # return f"$10^{{{int(val)}}}$"
            return f"$10^{{{val:g}}}$"
        
        for model_class in df['Model Class'].unique():
            sub_df = df[df['Model Class'] == model_class]
            label = legend_labels.get(model_class, model_class) if legend_labels else model_class

            # Get x, y values - apply log if needed
            x = np.log10(sub_df[x_keys[0]]) if x_keys[0] in log_scale_cols else sub_df[x_keys[0]]
            y = np.log10(sub_df[x_keys[1]]) if x_keys[1] in log_scale_cols else sub_df[x_keys[1]]
            z = sub_df['Mean FRF Error']
            
            ax.scatter(x, y, z, label=label)

        # Set labels
        ax.set_xlabel(x_keys[0])
        ax.set_ylabel(x_keys[1])
        ax.set_zlabel('Mean FRF Error')
        
        # Log format x and y axes if needed
        if x_keys[0] in log_scale_cols:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
            # True will only display 10^int gridlines
            # False will display many 10^non-int gridlines
        if x_keys[1] in log_scale_cols:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
            # True will only display 10^int gridlines
            # False will display many 10^non-int gridlines
        # Log format z-axis
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        # ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True)) 
        # set_major_locator doesn't seem to work on z-axis
        # Probably another matplotlib z-axis bug/issue/missing-feature
    else:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for model_class in df['Model Class'].unique():
            sub_df = df[df['Model Class'] == model_class]
            label = legend_labels.get(model_class, model_class) if legend_labels else model_class
            x = sub_df[x_keys[0]]
            y = sub_df['Mean FRF Error']
            ax.scatter(x,y,label=label)

        if x_keys[0] in log_scale_cols:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'{x_keys[0]}')
        ax.set_ylabel('Mean FRF Error')
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
    ['Trainable Parameters'],
    legend_labels=legend_mapping
)


plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Receptive Field'],
    legend_labels=legend_mapping
)

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Training Time (s)'],
    legend_labels=legend_mapping
)


plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Trainable Parameters', 'Training Time (s)'],
    legend_labels=legend_mapping
)

plot_model_performance(
    r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
    ['Trainable Parameters', 'Receptive Field'],
    legend_labels=legend_mapping
)
