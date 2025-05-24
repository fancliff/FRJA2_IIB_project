import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

    # Extract model class from Model ID
    df['Model Class'] = df['Model ID'].apply(lambda x: x.split('_')[-1])

    # Apply log-safe transformation
    log_scale_cols = ['Trainable Parameters', 'Mean FRF Error', 'Training Time (s)']
    # Only needed if columns may contain zeros which the above columns should not
    for col in log_scale_cols:
        df[col] = df[col].replace(0, np.nan)

    fig = plt.figure(figsize=(10, 7))

    if len(x_keys) == 1: # 2D plot
        x_key = x_keys[0]
        ax = fig.add_subplot(111)

        for model_class in df['Model Class'].unique():
            sub_df = df[df['Model Class'] == model_class]
            label = legend_labels.get(model_class, model_class) if legend_labels else model_class
            ax.scatter(sub_df[x_key], sub_df['Mean FRF Error'], label=label)

        if x_key in log_scale_cols:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(x_key)
        ax.set_ylabel('Mean FRF Error')
        ax.set_title(f'Mean FRF Error vs {x_key}')
        ax.legend()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    ####### 3D plotting not working currently ######
    # else: # 3D plot
    #     x_key, y_key = x_keys
    #     ax = fig.add_subplot(111, projection='3d')

    #     for model_class in df['Model Class'].unique():
    #         sub_df = df[df['Model Class'] == model_class]
    #         label = legend_labels.get(model_class, model_class) if legend_labels else model_class
    #         ax.scatter(
    #             sub_df[x_key], sub_df[y_key], sub_df['Mean FRF Error'], label=label
    #         )

    #     if x_key in log_scale_cols:
    #         ax.set_xscale('log')
    #     if y_key in log_scale_cols:
    #         ax.set_yscale('log')
    #     ax.zaxis._set_scale('log')
    #     ax.set_xlabel(x_key)
    #     ax.set_ylabel(y_key)
    #     ax.zaxis.set_label('Mean FRF Error')
    #     ax.set_title(f'Mean FRF Error vs {x_key} & {y_key}')
    #     ax.legend()

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


# plot_model_performance(
#     r'C:\Users\Freddie\Documents\IIB project repository\myenv\FRJA2_IIB_project\model_training_results.csv',
#     ['Trainable Parameters', 'Training Time (s)'],
#     legend_labels=legend_mapping
# )
