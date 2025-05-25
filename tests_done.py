###### Regression Test 1 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:26000]
out_channels_list = [
    [4,6,4],
    [4,8,4],
    [4,8,8,4],
    [4,6,8,6,4],
    [4,8,12,8,4],
    [4,4,8,8,4,4,4], # BSL
    [4,6,8,12,8,6,4],
    [4,4,8,8,12,8,8,4],
    [4,4,8,8,12,16,12,8,8,4],
    [4,4,6,6,6,8,8,6,6,6,4,4],
]
kernel_size_list = [
    [5],
    [7],
    [9], # BSL
    [11],
    [13],
]

###### Regression Test 2 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:32000]
out_channels_list = [
    [4,4,4,6,6,6,8,8,8,8,6,6,6,4,4,4],
    [4,4,4,6,6,6,8,8,10,10,8,8,6,6,6,4,4,4],
    [4,4,4,6,6,6,8,8,8,10,12,12,10,8,8,8,6,6,6,4,4,4],
]
kernel_size_list = [
    [9],
    [11],
    [13],
]

###### Res-Net Test 1 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:26000]
out_channels_list = [
    [4,6,4],
    [4,8,4],
    [4,6,8,4],
    [4,6,8,6,4],
    [4,8,12,8,4],
    [4,6,6,8,6,6,4],
    [4,6,6,8,8,6,6,4],
]
kernel_size_list = [
    [5],
    [7],
    [9],
    [11],
    [13],
]

###### Res-Net Test 2 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:32000]
out_channels_list = [
    [4,6,6,8,8,12,8,8,6,6,4],
    [4,4,6,6,6,8,8,8,6,6,6,4,4],
]
kernel_size_list = [
    [9],
    [11],
    [13],
]

###### Dense-Net Test 1 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:26000]
block_config_list = [
    [4,4],
    [4,4,4],
    [4,4,4,4],
    [4,4,4,4,4],
    [4,4,4,4,4,4],
    [4,4,4,4,4,4,4],
]
kernel_size_list = [
    9,
    11,
    13,
]
growth_rate_list = [
    2,
]
transition_channels_list = [
    4,
    6,
]

###### Dense-Net Test 2 ######
val_data = ['data'][:5000]
train_data = ['data'][20000:32000]
block_config_list = [
    [4,4,4,4,4],
    [4,4,4,4,4,4],
]
kernel_size_list = [
    11,
    13,
]
growth_rate_list = [
    4,
]
transition_channels_list = [
    8,
]