"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 10
image_size = 64

# params for source dataset
src_dataset = "REFUGE_SRC"
src_image_dir = 'data/REFUGE/trainImage_save_path_512'
src_mask_dir = 'data/REFUGE/MaskImage_save_path_512'
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "REFUGE_DST"
tgt_image_dir = 'data/REFUGE/valiImage_save_path_512'
tgt_mask_dir = 'data/REFUGE/valiMaskImage_save_path_512'
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 30 #100
log_step_pre = 5
eval_step_pre = 10 # 20
save_step_pre = 100
num_epochs = 200#1000
log_step = 5 # 100
save_step = 5 #100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

# params for saving the results
save_encoder_restore = "snapshots/ADDA-target-encoder-5.pt"
save_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
save_dir = "results"
