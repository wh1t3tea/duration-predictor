log_file = "train_logs.log"
device = "cuda:0"

#DATA

root_dir = "data"
batch_size = 128

#MODEL

embedding_size = 256
filter_size = 256
hidden = 256
speaker_embedding_size = 2048
n_conv_layers = 2

optim = "adam"
lr = 3e-4
weight_decay = 0
scheduler = "cosine"
step_decay = 6
decay = 0.92

max_epochs = 30
threshold_dict = {2: 1,
                  5: 0.85,
                  15: 0.5,
                  30: 0.1}

