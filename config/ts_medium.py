# medium model trained on tinystories
# example: python train.py config/ts_medium.py --train_file=train_small.bin --seed=1337

# data
dataset = 'tinystories'
train_file = 'train_large.bin'
vocab_size = 8000
init_from = 'scratch'
out_dir = 'out-ts-medium'

# model size
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.0

# budget: 1 hour of training on a T4
time_budget = 3600.0

# training settings (same for every model)
batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 200
weight_decay = 0.1
beta2 = 0.99
grad_clip = 1.0
dtype = 'float16' # T4 does not support bfloat16

# evaluation and logging
eval_interval = 500
eval_iters = 100
log_interval = 50
