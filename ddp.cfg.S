[Data]
data_dir = STAC
train_file = %(data_dir)s/train.json.tok.clean.split
dev_file = %(data_dir)s/dev.json.tok.clean
test_file = %(data_dir)s/test.json.tok.clean
pretrained_embeddings_file = ../glove.6B.100d.txt
max_vocab_size = 1000

[Save]
save_dir = ddp_model
config_file = %(save_dir)s/config.cfg
save_model_path = %(save_dir)s/model
save_vocab_path = %(save_dir)s/vocab
load_dir = ddp_model
load_model_path = %(load_dir)s/model
load_vocab_path = %(load_dir)s/vocab

[Network]
gru_layers = 1
word_dims = 100
relation_dims = 100
dropout_emb = 0.5
gru_hiddens = 250
dropout_gru_hidden = 0.5
mlp_arc_size = 500
mlp_rel_size = 500
use_structure = False

[Optimizer]
L2_REG = 1e-8
learning_rate = 1e-3
decay = .5
decay_steps = 500
beta_1 = .9
beta_2 = .9
epsilon = 1e-12
clip = 2.0

[Run]
train_iters = 50
train_batch_size = 8
test_batch_size = 4
validate_every = 100
save_after = 10
update_every = 1
max_edu_len = 100000

