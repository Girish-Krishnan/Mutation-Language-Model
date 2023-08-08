import torch
import yaml
import warnings
from model import BigramLanguageModel
from mapping import *

# Ignore warnings
warnings.filterwarnings("ignore")

# Set random seed
torch.manual_seed(1337)

# hyperparameters
hyperparams = yaml.load(open('hyperparameters.yaml'), Loader=yaml.FullLoader)
batch_size = hyperparams['batch_size']
block_size = hyperparams['block_size']
max_iters = hyperparams['max_iters']
eval_interval = hyperparams['eval_interval']
learning_rate = hyperparams['learning_rate']
eval_iters = hyperparams['eval_iters']
n_embd = hyperparams['n_embd']
n_head = hyperparams['n_head']
n_layer = hyperparams['n_layer']
dropout = hyperparams['dropout']

# Device - MPS
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# Device - CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=dropout, device=device)

# Load model
model.load_state_dict(torch.load('model.pt'))
m = model.to(device)

# Generate
# print("Generating from zero context:")
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

# generate from the model - given a context
print("Generating from context:")
context = torch.tensor(encode([(266, 'T'),(288, 'T'),(303, 'A')]), dtype=torch.long, device=device).unsqueeze(0)
print(decode(m.generate(context, max_new_tokens=20)[0].tolist()))