import torch
import yaml
import warnings
import tqdm
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

# Train and test splits
data = torch.tensor(encode(mutations_data), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=dropout, device=device)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        tqdm.tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save model
torch.save(model.state_dict(), 'model.pt')
print("Model saved")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))