import torch
import yaml
import random
import warnings
import numpy as np
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
# device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

# Device - CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train and test splits
data = [torch.tensor(encode(mutations_data[i]), dtype=torch.long) for i in range(len(mutations_data))]
random.shuffle(data)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split, index=0):
    # generate a small batch of data of inputs x and targets y
    data = train_data[index] if split == 'train' else val_data[index]
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
        losses = []
        for k in range(eval_iters):
            for idx in range(len(train_data) if split == 'train' else len(val_data)):
                X, Y = get_batch(split,index=idx)
                logits, loss = model(X, Y)
                losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, dropout=dropout, device=device)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_iter_losses = []

for iter in tqdm.tqdm(range(max_iters)):
    total_iter_loss = 0

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        tqdm.tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    for idx in range(len(train_data)):
        xb, yb = get_batch('train',index=idx)
        # evaluate the loss
        logits, loss = model(xb, yb)
        total_iter_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tqdm.tqdm.write(f"seq {idx}: loss {loss.item():.4f}")

        # Reset hidden state, since the next batch will be a different sequence
        # model.reset()

    total_iter_losses.append(total_iter_loss)

# Save model
torch.save(model.state_dict(), 'model.pt')
print("Model saved")

# Plot total iter losses
import matplotlib.pyplot as plt
plt.plot(total_iter_losses)
plt.xlabel('Iteration')
plt.ylabel('Total Iteration Loss')
plt.title('Total Iteration Loss vs. Iteration')
plt.savefig('total_iter_losses.png')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
