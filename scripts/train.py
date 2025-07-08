#!/usr/bin/env python3
"""Training script for the mutation language model."""
import argparse
import random
import warnings

import torch
import yaml
import numpy as np
import tqdm

from mlm.model import BigramLanguageModel
from mlm.mapping import encode, vocab_size
from mlm.dataset import load_dataset

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the mutation language model")
    parser.add_argument("data", help="Path to the training data text file")
    parser.add_argument("--config", default="hyperparameters.yaml", help="Path to the hyperparameter YAML file")
    parser.add_argument("--output", default="model.pt", help="Where to save the trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    hyperparams = yaml.safe_load(open(args.config))
    batch_size = hyperparams["batch_size"]
    block_size = hyperparams["block_size"]
    max_iters = hyperparams["max_iters"]
    eval_interval = hyperparams["eval_interval"]
    learning_rate = hyperparams["learning_rate"]
    eval_iters = hyperparams["eval_iters"]
    n_embd = hyperparams["n_embd"]
    n_head = hyperparams["n_head"]
    n_layer = hyperparams["n_layer"]
    dropout = hyperparams["dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_dataset(args.data)
    data = [torch.tensor(encode(seq), dtype=torch.long) for seq in raw_data]
    random.shuffle(data)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split, index=0):
        dataset = train_data[index] if split == "train" else val_data[index]
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        x = torch.stack([dataset[i:i + block_size] for i in ix])
        y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                for idx in range(len(train_data) if split == "train" else len(val_data)):
                    X, Y = get_batch(split, index=idx)
                    _, loss = model(X, Y)
                    losses.append(loss.item())
            out[split] = np.mean(losses)
        model.train()
        return out

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        device=device,
    ).to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_iter_losses = []
    for iter in tqdm.tqdm(range(max_iters)):
        total_iter_loss = 0.0
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            tqdm.tqdm.write(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        for idx in range(len(train_data)):
            xb, yb = get_batch("train", index=idx)
            _, loss = model(xb, yb)
            total_iter_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tqdm.tqdm.write(f"seq {idx}: loss {loss.item():.4f}")
        total_iter_losses.append(total_iter_loss)

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
