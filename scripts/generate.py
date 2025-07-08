#!/usr/bin/env python3
"""Generate sequences using a trained model."""
import argparse

import torch
import yaml

from mlm.model import BigramLanguageModel
from mlm.mapping import encode, decode, vocab_size


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sequences from a trained model")
    parser.add_argument("model", help="Path to the trained model file")
    parser.add_argument("--config", default="hyperparameters.yaml", help="Path to the hyperparameter YAML file")
    parser.add_argument("--context", default="", help="Comma separated list of mutations like 266:T,288:T")
    parser.add_argument("--tokens", type=int, default=20, help="Number of tokens to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    hyperparams = yaml.safe_load(open(args.config))
    block_size = hyperparams["block_size"]
    n_embd = hyperparams["n_embd"]
    n_head = hyperparams["n_head"]
    n_layer = hyperparams["n_layer"]
    dropout = hyperparams["dropout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        device=device,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    if args.context:
        context_list = []
        for item in args.context.split(','):
            position, base = item.split(':')
            context_list.append((int(position), base))
        context = torch.tensor(encode(context_list), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=args.tokens)[0].tolist()
    print(decode(generated))


if __name__ == "__main__":
    main()
