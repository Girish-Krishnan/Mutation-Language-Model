# Mutation Language Model

This repository contains a simple Transformer based language model trained on SARS‑CoV‑2 mutation data. It was refactored to be easier to use and extend.

## Dataset

The training data is a text file where each line contains a comma separated list of mutations. A mutation is represented as a position and the substituted base, for example `266T`.

Firstly, you can download `paths.txt` as follows. `paths.txt` must contain colon separated lines where the second field holds the comma separated mutations.

```bash
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=17Ca3N3ZZutAW7iYbZ4UmClzWvW9zHDhL" -O paths.txt
```

Then, you can prepare the dataset from `paths.txt` using the provided script:

```bash
python mlm/prepare_dataset.py paths.txt data.txt
```

## Training

Training is performed with `scripts/train.py`. It expects the mutations file and an optional hyperparameter YAML file.

```bash
python scripts/train.py data.txt --config hyperparameters.yaml --output model.pt
```

The training script saves the model to the specified output path.

## Generation

To generate new mutation sequences from a trained model use `scripts/generate.py`:

```bash
python scripts/generate.py model.pt --context 266:T,288:T --tokens 20
```

The optional `--context` argument provides a starting sequence of mutations using the `position:base` format. If omitted generation starts from an empty context.

## Package Layout

- `mlm/` – package with the model, dataset utilities and helpers.
- `scripts/` – command line tools for training and generation.
- `hyperparameters.yaml` – default hyperparameters used by the scripts.

## Requirements

The code requires Python 3.8+ and PyTorch. Install dependencies with:

```bash
pip install torch tqdm pyyaml
```

## License

This project is provided as-is for educational purposes.
