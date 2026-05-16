# tinyterp

This repository will hold my personal explorations of the mechanistic interpretability domain. I'll
be re-implementing concepts on a smaller scale to convince myself of my own understanding,
minimizing the dependencies on the critical path of training and inference.

## Development Notes

### Setup

```shell
git clone https://www.github.com/trironkk/tinyterp
cd tinyterp
# Install uv at https://docs.astral.sh/uv/getting-started/installation/
uv sync
uv run nbstripout --install --attributes .gitattributes
```

### Recipes

```shell
# Run tests
uv run pytest

# Check NVIDIA detected hardware
nvidia-smi

# Start VSCode
code .
```

## Project Logs

### Environment Setup

```shell
uv --version
# uv 0.11.14 (x86_64-unknown-linux-gnu)

uv python pin
# 3.12
```

### Differentiation

> TODO

### Transformer

> TODO

### Sparse Autoencoders

> TODO

### Circuits

> TODO

### Evaluations

> TODO

## References

> TODO
