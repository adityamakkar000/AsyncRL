# annealedRL

## Setup

1. Git clone and then run `uv sync`
2. Run `pre-commit install`
3. Install `https://github.com/adityamakkar000/Mesh` 

## Train


## Eval

Eval is only supported on a single-host TPU vm. Ideally you want a tpu-v6e-8. 

1. Run `uv sync --extra eval` to sync all packages for eval
2. Make eval config in `src/configs/eval/<your_eval_name>.yaml` 
3. Run `python -m src.eval --config-name <your_eval_name>` 


