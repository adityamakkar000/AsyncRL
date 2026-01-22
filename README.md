# annealedRL

## Setup

1. Git clone and then run `uv sync`
2. Run `pre-commit install`
3. Install `https://github.com/adityamakkar000/Mesh` 
4. Use the bash scripts in `src/scripts/tpu/main.sh` to allocate and list the external IPs of a TPU cluster
5. Create a `~/.config/mesh/cluster.yaml` file following the format in ![MESH readme]()
6. Make a copy of the  `.env` and file in all the keys
    - For a STAX token make a read-only PAT in github to allow Mesh to sync when running uv sync in the cluster

## Train

To train create your config and run
`mesh run <your_cluster> "python -m src.train --config-name <your_config_name> key=value [...]`


## Eval

Eval is only supported on a single-host TPU vm, ideally you want a tpu-v6e-8. 

1. Make eval config in `src/configs/eval/<your_eval_name>.yaml` 
2. Run `mesh run <cluster> "python -m src.eval --config-name <your_eval_name>"` 


