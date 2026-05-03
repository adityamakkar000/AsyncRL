# annealedRL

## Setup

1. Git clone and then run `uv sync`
2. Run `pre-commit install`
3. Install `https://github.com/adityamakkar000/Mesh` 
4. Use the bash scripts in `src/scripts/tpu/main.sh` to allocate and list the external IPs of a TPU cluster
5. Create a `~/.config/mesh/cluster.yaml` file following the format in ![MESH readme](https://github.com/adityamakkar000/Mesh/blob/main/README.md)
6. Make a copy of the  `.env` and file in all the keys
    - For a STAX token make a read-only PAT in github to allow Mesh to sync when running uv sync in the cluster

## Train

1. make sure mesh is updated
    - rm -rf ~/.local/bin/mesh
    - install as before: ```curl -fsSL https://raw.githubusercontent.com/adityamakkar000/mesh/main/scripts/install.sh | bash``` 
2. update cluster.yaml with mac-mini as "server" and the ssh file you gave to Chinmay. Note: Make sure it is server as the name
```
server:
  user: dev
  identity_file: <your private key here>
  hosts:
    - 100.79.104.73
```
3. Update `.env` with the following keys
```
SSH_IDENTITY_FILE="/Users/dev/.ssh/google_compute_engine"
TPU_USERNAME="dev"
TPU_SERVER_URL="http://100.79.104.73:8000"
```
4. Add `src/scripts/server/endpoints.sh` to your `.zshrc` file and then reload and do `server_ls`. You should see a table now. 

Notes: 
- Before you delete, GET the jobs. Do not delete on provisioning. if you do, then go manually to GCP to cleanup TPU
- To get full command, redirect. Eg `server_ls > jobs.txt`
    

#### Run using `launch.py`
- Comment 1st line in `mesh.yaml` prerun

#### Run using Mesh manually
- Change annealing config as required
- Uncomment 1st line in `mesh.yaml` prerun

## Eval

Eval is only supported on a single-host TPU vm, ideally you want a tpu-v6e-8. 

1. Make eval config in `src/configs/eval/<your_eval_name>.yaml` 
2. Run `mesh run <cluster> "python -m src.eval --config-name <your_eval_name>"` 

## Dataset

The dataset layer lives in `src/data/`