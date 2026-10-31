import torch, os

log_dir = os.environ.get("vrbea_log_dir", os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(log_dir, exist_ok=True)

n_obs = [50,100,200]

nsims = 10
seed_num = 1997
param = [-1,1]

active = ['edges','triangles']

perturb =0.0

