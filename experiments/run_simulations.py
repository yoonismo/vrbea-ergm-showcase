import os, csv, time, torch
import numpy as np
from collections import defaultdict

from datetime import datetime
from joblib import Parallel, delayed


from ergm_showcase.utils import graph_generator, ergm_estimator, vec_to_str
from experiments import config as cfg


n_jobs = 20

n_obs = cfg.n_obs
nsims = cfg.nsims
seed_num = cfg.seed_num
param = cfg.param
active = cfg.active
perturb = cfg.perturb

log_dir = cfg.log_dir

date_str = datetime.now().strftime("%Y%m%d")

def one_lap(g, adj, n):
    est_mple, est_mle, init, t_mple, t_mle = ergm_estimator(
        param, perturb, n, adj, seed_num+g, active)
    return [ est_mple.tolist(), est_mle.tolist(), init.tolist(), t_mple, t_mle]

for n in n_obs:
    _, _, adjacency_torch= graph_generator(param, n, nsims, seed_num, active)
    adjacency_np = adjacency_torch.numpy()
   
    fname = (f"{date_str}_"f"obs{n}_"
             f"MPLE_MCMCMLE_{vec_to_str(active)}_{vec_to_str(param)}_"
             f"perturb{perturb}.csv")
    
    log_path = os.path.join(log_dir, fname)

    with open(log_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["init", "theta_mple", "time_mple", "theta_mle", "time_mle"])

        total_time_task_start = time.time()
    
        t0 = time.time()

        tasks = (delayed(one_lap)(g, adjacency_np[g], n) for g in range(nsims))
        for g, row in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(tasks)):  
            csv_writer.writerow([row[2], row[0], row[3], row[1], row[4]])
            f.flush()    
            print(f"[n={n}] sim {g} done ({time.time() - t0:.1f}s)", flush=True)
        
        print(f"[n={n}] all {nsims} done, total {time.time() - t0:.1f}s")
            