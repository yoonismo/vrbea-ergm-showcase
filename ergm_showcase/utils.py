'''
symmetrization
graph_generator (check)
vec_to_str
tensor_str_to_array
'''
import time
import re
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def graph_generator(parameter, nobs, nsim, seed_number, active_terms):
    import rpy2.robjects as robjects
## To import R packages, rpy2 is providing a function rpy2.robjects.packages.importr()
## that makes that step very similar to importing python packages
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.packages import importr, data
    from rpy2.robjects import FloatVector
    base = importr("base")

    network = importr("network", on_conflict="warn")
    sna = importr("sna", on_conflict="warn")
    ergm = importr("ergm", on_conflict="warn")
    """
    FINAL CORRECTED VERSION: Builds the formula string in Python to avoid rpy2 bugs.
    """
    assert len(parameter) == len(active_terms), "Length of parameter must match length of active_terms"

    # Build the R formula string here in Python
    term_map = {
        "edges": "edges",
        "nodematch": 'nodematch("x")',
        "kstar2": "kstar(2)",
        "triangle": "triangles",
        "triangles": "triangles"
    }

    formula_parts = [term_map.get(term, term) for term in active_terms]
    r_formula_string = "~ " + " + ".join(formula_parts)

    start_generation = time.time()

    robjects.r(f'''
    library(network)
    library(ergm)

    graph_generator <- function(parameter, nobs, nsim, seed_num, active_terms, formula_from_python) {{

        set.seed(seed_num)

        # Basis network setup
        g <- network.initialize(nobs, directed = FALSE)
        x <- rbinom(nobs, 1, 0.5)
        set.vertex.attribute(g, "x", x)

        # The formula is now just received as a string and converted
        ergm_formula <- as.formula(formula_from_python)
        cat("Using formula received from Python:", deparse(ergm_formula), "\\n")

     
        formula_with_g <- as.formula(paste("g", formula_from_python))
        stat_names <- names(summary(formula_with_g))
        cat("Statistic names (coef names expected):", stat_names, "\\n")

        # Dynamic coefficient scaling
        scale_lookup <- c(
            "edges"       = 2,
            "nodematch.x" = 2,
            "kstar2"      = 1/nobs,
            "kstar(2)"    = 1/nobs,
            "triangle"    = 1/nobs,
            "triangles"   = 1/nobs
        )

        # length match
        if (length(parameter) != length(stat_names)) {{
            stop(sprintf("Length mismatch: got %d parameters for %d stats (%s).",
                         length(parameter), length(stat_names), paste(stat_names, collapse=", ")))
        }}

        active_scales <- scale_lookup[stat_names]
   
        active_scales[is.na(active_scales)] <- 1

        scaled_coefs <- setNames(parameter * active_scales, stat_names)

        cat("Coefficient names:", names(scaled_coefs), "\\n")
        cat("Coefficient values:", scaled_coefs, "\\n")

	cat("class(g):", class(g), "\n")
        cat("is.network(g):", is.network(g), "\n")
        cat("class(formula_with_g):", class(formula_with_g), "\n")
        cat("env match:", identical(environment(formula_with_g), environment()), "\n")


        g0 <- simulate(
            formula_with_g, nsim = nsim, coef = scaled_coefs, basis = g,
            control = control.simulate(MCMC.burnin = 1e7, MCMC.interval = 1e4)
        )

  
        tobs_list <- list()
        for(i in 1:length(g0)) {{

            formula_with_network <- as.formula(paste("g0[[", i, "]]", formula_from_python))
            stats <- summary(formula_with_network)
            tobs_list[[i]] <- stats
        }}


        tobs <- do.call(rbind, tobs_list)


        if(ncol(tobs) == length(active_terms)) {{
            colnames(tobs) <- active_terms
        }} else {{
            cat("Warning: Number of statistics doesn't match active_terms\\n")
            cat("Stats returned:", ncol(tobs), "Active terms:", length(active_terms), "\\n")
        }}

        return(list(
            tobs    = tobs,
            x       = x,
            adjacen = lapply(g0, as.matrix)
        ))
    }}
    ''')

    r_graph_generator = robjects.globalenv['graph_generator']
    r_param = robjects.FloatVector(parameter)
    print("parameter: ", parameter)
    r_active_terms = robjects.StrVector(active_terms)
    print("active network statistics: ", active_terms)

    # Pass the pre-built formula string to the R function
    outcome = r_graph_generator(r_param, nobs, nsim, seed_number, r_active_terms,
                                formula_from_python=r_formula_string)

    end_generation = time.time()
    print("Generation time elapsed: ", end_generation - start_generation)

    tobs_tens = torch.tensor(np.array(outcome.rx2('tobs')), dtype=torch.float32)
    x_r = list(outcome.rx2('x'))
    xxx = torch.tensor(x_r, dtype=torch.float32).view(-1, 1)

    adjacency_list = []
    for j in range(len(list(outcome.rx2('adjacen')))):
        adjacency_list.append(torch.tensor(np.array(outcome.rx2('adjacen')[j])))
    adj_torch = torch.stack(adjacency_list, dim=0).to(torch.float32)

    return tobs_tens, xxx, adj_torch

def ergm_estimator(parameter, perturbation, nobs, adjacen, seed_number, active_terms):
    import rpy2.robjects as robjects
    ## To import R packages, rpy2 is providing a function rpy2.robjects.packages.importr()
    ## that makes that step very similar to importing python packages
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.packages import importr, data
    from rpy2.robjects import FloatVector


    assert len(parameter) == len(active_terms), "Length of parameter must match length of active_terms"
    
    # Build the R formula string here in Python
    term_map = {
            "edges": "edges",
            "nodematch": 'nodematch("x")',
            "kstar2": "kstar(2)",
            "triangle": "triangles",
            "triangles": "triangles"
    }
    
    formula_parts = [term_map.get(term, term) for term in active_terms]
    r_formula_string = "~ " + " + ".join(formula_parts)
    
    

    rng = np.random.default_rng(seed_number)                 # 복제마다 다른 seed를 주면 재현 가능
    

    A_r = robjects.r["matrix"](robjects.FloatVector(adjacen.flatten()), nrow=nobs)

    init_perturb = np.array(parameter) + rng.uniform(-perturbation, +perturbation, size = len(parameter))

    term_scale = {"edges": 2.0, "nodematch": 2.0,
                  "kstar2": 1.0 / nobs, "triangle": 1.0 / nobs, "triangles": 1.0 / nobs}
    scales = np.array([term_scale[t] for t in active_terms])

    init_point = robjects.FloatVector(init_perturb * scales)
    
    robjects.r('''
        library(network)
        library(ergm)
        fit_mple <- function(adj, formula_str) {
            g   <- network(adj, directed = FALSE)
            fit <- ergm(as.formula(paste("g", formula_str)), estimate = "MPLE")
            coef(fit)
        }
        fit_mle <- function(adj, formula_str, initial) {
            g   <- network(adj, directed = FALSE)
            fit <- ergm(as.formula(paste("g", formula_str)), estimate = "MLE",
                        control = control.ergm(main.method   = "Stochastic-Approximation",
                                               MCMC.burnin   = 100000,
                                               MCMC.interval = 1000,
                                               init          = initial))
            coef(fit)
        }
    ''')

    fit_mple = robjects.globalenv["fit_mple"]
    fit_mle = robjects.globalenv["fit_mle"]

    t0 = time.time()
    est_mple = np.array(fit_mple(A_r, r_formula_string))/scales
    est_duration_mple = time.time()-t0

    t1 = time.time()
    est_mle = np.array(fit_mle(A_r,r_formula_string, init_point))/scales
    est_duration_mle = time.time()-t1

    return est_mple, est_mle, init_perturb, est_duration_mple, est_duration_mle

def nodalattributes(x, ind, col):
    '''
        Purpose of this function:
        1. provide a binary output {0,1} when given two agents are linked each other
        2. provide a measure of homophily between two agents according to types of covariates:
            1) if a given covariate is binary {-1,1}, 1 if they match, 0 otherwise;
            2) if a given covariate is continuous, the measure is the absolute discrepancy between them.

        Parameters:
        x: covariate
        iu, ju: index of agents i, j
        graph: adjaceny matrix
        ind: types
            1) "edge" gives a binary outcome {0,1}, depending on whether given agents i and j are connected each other
            2) "binary" gives a binary outcome {0,1}, depending on whether their covariates match
            3) "continuous" gives the absolute difference between their covariates.

            Remark: One can add more properties of interest
    '''
    n = x.shape[0]
    link = torch.zeros([n,n], dtype = torch.float64)

    dimension = torch.arange(n)



    if ind == "binary":
   
        link = torch.triu((x[:, col].view(-1, 1) == x[:, col].view(1, -1)).float(), 1)
    elif ind == "diff":
        link = torch.triu(torch.abs(x[:, col].unsqueeze(0) - x[:, col].unsqueeze(1)),1)
    elif ind == "nodecov":
        link = torch.triu(x[:, col].unsqueeze(0) + x[:, col].unsqueeze(1),1)
    elif ind == "prod":
        link = torch.triu(x[:, col].unsqueeze(0) * x[:, col].unsqueeze(1),1)
    else:
        raise Exception("Nodal attribute is not defined.")

    return link + torch.transpose(link,0,1)



def symmetrization(v,n):

    '''
    v: a vector of elements within (0,1) with length of n*(n-1)/2
    n: the size of mu
    '''
    temp = torch.zeros((n,n)).to(dtype = v.dtype, device = v.device)
    row, col = torch.triu_indices(temp.size()[0],  temp.size()[0],1)

    temp[row, col]=v

    output = temp + torch.transpose(temp, 0,1)
    output.fill_diagonal_(0.0)
    return output 

def nedges(muu):

    return 1/2*torch.sum(muu).to(device)

def nodecov(x, ind, col, muu):
    x = x.to(device)
    return 1/2*torch.sum(nodalattributes(x,  ind, col)*muu)

def ntwo_star1(muu):
    return ((torch.sum(muu @muu)-torch.sum(torch.diag(muu @muu)))/ (2)).to(device)

def ntriad1(muu):
    return (torch.trace(muu @ muu @ muu)/(6)).to(device)


def kldiv(muu):
    nn = muu.size()[0]
    temp = torch.add(muu*torch.log(muu),
                     (1-muu)*torch.log(1-muu))
    temp = torch.sum(temp[~torch.isnan(temp)])/2
    return temp.to(device)

def l2regular(muu,epsilon):
    return (epsilon/2*torch.norm(muu, p='fro')**2).to(device)

############################################# gradient or change statistics
### They return upper triangle of mu under symmetry
def cw_derivative_edge(muu):
    return torch.triu(torch.ones_like(muu, dtype = torch.float64, device = device),1)

def cw_derivative_nodecov (x, ind, col, muu):
    return torch.triu(nodalattributes(x,  ind, col),1).to(dtype = torch.float64, device = device)

def cw_derivative_twostar(muu): 
    nn = muu.size(0)
    room = 1e-4

    d = torch.sum(muu, dim=1) 

    D_sum = d.unsqueeze(1) + d.unsqueeze(0)

    temp = (D_sum - 2*muu)
    temp.fill_diagonal_(0.0)
    temp = torch.triu(temp, 1).to(dtype = torch.float64, device = device)
    return temp


def cw_derivative_triad(muu): 
    n= muu.size()[0]
    temp=(muu @ muu)
    temp = torch.triu(temp,1).to(dtype = torch.float64, device = device)

    return temp
def cw_derivative_kldiv(muu):
    nn = muu.size()[0]
    room = 1e-4
    muu = muu.clamp(room, 1-room)
    temp = torch.log(muu/(1-muu))
    return torch.triu(temp,1).to(dtype = torch.float64, device = device)

# regularization term
def cw_derivative_regular(muu, epsilon):
    return 2*torch.triu(epsilon*muu,1).to(dtype = torch.float64, device = device)


def vec_to_str(vec):
    parts = []
    for v in vec:
        s = str(v)
        if '.' in s:
            s = s.replace('.', 'p')  
        parts.append(s)
    return "_".join(parts)

def tensor_str_to_array_p(s):
    """Parse a value that may already be numeric, or a string like
    'tensor(3.14)' / 'tensor(3.14, dtype=torch.float64)', into a NumPy array."""
    if isinstance(s, (int, float)):
        return np.array(s)
    s = str(s).strip()
   
    inner = s[s.find('(') + 1: s.rfind(')')]
    inner = re.split(r',\s*\w+\s*=', inner)[0]  

    lst = ast.literal_eval(inner)
   
    return np.array(lst)

def tensor_str_to_array_b(s):

    inner = s[s.find('['): s.rfind(']')+1]

    lst = ast.literal_eval(inner)

    return np.array(lst)

def summary_stat(dataframe, parameter, active_terms):

    all_thetas = [tensor_str_to_array_b(s) for s in dataframe.iloc[:,0]]
    stacked_theta = np.stack(all_thetas)

    mean_np = np.mean(stacked_theta,0)


    median_np = np.median(stacked_theta,0)
  
    var_np = np.var(stacked_theta,0, ddof=1)
    
    bias_np = np.array(parameter, dtype=np.float64) - mean_np

    se_np = np.sqrt(var_np)

    quant_005 = []
    quant_095 = []
    for d in range(stacked_theta.shape[1]):
        quant_005.append(np.quantile(stacked_theta[:,d],0.05))
        quant_095.append(np.quantile(stacked_theta[:,d],0.95))

  
    all_times = [tensor_str_to_array_p(s) for s in dataframe["time"]]

    stacked_time = np.stack(all_times)

    total_time_elapsed = np.sum(stacked_time)

    mean_time_elapsed = np.mean(stacked_time)

    median_time_elapsed = np.median(stacked_time)

    var_time_elapsed = np.var(stacked_time, ddof=1)
    se_time_elapsed = np.sqrt(var_time_elapsed)

    quant_005_time = np.quantile(stacked_time,0.05)
    quant_095_time = np.quantile(stacked_time,0.95)

    theta_stack = np.c_[bias_np, mean_np, median_np, se_np, np.array(quant_005), np.array(quant_095)].T

    time_stack = np.r_[total_time_elapsed, mean_time_elapsed,median_time_elapsed,se_time_elapsed,\
                                    quant_005_time, quant_095_time]


    df_construc = pd.DataFrame(np.c_[theta_stack, time_stack],
                               columns = list(active_terms) + ["time"])

    df_construc.index = ["bias","mean", "median",  "SE", "0.05","0.95"]
    return df_construc