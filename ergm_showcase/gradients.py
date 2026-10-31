from .utils import device, nodalattributes, symmetrization, nedges, nodecov, ntwo_star1, ntriad1, kldiv, l2regular, cw_derivative_edge, cw_derivative_nodecov, cw_derivative_twostar, cw_derivative_triad, cw_derivative_kldiv, cw_derivative_regular
import torch
import math
class customizedgradient_upper(torch.autograd.Function):  # l_n
    @staticmethod
    def forward(ctx, theta_tensor, mu_tensor, graph_tensor, eps, active_terms, xx_tensor = None, ind=None, col=None):
        '''
        arguments:
        theta_tensor: theta
        mu_tensor: vector of upper triangle of mu
        graph_tensor: adjacency matrix
        eps: regularization parameter
        active_terms:


        '''

        assert theta_tensor.ndim == 1, "theta must be a 1-D vector"
        assert theta_tensor.numel() == len(active_terms), "Length of parameter must match length of active_terms"
        assert active_terms[0] == "edges", "First active term must be 'edges' (unscaled)"

        room = 1e-4
        ## the following are for ctx to save objects for backward pass.
        ## The first ctx object is a boolean object that if mu_tensor is saved as a vectorized lower-triangle symmetric matrix (vech),
        ## then it returns True, False otherwise
        ctx.mu_is_vech = (mu_tensor.dim()==1)

        ## mu_tensor.shape returns an object "torch.Size," which may not be portable for ctx.
        ## Hence, tuple wraps the object for ctx to save it in a safer manner.
        ctx.mu_shape = tuple(mu_tensor.shape)

        ## The following is if statement to recover mu_sym from the ctx object.
        ## If mu_tensor is a vech, then it is a vectorized lower-triangle of a symmetric matrix \mu.
        ## Hence, using the number of elements mu_tensor.numel(), I recover it using "symmetrization" method.
        if ctx.mu_is_vech:
            mm = mu_tensor.numel()
            nn = int((1+math.sqrt(1+8*mm))/2)
            mu_sym = symmetrization(mu_tensor, nn)
        ## Otherwise, I take the first element in the torch.Size object and objectify mu_tensor as mu_sym.
        else:
            nn = mu_tensor.size()[0]
            mu_sym = mu_tensor

        ## The following line converts any NaN (Not a Number) values into 0.0
        ## positive infinity into 1-room, and negative infinity into room in the target matrix mu_sym.
        ## This in-place operation stabilizes the forward pass using mu_sym.
        modified_mu = torch.nan_to_num(mu_sym, nan=0.0, posinf=1 - room, neginf=room)

        ## reassures that it is projected onto between 0 and 1
        safe_mu = torch.clamp(modified_mu, min=room, max=1 - room)

        ## in-place operations modify the current object and its parents, which autograd fail to work.
        ## Hence I use .clone() to create an independent copy of the object to safely do in-place operation.
        safe_mu = safe_mu.clone()  # Avoid in-place op on views

        ## we symmetrize mu and fill the diagonal with 0, with torch.no_grad (avoid in-place operation)
        with torch.no_grad():

            safe_mu.fill_diagonal_(0.0)



        ## Graph-related computations. Note that two star and triangle methods do not have 1/n scaling.

        function_map = {"edges": nedges,
                "kstar2": ntwo_star1,
                "triangles": ntriad1}
        # Caveat: excluded_map: reserved for models with nodal covariates (e.g., nodecov).
        # Not used in current edge+triangle simulations.
        excluded_map =  {"nodecov": nodecov, #nodalattributes(x, ind, col)
                        "regular": l2regular}
        ## Since edge has different scaling, we need to append it first outside for-loop.
        assert active_terms[0] == "edges", "The first term of active terms must be edges."
        stat_edge = function_map[active_terms[0]](graph_tensor)
        stat_mu = function_map[active_terms[0]](safe_mu)
        ss_list = [stat_edge]
        mu_list = [stat_mu]
        if len(active_terms)>1:

            for t in active_terms[1:]:
                ss_list.append(function_map[t](graph_tensor)/nn)
                mu_list.append(function_map[t](safe_mu)/nn)



        ## stack all the network statistics into a tensor


        potential_graph = torch.stack(ss_list).to(dtype=torch.float64, device=theta_tensor.device)
        potential_mu    = torch.stack(mu_list).to(dtype=torch.float64, device=theta_tensor.device)


        ## Mu (lower-level variable)-related computations

        ## KL divergence
        kldiv_mu = kldiv(safe_mu)

        ## F_n with regularization
        output = -((theta_tensor.T @ potential_graph)
                   - ((theta_tensor.T @ potential_mu - kldiv_mu)-l2regular(safe_mu,eps)))/ (nn**2)
        #output = output.clone().detach().requires_grad_(True).to(device)


        ## save relevant tensors for backward process, where should be a right location to put this line of code?
        ctx.save_for_backward(theta_tensor, safe_mu, graph_tensor)

        ctx.xx = xx_tensor
        ctx.ind = ind
        ctx.col = col
        ctx.eps = eps
        ctx.active = active_terms
        return output

    @staticmethod
    def backward(ctx, grad_output):
        room = 1e-4


        theta_tensor, mu_tensor, graph_tensor = ctx.saved_tensors

        xx_tensor = ctx.xx
        ind = ctx.ind
        col = ctx.col
        eps = ctx.eps
        active_terms = ctx.active


        nn = mu_tensor.size()[0]


        safe_mu = mu_tensor.clamp(min = room, max = 1-room)


        # Custom gradient for theta
        function_map = {"edges": nedges,
                "kstar2": ntwo_star1,
                "triangles": ntriad1}
        excluded_map =  {"nodecov": nodecov, #nodalattributes(x, ind, col)
                        "regular": l2regular,
                        "kldiv": kldiv}


        grad_edge = -(function_map[active_terms[0]](graph_tensor) - function_map[active_terms[0]](safe_mu))/nn**2
        custom_grad_theta_list = [grad_edge]

        if len(active_terms[1:])!=0:

            for t,s in enumerate(active_terms[1:], start =1):
                custom_grad = -(function_map[active_terms[t]](graph_tensor) - function_map[active_terms[t]](safe_mu))/nn**2
                custom_grad_theta_list.append(custom_grad/nn)
   

        custom_grad_theta = torch.stack(custom_grad_theta_list).to(dtype=torch.float64, device = theta_tensor.device)

        if ctx.mu_is_vech:
            custom_grad_vech_mu = torch.zeros(ctx.mu_shape).to(dtype=torch.float64, device = theta_tensor.device)
        else:
            custom_grad_vech_mu = torch.zeros_like(safe_mu).to(dtype=torch.float64, device = theta_tensor.device)


        ## Make the diagonal of gradient of mu zero. Here torch.arange(mu.size) returns the array of 0 to n-1.
   

        return custom_grad_theta * grad_output, custom_grad_vech_mu * grad_output, None, None, None, None, None, None
   






######################################################
class customizedgradient_lower(torch.autograd.Function):  # Gamma_n
    @staticmethod
    def forward(ctx, theta_tensor, mu_tensor, eps, active_terms, xx_tensor = None, ind = None, col = None): #theta_tensor, mu_tensor, graph_tensor, eps, active_terms, xx_tensor = None, ind=None, col=None


        assert theta_tensor.ndim == 1, "theta must be a 1-D vector"
        assert theta_tensor.numel() == len(active_terms), "Length of parameter must match length of active_terms"
        assert active_terms[0] == "edges", "First active term must be 'edges' (unscaled)"

        ctx.mu_is_vech = (mu_tensor.dim()==1)
        ctx.mu_shape = tuple(mu_tensor.shape)

        if ctx.mu_is_vech:
            mm = mu_tensor.numel()
            nn = int((1+math.sqrt(1+8*mm))/2)
            mu_sym = symmetrization(mu_tensor, nn)
        ## projection: if NaN value is generated, then it goes to zero; if the value is positive infinite, then it is 1-1e-4; otherwise 1e-4
        else:
            nn = mu_tensor.size()[0]
            mu_sym = mu_tensor
        room = 1e-4



        modified_mu = torch.nan_to_num(mu_sym, nan=0.0, posinf=1 - room, neginf=room)

        ## reassures that it is projected onto between 0 and 1
        safe_mu = torch.clamp(modified_mu, min=room, max=1 - room)

        safe_mu = safe_mu.clone()  # Avoid in-place op on views

        ## we symmetrize mu and fill the diagonal with 0, with torch.no_grad (avoid in-place operation)
        with torch.no_grad():
     
            safe_mu.fill_diagonal_(0.0)


        function_map = {"edges": nedges,
                "kstar2": ntwo_star1,
                "triangles": ntriad1}
        excluded_map =  {"nodecov": nodecov, #nodalattributes(x, ind, col)
                        "regular": l2regular,
                        "kldiv": kldiv}

        stat_mu = function_map[active_terms[0]](safe_mu)

        mu_list = [stat_mu]
        if len(active_terms[1:])!=0:

            for t,s in enumerate(active_terms[1:], start =1):

                mu_list.append(function_map[active_terms[t]](safe_mu)/nn)
        

        potential_mu    = torch.stack(mu_list).to(dtype=torch.float64, device=theta_tensor.device)

        kldiv_mu = kldiv(safe_mu)

        output = - ((theta_tensor.T @ potential_mu - kldiv_mu)-l2regular(safe_mu,eps))/ (nn**2)



        mask = torch.triu(torch.ones_like(safe_mu, dtype=torch.bool), diagonal=1)

        vech_mu = safe_mu[mask]

        ctx.save_for_backward(theta_tensor, vech_mu)
        ctx.nn = nn
        ctx.xx = xx_tensor
        ctx.ind = ind
        ctx.col = col
        ctx.eps = eps
        ctx.active = active_terms
        return output

    @staticmethod
    def backward(ctx, grad_output):


        theta_tensor, vech_mu = ctx.saved_tensors

        nn = ctx.nn
        xx_tensor = ctx.xx
        ind = ctx.ind
        attr_col = ctx.col
        eps = ctx.eps
        active_terms = ctx.active

        mu_sym = symmetrization(vech_mu, nn)
        room = 1e-4
        safe_mu = torch.clamp(mu_sym, min=room, max=1 - room)

        safe_mu = safe_mu.clone()  # Avoid in-place op on views

        ## we symmetrize mu and fill the diagonal with 0, with torch.no_grad (avoid in-place operation)
        with torch.no_grad():
         
            safe_mu.fill_diagonal_(0.0)

        ## gradient with respect to theta
        function_map = {"edges": nedges,
                "kstar2": ntwo_star1,
                "triangles": ntriad1}
        excluded_function =  {"nodecov": nodecov, #nodalattributes(x, ind, col)
                        "regular": l2regular,
                        "kldiv": kldiv}

        grad_edge = -function_map[active_terms[0]](safe_mu)/nn**2
        custom_grad_theta_list = [grad_edge]

        if len(active_terms[1:])!=0:

            for t,s in enumerate(active_terms[1:], start =1):
                custom_grad =  - function_map[active_terms[t]](safe_mu)/nn**2
                custom_grad_theta_list.append(custom_grad/nn)


        custom_grad_theta = torch.stack(custom_grad_theta_list).to(dtype=torch.float64, device=theta_tensor.device)

        ## gradient with respect to mu
        gradient_map = {"edges": cw_derivative_edge,
                "kstar2": cw_derivative_twostar,
                "triangles": cw_derivative_triad}


        grad_mu_edge = gradient_map[active_terms[0]](safe_mu)
        custom_grad_mu_list = [grad_mu_edge]

        if len(active_terms[1:])!=0:

            for t,s in enumerate(active_terms[1:], start =1):
                custom_grad = (gradient_map[active_terms[t]](safe_mu))
                custom_grad_mu_list.append(custom_grad/nn)
        #        print(f"customized gradient with respect to mu for {s} computed and appended")

        grad_mu = torch.stack(custom_grad_mu_list).to(dtype=torch.float64, device=theta_tensor.device)
        #print(grad_mu.size())

        excluded_grad = {"kldiv": cw_derivative_kldiv,
                "regular": cw_derivative_regular}

        grad_kl = (cw_derivative_kldiv(safe_mu))
        grad_l2 = (cw_derivative_regular(safe_mu, eps))


        custom_grad_mu = -(((theta_tensor.view(-1,1,1)*grad_mu).sum(dim=0) - grad_kl)-grad_l2)/nn**2
   
        if ctx.mu_is_vech:
            row, col = torch.triu_indices(nn, nn, 1)
            custom_grad_vech_mu = custom_grad_mu[row,col]
            custom_grad_vech_mu = custom_grad_vech_mu.clone().to(dtype=torch.float64, device=theta_tensor.device)   # Create a new copy to avoid modifying inplace
        else:
            custom_grad_vech_mu = custom_grad_mu.clone()# custom_grad_mu + custom_grad_mu.T
            custom_grad_vech_mu.fill_diagonal_(0.0)


        custom_grad_theta = custom_grad_theta.to(dtype=torch.float64, device=theta_tensor.device)


        ## the number of outputs in backward must be equal to the number of inputs in forward.
        return custom_grad_theta * grad_output, custom_grad_vech_mu * grad_output, None, None, None, None, None