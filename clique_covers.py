from independent_set_solver import solve_max_independent_set_integer, solve_max_independet_set_KAMIS
from ellipse_utils import (get_lj_ellipse, 
                           get_lj_ellipse_homogeneous_rep, 
                           arrange_homogeneous_ellipse_matrix_to_vector, 
                           build_quadratic_features)
from pydrake.all import Hyperellipsoid, Solve, CommonSolverOption, MathematicalProgram, SolverOptions
import numpy as np
import networkx as nx
import subprocess

_REDUVCCPATH = "/home/peter/git/ExtensionCC_test/ExtensionCC"

def networkx_to_metis_format(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    metis_lines = [f"{num_nodes} {num_edges} {0}\n"]
    
    for node in range(num_nodes):
        neighbors = " ".join(str(neighbor + 1) for neighbor in graph.neighbors(node))
        metis_lines.append(neighbors + "\n")
    
    return metis_lines


def compute_cliques_REDUVCC(ad_mat, maxtime = 30):
    #this is messed up, nx adds self edges when initializing from sparse matrix
    nx_graph = nx.Graph(ad_mat.toarray())
    metis_lines = networkx_to_metis_format(nx_graph)
    edges = 0
    for i in range(ad_mat.shape[0]):
        #for j in range(i+1, ad_mat.shape[0]):
        edges+=np.sum(ad_mat[i, i+1:])    
    with open("tmp/vgraph.metis", "w") as f:
        f.writelines(metis_lines)
        f.flush()  # Flush the buffer to ensure data is written immediately
        f.close()
    binary_loc = f"{_REDUVCCPATH}/out/optimized/vcc "
    options = f"--solver_time_limit={maxtime} --seed=5 --run_type=ReduVCC --output_cover_file=tmp/cliques.txt "
    file = "tmp/vgraph.metis"
    command = binary_loc + options + file
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    print(str(str(output)[2:-1]).replace('\\n', '\n '))
    with open("tmp/cliques.txt", "r") as f:
        cliques_1_index = f.readlines()
    cliques_1_index = [c.split(' ') for c in cliques_1_index]
    cliques = [np.array([int(c)-1 for c in cli]) for cli in cliques_1_index]
    cliques = sorted(cliques, key=len)[::-1]
    return cliques

def compute_greedy_clique_partition(adj_mat, min_cliuqe_size, worklimit =100):
    cliques = []
    done = False
    adj_curr = adj_mat.copy()
    adj_curr = 1- adj_curr
    np.fill_diagonal(adj_curr, 0)
    ind_curr = np.arange(len(adj_curr))
    while not done:
        val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr, worklimit=worklimit) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
        #non_max_ind_local = np.arange(len(adj_curr))
        #non_max_ind_local = np.delete(non_max_ind_local, ind_max_clique_local, None)
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        cliques.append(index_max_clique_global.reshape(-1))
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 0)
        adj_curr = np.delete(adj_curr, ind_max_clique_local, 1)
        ind_curr = np.delete(ind_curr, ind_max_clique_local)
        if len(adj_curr) == 0 or len(cliques[-1])<min_cliuqe_size:
            done = True
    return cliques

from ellipse_utils import switch_ellipse_description

def compute_outer_LJ_sphere(pts):
    dim = pts[0].shape[0]
    # pts = #[pt1, pt2]
    # for _ in range(2*dim):
    #     m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
    #     pts.append(m)
    upper_triangular_indeces = []
    for i in range(dim-1):
        for j in range(i+1, dim):
            upper_triangular_indeces.append([i,j])

    upper_triangular_indeces = np.array(upper_triangular_indeces)
    prog = MathematicalProgram()
    inv_radius = prog.NewContinuousVariables(1, 'rad')
    A = inv_radius*np.eye(dim)
    b = prog.NewContinuousVariables(dim, 'b')
    prog.AddMaximizeLogDeterminantCost(A)
    for idx, pt in enumerate(pts):
        pt = pt.reshape(dim,1)
        S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
        prog.AddPositiveSemidefiniteConstraint(S)
        prog.AddLinearEqualityConstraint(S[0,0] == 0.9)
        v = (A@pt + b.reshape(dim,1)).T
        c = (S[1:,1:]-np.eye(dim)).reshape(-1)
        for idx in range(dim):
            prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
        for ci in c:
            prog.AddLinearEqualityConstraint(ci, 0 )

    prog.AddPositiveSemidefiniteConstraint(A) # eps * identity

    # for aij in A[upper_triangular_indeces[:,0], upper_triangular_indeces[:,1]]:
    #     prog.AddLinearConstraint(aij == 0)
    prog.AddPositiveSemidefiniteConstraint(10000*np.eye(dim)-A)

    sol = Solve(prog)
    if sol.is_success():
        HE, _, _ =switch_ellipse_description(sol.GetSolution(inv_radius)*np.eye(dim), sol.GetSolution(b))
    return HE

from pydrake.all import GurobiSolver

def max_clique_w_ellipsoidal_cvx_hull_constraint(adj_mat, 
                                                 graph_vertices, 
                                                 c=None, 
                                                 min_eig = 1e-3, 
                                                 max_eig = 5e-2, 
                                                 r_scale = 1.0, 
                                                 M_vals = None,
                                                 worklimit = 400,
                                                 warm_start = True,
                                                 use_sdd = False):
    """ 
    adj_mat: nxn {0,1} binary adjacency matrix
    graph_vertices: nxdim vertex locations
    c: nx1 cost vector for the vertices (used for computing covers)
    min_eig: minimum eigen value of decision boundary
    max_eig: maximum eigen value of decision boundary
    M_vals: nx1 setting this vector overrides the BigM values 
    """
    add_extra = True

    assert adj_mat.shape[0] == len(graph_vertices)
    assert r_scale>=0.5
    
    #assert graph_vertices[0, :].shape[0] == points_to_exclude.shape[1]
    dim = graph_vertices.shape[1]
    n = adj_mat.shape[0]
    if M_vals is None:
        #compute radius of circumscribed sphere of all points to get margin size
        HS = compute_outer_LJ_sphere(graph_vertices)
        radius = 1/(HS.A()[0,0]+1e-6)
        center = HS.center()
        dists = np.linalg.norm((graph_vertices-center.reshape(1,-1)), axis=1)
        M_vals = max_eig*(dists+r_scale*radius)**2
    else:
        assert M_vals.shape[0] ==n 

    fq = build_quadratic_features(graph_vertices)
    if c is None:
        c = np.ones((n,))
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    Emat = prog.NewSymmetricContinuousVariables(dim+1)
    hE = arrange_homogeneous_ellipse_matrix_to_vector(Emat)
    prog.AddLinearCost(-np.sum(c*v))

    for i in range(0,n):
        for j in range(i+1,n):
            if adj_mat[i,j] == 0:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)

    for i in range(n):
        val = hE.T@fq[i,:]
        prog.AddLinearConstraint(val>=1-v[i])
        prog.AddLinearConstraint(val<=1+M_vals[i]*(1-v[i])) #

    #force non-trivial solutions
    pd_amount = min_eig *np.eye(dim)
    if use_sdd:
        prog.AddScaledDiagonallyDominantMatrixConstraint(Emat[:-1, :-1]-pd_amount)
    else:
        prog.AddPositiveDiagonallyDominantMatrixConstraint(Emat[:-1, :-1]-pd_amount)    
 
    if add_extra:
        lamb = prog.NewContinuousVariables(n,n)
        eq_cons = lamb@graph_vertices - np.diag(v)@graph_vertices
        for row in eq_cons:
            for c in row:
                prog.AddLinearConstraint(c ==0)

        for i in range(n):
            for j in range(n):
                prog.AddLinearConstraint(lamb[i,j] >=0)
                prog.AddLinearConstraint(lamb[i,j] <=v[j])

        sumlamb = np.sum(lamb, axis =1)
        cons_sum = sumlamb-v
        for c in cons_sum:
            prog.AddLinearConstraint(c==0)
        prog.AddLinearCost(-np.sum(lamb))
        
    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    solver_options.SetOption(GurobiSolver.id(), 'WorkLimit', worklimit)
    



    if warm_start:
        adj_inv = adj_mat.copy()
        adj_inv = 1- adj_inv
        np.fill_diagonal(adj_inv, 0)
        ind_curr = np.arange(len(adj_inv))
        val, max_clique = solve_max_independent_set_integer(adj_inv)
        points_max_clique = graph_vertices[max_clique] 
        E_guess = get_lj_ellipse_homogeneous_rep(points_max_clique)
        v_guess= np.zeros(n)
        v_guess[max_clique] =1
        prog.SetInitialGuess(v, v_guess)
        prog.SetInitialGuess(Emat, E_guess)
    result = Solve(prog, solver_options=solver_options)
    
    print(result.is_success())
    return  -result.get_optimal_cost(), np.where(np.abs(result.GetSolution(v)-1)<=1e-4)[0], result.GetSolution(Emat), result.GetSolution(v), M_vals

def compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint(adj_mat, 
                                                                     pts, 
                                                                     smin =10, 
                                                                     max_aspect_ratio = 8, 
                                                                     r_scale = 1.0, 
                                                                     worklimit = 400, 
                                                                     warm_start = False,
                                                                     use_sdd = False):
    assert adj_mat.shape[0] == len(pts)
    print(f"[CLIQUECOVERELLIPSOID] Using {'Scaled 'if use_sdd else''}DD Constraint")
    LJS = compute_outer_LJ_sphere(pts)
    radius = 1/(LJS.A()[0,0]+1e-6)
    min_eig = radius/10 * 1e-3
    max_eig = max_aspect_ratio*min_eig
    #radius = 1/(HS.A()[0,0]+1e-6)
    center = LJS.center()
    dists = np.linalg.norm((pts-center.reshape(1,-1)), axis=1)
    M_vals = max_eig*(dists+r_scale*radius)**2

    cliques = []
    done = False
    pts_curr = pts.copy()
    adj_curr = adj_mat.copy()
    ind_curr = np.arange(len(adj_curr))
    c = np.ones((adj_mat.shape[0],))
    boundaries = []
    while not done:
        val, ind_max_clique_local,dec_boundary,_,_ = max_clique_w_ellipsoidal_cvx_hull_constraint(adj_curr, 
                                                                                                  pts_curr, 
                                                                                                  c,  
                                                                                                  min_eig, 
                                                                                                  max_eig, 
                                                                                                  r_scale, 
                                                                                                  M_vals = M_vals, 
                                                                                                  worklimit=worklimit,
                                                                                                  warm_start=warm_start,
                                                                                                  use_sdd = use_sdd)
        boundaries+= [dec_boundary]
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        c[ind_max_clique_local] = 0
        cliques.append(index_max_clique_global.reshape(-1))
        if val< smin:
            done = True
    return cliques, boundaries

def max_clique_w_cvx_hull_constraint_reduced(adj_mat, 
                                             graph_vertices, 
                                             c = None, 
                                             d_min = 1e-2, 
                                             alpha_max = 0.85*np.pi/2, 
                                             c_bounds = None,
                                             Mij = None):
    assert adj_mat.shape[0] == len(graph_vertices)
    assert alpha_max>=0 and  alpha_max<= 0.99*np.pi/2 
    #assert graph_vertices[0, :].shape[0] == points_to_exclude.shape[1]
    
    dim = graph_vertices.shape[1]
    #compute radius of circumscribed sphere of all points to get soft margin size
    # HS = compute_outer_LJ_sphere(graph_vertices)
    # radius = 3.5*1/(HS.A()[0,0]+1e-6)
    n = adj_mat.shape[0]
    if c is None:
        c = np.ones((n,))
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    prog.AddLinearCost(-np.sum(c*v))
    
    #hyperplanes
    ci = prog.NewContinuousVariables(n, dim)
    if c_bounds is None and Mij is None:
        c_bounds = np.zeros(n)#1/(np.sqrt(dim)*d_min*np.ones((n,)))#
        for i in range(n):
            dists = np.linalg.norm(graph_vertices - graph_vertices[i, :].reshape(1,-1), axis = 1) 
            dists_red = np.delete(dists, i)
            d_lower = np.max([np.min(dists_red), d_min])
            c_bounds[i] = 1/(np.cos(alpha_max)*d_lower)

        Mij = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                Mij[i,j] = np.sqrt(dim)*c_bounds[i]*np.linalg.norm(graph_vertices[i,:]-graph_vertices[j,:]) + 1
                Mij[j,i] = np.sqrt(dim)*c_bounds[j]*np.linalg.norm(graph_vertices[i,:]-graph_vertices[j,:]) + 1
    else:
        assert c_bounds.shape[0] == n and not np.any(c_bounds<0)
        assert Mij.shape[0]==Mij.shape[1] and Mij.shape[0]==n and not np.any(Mij<0)
    #Mij = cbd*np.abs(graph_vertices@graph_vertices.T) #+4*cbd
    
    for i in range(0,n):
        for j in range(i+1,n):
            if adj_mat[i,j] == 0:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)


    for i in range(0,n):
        for d in range(dim):
            prog.AddLinearConstraint(ci[i,d] <= c_bounds[i])
            prog.AddLinearConstraint(ci[i,d] >= -c_bounds[i])

    for i in range(n):
        for j in range(n):
            if i!=j:
                cons = (graph_vertices[j, :] - graph_vertices[i,:])@ci[i,:] - (1-v[i]) + Mij[i,j]*(1-v[j])
                prog.AddLinearConstraint(cons >=0)

    solver_options = SolverOptions()
    #solver_options.SetOption(GurobiSolver.id(), 'OptimalityTol', 1e-6)
    #solver_options.SetOption(GurobiSolver.id(), 'MIPGap', 1e-6)
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    result = Solve(prog, solver_options=solver_options)
    print(result.is_success())
    return -result.get_optimal_cost(), np.where(result.GetSolution(v)>=0.9)[0]

def max_clique_w_cvx_hull_constraint(adj_mat, graph_vertices, c = None):
    assert adj_mat.shape[0] == len(graph_vertices)
    #assert graph_vertices[0, :].shape[0] == points_to_exclude.shape[1]
    
    dim = graph_vertices.shape[1]
    #compute radius of circumscribed sphere of all points to get soft margin size
    HS = compute_outer_LJ_sphere(graph_vertices)
    radius = 3.5*1/(HS.A()[0,0]+1e-6)
    n = adj_mat.shape[0]
    if c is None:
        c = np.ones((n,))
    prog = MathematicalProgram()
    v = prog.NewBinaryVariables(n)
    prog.AddLinearCost(-np.sum(c*v))
    
    #hyperplanes
    lambdas = prog.NewContinuousVariables(n, dim+1)
    #slack variables for soft margins
    gammas = prog.NewContinuousVariables(n, n)

    Points_mat = np.concatenate((graph_vertices,np.ones((n,1))), axis =1)
    #Exclusion_points_mat =  np.concatenate((points_to_exclude,np.ones((num_points_to_exclude,1))), axis =1)

    for i in range(0,n):
        for j in range(i+1,n):
            if adj_mat[i,j] == 0:
                prog.AddLinearConstraint(v[i] + v[j] <= 1)

    for i in range(n):
        constraint1 = -Points_mat@lambdas[i,:]+2*radius*gammas[i,:]
        constraint2 = Points_mat[i,:]@lambdas[i,:]  #+ np.sum(gammas)
        for k in range(n):
            prog.AddLinearConstraint(constraint1[k] >=0)

        prog.AddLinearConstraint(constraint2>=1-v[i]) #

    for i in range(n):
        gammas_point_i = gammas[i, :]    
        for vi, gi in zip(v, gammas_point_i):
            prog.AddLinearConstraint(gi >= (vi-1))

        for vi,gi in zip(v, gammas_point_i):
            prog.AddLinearConstraint((1-vi)>= gi )


    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    result = Solve(prog, solver_options=solver_options)
    print(result.is_success())
    return -result.get_optimal_cost(), np.where(result.GetSolution(v)>=0.9)[0]

def compute_greedy_clique_partition_convex_hull(adj_mat, pts, smin = 10, mode = 'reduced', d_min = 1e-2, alpha_max = 0.85*np.pi/2):
    assert adj_mat.shape[0] == len(pts)
    assert mode in ['reduced', 'full']
    cliques = []
    done = False
    pts_curr = pts.copy()
    adj_curr = adj_mat.copy()
    ind_curr = np.arange(len(adj_curr))
    c = np.ones((adj_mat.shape[0],))

    if mode=='reduced':
        #precompute bigM values and c_bounds
        n = adj_mat.shape[0]
        dim = pts.shape[1]
        c_bounds = np.zeros(n)#1/(np.sqrt(dim)*d_min*np.ones((n,)))#
        for i in range(n):
            dists = np.linalg.norm(pts - pts[i, :].reshape(1,-1), axis = 1) 
            dists_red = np.delete(dists, i)
            d_lower = np.max([np.min(dists_red), d_min])
            c_bounds[i] = 1/(np.cos(alpha_max)*d_lower)

        Mij = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                Mij[i,j] = np.sqrt(dim)*c_bounds[i]*np.linalg.norm(pts[i,:]-pts[j,:]) + 1
                Mij[j,i] = np.sqrt(dim)*c_bounds[j]*np.linalg.norm(pts[i,:]-pts[j,:]) + 1

    while not done:
        if mode == 'reduced':
            val, ind_max_clique_local = max_clique_w_cvx_hull_constraint_reduced(adj_curr, pts_curr, c, d_min, alpha_max, c_bounds=c_bounds, Mij=Mij)
        else:
            val, ind_max_clique_local = max_clique_w_cvx_hull_constraint(adj_curr, pts_curr,c)
        index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
        c[ind_max_clique_local] = 0
        cliques.append(index_max_clique_global.reshape(-1))
        if val< smin:
            done = True
    return cliques

# def compute_greedy_clique_edge_cover(adj_mat, min_cliuqe_size):
#     cliques = []
#     done = False
#     adj_curr = adj_mat.copy()
#     adj_curr = 1- adj_curr
#     np.fill_diagonal(adj_curr, 0)
#     ind_curr = np.arange(len(adj_curr))
#     #we compute a weighted max independent set, with the weigths

#     #failure case max independent set = max nr edges -> only adding m^2 - previously covered edges, 
#     # need to subtract already covered edges from cost function? yes! but how
#     # cost = -m(m-1)/2 - covered edges vi*vj*weight
#     # naive implementation keeps the size of the program constant and 
#     # marks off all edges covered by previous cliques in a upper triagular matrix
#     # a better implementation will remove vertices from the program if all its outgoing edges have been covered
    

#     while not done:
#         val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
#         #non_max_ind_local = np.arange(len(adj_curr))
#         #non_max_ind_local = np.delete(non_max_ind_local, ind_max_clique_local, None)
#         index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
#         cliques.append(index_max_clique_global.reshape(-1))
#         adj_curr = np.delete(adj_curr, ind_max_clique_local, 0)
#         adj_curr = np.delete(adj_curr, ind_max_clique_local, 1)
#         ind_curr = np.delete(ind_curr, ind_max_clique_local)
#         if len(adj_curr) == 0 or len(cliques[-1])<min_cliuqe_size:
#             done = True
#     return cliques

def find_clique_neighbors(adj_mat, clique):
    nei = []
    cl_mem = set([i for i in clique])
    for c in clique:
        all_nei = set([i for i in np.where(adj_mat[c,:] ==1)[0]])
        nei.append(all_nei - (all_nei&(cl_mem -set(np.array([c])))))
    nei_clique = nei[0]
    if len(clique)>1:
        for n in nei[1:]:
            nei_clique = nei_clique&n
    return np.array(list(nei_clique))

def extend_cliques(adj_mat, cliques):
    extended_cliques = []
    for clique in cliques:
        idx_neighbors = find_clique_neighbors(adj_mat, clique)
        ## there is a bug here
        if len(idx_neighbors):
            ad_nei = adj_mat[idx_neighbors,:]
            ad_nei = ad_nei[:,idx_neighbors]
            ad_inv = 1-1.0*ad_nei
            np.fill_diagonal(ad_inv, 0)
            _, indx_nei_clique = solve_max_independent_set_integer(ad_inv)
            idx_extend = [idx_neighbors[i] for i in indx_nei_clique] 
            extended_cliques.append(np.array( clique.tolist() + idx_extend))
        else:
            extended_cliques.append(clique)
    return extended_cliques

def compute_minimal_clique_partition_nx(adj_mat):
    n = len(adj_mat)

    adj_compl = 1- adj_mat
    np.fill_diagonal(adj_compl, 0)
    graph = nx.Graph(adj_compl)
    sol = nx.greedy_color(graph, strategy='largest_first', interchange=True)

    colors= [sol[i] for i in range(n)]
    unique_colors = list(set(colors))
    cliques = []
    nr_cliques = len(unique_colors)
    for col in unique_colors:
        cliques.append(np.where(np.array(colors) == col)[0])
    return cliques

def get_iris_metrics(cliques, collision_handle):
    #seed_ellipses = [get_lj_ellipse(k) for k in cliques]
    seed_ellipses = [Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(k.T, rank_tol = 1e-12) for k in cliques]
    seed_points = []
    for k,se in zip(cliques, seed_ellipses):
        center = se.center()
        dim = len(se.center())
        if collision_handle(center):
            distances = np.linalg.norm(np.array(k).reshape(-1,dim) - center, axis = 1).reshape(-1)
            mindist_idx = np.argmin(distances)
            seed_points.append(k[mindist_idx])
        else:
            seed_points.append(center)

    #rescale seed_ellipses
    mean_eig_scaling = 1000
    seed_ellipses_scaled = []
    for e in seed_ellipses:
        eigs, _ = np.linalg.eig(e.A())
        mean_eig_size = np.mean(eigs)
        seed_ellipses_scaled.append(Hyperellipsoid(e.A()*(mean_eig_scaling/mean_eig_size), e.center()))
    #sort by size
    #idxs = np.argsort([s.Volume() for s in seed_ellipses])[::-1]
    hs = seed_points#[seed_points[i] for i in idxs]
    se = seed_ellipses_scaled #[seed_ellipses_scaled[i] for i in idxs]
    return hs, se, seed_ellipses