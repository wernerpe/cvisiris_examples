from independent_set_solver import solve_max_independent_set_integer, solve_max_independet_set_KAMIS
from ellipse_utils import get_lj_ellipse
from pydrake.all import Hyperellipsoid
import numpy as np
import networkx as nx
import subprocess

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
    binary_loc = "/home/peter/git/ExtensionCC_test/ExtensionCC/out/optimized/vcc "
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

def compute_greedy_clique_partition(adj_mat, min_cliuqe_size):
    cliques = []
    done = False
    adj_curr = adj_mat.copy()
    adj_curr = 1- adj_curr
    np.fill_diagonal(adj_curr, 0)
    ind_curr = np.arange(len(adj_curr))
    while not done:
        val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
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

def compute_greedy_clique_edge_cover(adj_mat, min_cliuqe_size):
    cliques = []
    done = False
    adj_curr = adj_mat.copy()
    adj_curr = 1- adj_curr
    np.fill_diagonal(adj_curr, 0)
    ind_curr = np.arange(len(adj_curr))
    #we compute a weighted max independent set, with the weigths

    #failure case max independent set = max nr edges -> only adding m^2 - previously covered edges, 
    # need to subtract already covered edges from cost function? yes! but how
    # cost = -m(m-1)/2 - covered edges vi*vj*weight
    # naive implementation keeps the size of the program constant and 
    # marks off all edges covered by previous cliques in a upper triagular matrix
    # a better implementation will remove vertices from the program if all its outgoing edges have been covered
    

    while not done:
        val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr) #solve_max_independet_set_KAMIS(adj_curr, maxtime = 5) #
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

# def compute_greedy_clique_partition_edge_removal(adj_mat):
#     cliques = []
#     done = False
#     adj_curr = adj_mat.copy()
#     adj_curr = 1- adj_curr
#     np.fill_diagonal(adj_curr, 0)
#     ind_curr = np.arange(len(adj_curr))
#     while not done:
#         val, ind_max_clique_local = solve_max_independent_set_integer(adj_curr)
#         #non_max_ind_local = np.arange(len(adj_curr))
#         #non_max_ind_local = np.delete(non_max_ind_local, ind_max_clique_local, None)
#         index_max_clique_global = np.array([ind_curr[i] for i in ind_max_clique_local])
#         cliques.append(index_max_clique_global.reshape(-1))
#         adj_curr = np.delete(adj_curr, ind_max_clique_local, 0)
#         adj_curr = np.delete(adj_curr, ind_max_clique_local, 1)
#         ind_curr = np.delete(ind_curr, ind_max_clique_local)
#         if len(adj_curr) == 0:
#             done = True
#     return cliques

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