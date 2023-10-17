from pydrake.geometry.optimization import IrisOptions
from functools import partial
import numpy as np
from pydrake.all import SceneGraphCollisionChecker
from region_generation import SNOPT_IRIS_ellipsoid
from visibility_clique_decomposition import VisCliqueInflation
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from visibility_logging import CliqueApproachLogger
from environments import get_environment_builder
    
N = 500
eps = 0.1
approach = 1
ap_names = ['redu', 'greedy', 'nx', 'cvx_hull']

max_iterations_clique = 10
extend_cliques = False
require_sample_point_is_contained = True
iteration_limit = 1
configuration_space_margin = 1.e-4
termination_threshold = -1
num_collision_infeasible_samples = 19
relative_termination_threshold = 0.02
pts_coverage_estimator = 5000
min_clique_size =10
seed = 1
np.random.seed(seed) 

cfg = {
    'seed': seed,
    'N': N,
    'eps': eps,
    'max_iterations_clique': max_iterations_clique,
    'min_clique_size': min_clique_size,
    'approach': approach,
    'extend_cliques': extend_cliques,
    'require_sample_point_is_contained':require_sample_point_is_contained,
    'iteration_limit': iteration_limit,
    'configuration_space_margin':configuration_space_margin,
    'termination_threshold':termination_threshold,
    'num_collision_infeasible_samples':num_collision_infeasible_samples,
    'relative_termination_threshold':relative_termination_threshold,
    'pts_coverage_estimator':pts_coverage_estimator}

env_builder = get_environment_builder('3DOFFLIPPER')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = env_builder(True)

robot_instances =[plant.GetModelInstanceByName("iiwaonedof"), plant.GetModelInstanceByName("iiwatwodof")]
checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                robot_model_instances = robot_instances,
                distance_function_weights =  [1] * plant.num_positions(),
                #configuration_distance_function = _configuration_distance,
                edge_step_size = 0.125)

scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) 
q_min = np.array([-2.6, -2., -1.9])
q_max = np.array([ 2.6,  2.,  1.9])
col_func_handle_ = get_col_func(plant, plant_context)
sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)
estimate_coverage = get_coverage_estimator(sample_cfree, pts = pts_coverage_estimator)
vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 

import pickle
import os
import time

name_vg= f"tmp/vg_test_cvx_hull_ell_3DOF_{seed}_{N}.pkl"
if os.path.exists(name_vg):
    with open(name_vg, 'rb') as f:
        d = pickle.load(f)
        pts = d['pts']
        ad_mat = d['ad_mat']
else:
    pts, _ = sample_cfree(N, 3000, [])
    ad_mat = (vgraph_handle(pts)).toarray()
    with open(name_vg, 'wb') as f:
        pickle.dump({'pts': pts, 'ad_mat': ad_mat}, f)

from clique_covers import (compute_greedy_clique_partition,
                           compute_greedy_clique_partition_convex_hull,
                           compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint)

d_min = 1e-2 
alpha_max = 0.81*np.pi/2
name_hyp = f"tmp/res_part_hyp_reduced_{N}_{seed}_{d_min}_{alpha_max}.pkl"
if os.path.exists(name_hyp):
    with open(name_hyp, 'rb') as f:
        d = pickle.load(f)
        cliques_hyp = d['cliques_hyp']
        t_hyp = d['t_hyp']
else:
    t1 = time.time()
    cliques_hyp = compute_greedy_clique_partition_convex_hull(ad_mat, pts, smin=10, mode='reduced')
    t2 = time.time()
    t_hyp = t2-t1
    with open(name_hyp, 'wb') as f:
        pickle.dump({'cliques_hyp': cliques_hyp, 't_hyp': t_hyp }, f)

name_hyp = f"tmp/res_part_hyp_full_{N}_{seed}.pkl"
if os.path.exists(name_hyp):
    with open(name_hyp, 'rb') as f:
        d = pickle.load(f)
        cliques_hyp_full = d['cliques_hyp']
        t_hyp_full = d['t_hyp']
else:
    t1 = time.time()
    cliques_hyp_full = compute_greedy_clique_partition_convex_hull(ad_mat, pts, smin=10, mode='full')
    t2 = time.time()
    t_hyp_full = t2-t1
    with open(name_hyp, 'wb') as f:
        pickle.dump({'cliques_hyp': cliques_hyp, 't_hyp': t_hyp }, f)

t1 = time.time()
cliques_e, emats = compute_greedy_clique_cover_w_ellipsoidal_convex_hull_constraint(ad_mat, pts, smin=10)
t2 = time.time()
t_e = t2-t1
t1 = time.time()
cliques = compute_greedy_clique_partition(ad_mat, min_cliuqe_size = 10)
t2 = time.time()
t_normal = t2-t1

results = [f"unconstrained {t_normal:.3f}s, {len(pts)} Vertices, {len(cliques)} Cliques, clique 1: {len(cliques[0])}", 
         f"ell constraint {t_e:.3f}s, {len(pts)} Vertices, {len(cliques_e)} Cliques, clique 1: {len(cliques_e[0])} ",
         f"hyp red constraint {t_hyp:.3f}s, {len(pts)} Vertices, {len(cliques_hyp)} Cliques, clique 1: {len(cliques_hyp[0])}",
         f"hyp full {t_hyp_full:.3f}s, {len(pts)} Vertices, {len(cliques_hyp_full)} Cliques, clique 1: {len(cliques_hyp_full[0])}",
         ]

for r in results:
    print(r)