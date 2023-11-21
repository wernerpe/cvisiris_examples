import numpy as np
from functools import partial
import ipywidgets as widgets
from IPython.display import display

from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.all import (PiecewisePolynomial, 
                         InverseKinematics, 
                         Sphere, 
                         Rgba, 
                         RigidTransform, 
                         RotationMatrix, 
                         Solve,
                         MathematicalProgram,
                         RollPitchYaw,
                         Cylinder)
import time
import pydrake

from environments import get_environment_builder
from visualization_utils import plot_points, plot_regions
from pydrake.all import VPolytope, Role
# from task_space_seeding_utils import (get_cvx_hulls_of_bodies, 
#                             #          get_task_space_sampler,
#                              #         get_ik_problem_solver)

from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from region_generation import SNOPT_IRIS_ellipsoid_parallel
from pydrake.all import SceneGraphCollisionChecker
from visibility_logging import CliqueApproachLogger
from visibility_clique_decomposition import VisCliqueInflation

plant_builder = get_environment_builder('7DOFBINS')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)

scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
    diagram_context)

geom_names = ['bin_base', 'bin_base', 'shelves_body']
model_names = ['binL', 'binR', 'shelves']
#cvx_hulls_of_ROI, bodies = get_cvx_hulls_of_bodies(geom_names, model_names, plant, scene_graph, scene_graph_context)
#plot_regions(meshcat, cvx_hulls_of_ROI, opacity=0.2)

# ik_solver_S = get_ik_problem_solver(plant, 
#                                   plant_context, 
#                                   [plant.GetFrameByName('body')],
#                                   [np.array([0,0.1,0])], 
#                                   collision_free=True,
#                                   track_orientation=True)
q0  = np.zeros(7)
plant.SetPositions(plant_context, q0)
plant.ForcedPublish(plant_context)
t0 = plant.EvalBodyPoseInWorld(plant_context,  plant.GetBodyByName("body")).translation()       

seed = 1
N = 1500
eps = 0.4
ts_fraction = 0.0
max_iterations_clique = 10
min_clique_size = 14
approach = 1
ap_names = ['redu', 'greedy', 'nx', 'cvxh', 'cvxh_ell']
extend_cliques = False

require_sample_point_is_contained = True
iteration_limit = 1
configuration_space_margin = 1.e-3
termination_threshold = -1
num_collision_infeasible_samples = 35
relative_termination_threshold = 0.02

pts_coverage_estimator = 5000
cfg = {'seed': seed,
        'N': N,
        'eps': eps,
        'use_ts': True,
        'ts_fraction': ts_fraction,
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

q_min = plant.GetPositionLowerLimits()*1
q_max =  plant.GetPositionUpperLimits()*1

col_func_handle_ = get_col_func(plant, plant_context)
sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)
# sample_handle_ts = get_task_space_sampler(cvx_hulls_of_ROI, 
#                                           ik_solver_S,
#                                           q0,
#                                           t0,
#                                           collision_free=True
#                                           )
estimate_coverage = get_coverage_estimator(sample_cfree, pts = pts_coverage_estimator)

snopt_iris_options = IrisOptions()
snopt_iris_options.require_sample_point_is_contained = require_sample_point_is_contained
snopt_iris_options.iteration_limit = iteration_limit
snopt_iris_options.configuration_space_margin = configuration_space_margin
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = termination_threshold
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.num_collision_infeasible_samples = num_collision_infeasible_samples
snopt_iris_options.relative_termination_threshold = relative_termination_threshold
robot_instances = [plant.GetModelInstanceByName("iiwa"), plant.GetModelInstanceByName("wsg")]

checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                    robot_model_instances = robot_instances,
                    distance_function_weights =  [1] * plant.num_positions(),
                    #configuration_distance_function = _configuration_distance,
                    edge_step_size = 0.125)

# def sample_handle_joint(N, M, regions, frac_ts_samples = ts_fraction, collision_free=True):
#     N_ts = int(frac_ts_samples*N)
#     N_c = N-N_ts
#     print(f"[JOINT TS C SAMPLING] Sampling {N_ts} points in taskspace points via IK")
#     pts_q_ts, pts_t, is_full_ts = sample_handle_ts(N_ts, regions, collision_free=collision_free)
#     print(f"[JOINT TS C SAMPLING] Sampling {N_c} points uniformly in Cspace")
#     pts_q_c, is_full = sample_cfree(N_c, M, regions)
#     return np.concatenate((pts_q_ts, pts_q_c), axis=0), is_full

vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 

clogger = CliqueApproachLogger(f"7dof_iiwa_bins_taskspace",f"{ap_names[approach]}", estimate_coverage=estimate_coverage, cfg_dict=cfg)
iris_handle = partial(SNOPT_IRIS_ellipsoid_parallel,
                        region_obstacles = [],
                        logger = clogger, 
                        plant_builder = plant_builder,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1- eps)

vcd = VisCliqueInflation(N, 
                    eps,
                    max_iterations=max_iterations_clique,
                    sample_cfree = sample_cfree,
                    col_handle= col_func_handle_,
                    build_vgraph=vgraph_handle,
                    iris_w_obstacles=iris_handle,
                    verbose = True,
                    logger=clogger,
                    approach=approach,
                    extend_cliques=extend_cliques,
                    min_clique_size = min_clique_size
                    )
regs = vcd.run()