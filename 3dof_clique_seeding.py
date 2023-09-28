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
for seed in range(1):
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

   
    snopt_iris_options = IrisOptions()
    snopt_iris_options.require_sample_point_is_contained =require_sample_point_is_contained
    snopt_iris_options.iteration_limit =iteration_limit
    snopt_iris_options.configuration_space_margin =configuration_space_margin
    snopt_iris_options.termination_threshold =termination_threshold
    snopt_iris_options.num_collision_infeasible_samples =num_collision_infeasible_samples
    snopt_iris_options.relative_termination_threshold =relative_termination_threshold

    iris_handle = partial(SNOPT_IRIS_ellipsoid, 
                        region_obstacles = [],
                        logger = None, 
                        plant = plant, 
                        context = diagram_context,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1- eps)

    vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
    clogger = CliqueApproachLogger(f"3dof_flipper_",f"{ap_names[approach]}",  estimate_coverage=estimate_coverage, cfg_dict=cfg)

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
print('done')
