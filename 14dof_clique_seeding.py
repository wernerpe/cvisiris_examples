from pydrake.all import (
                        StartMeshcat,
                        LoadModelDirectives,
                        ProcessModelDirectives,
                        MeshcatVisualizerParams,
                        MeshcatVisualizer,
                        SceneGraphCollisionChecker,
                        Role,
                        RobotDiagramBuilder,
                        IrisOptions,
                        )
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
import numpy as np
import os
from functools import partial

from visibility_logging import CliqueApproachLogger
from visibility_clique_decomposition import VisCliqueInflation
from region_generation import SNOPT_IRIS_ellipsoid, SNOPT_IRIS_ellipsoid_parallel

seed = 1
N = 4000
eps = 0.6
max_iterations_clique = 10
min_clique_size = 40
approach = 0
ap_names = ['redu', 'greedy', 'nx']
extend_cliques = False

require_sample_point_is_contained = True
iteration_limit = 1
configuration_space_margin = 2.e-3
termination_threshold = -1
num_collision_infeasible_samples = 15
relative_termination_threshold = 0.02

pts_coverage_estimator = 5000
cfg = {'seed': seed,
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

np.random.seed(seed)
from environments import get_environment_builder
plant_builder = get_environment_builder('14DOFIIWAS')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)
robot_instances = [plant.GetModelInstanceByName("iiwa_left"), plant.GetModelInstanceByName("iiwa_right"), plant.GetModelInstanceByName("wsg_left"), plant.GetModelInstanceByName("wsg_right")]

checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                robot_model_instances = robot_instances,
                distance_function_weights =  [1] * plant.num_positions(),
                #configuration_distance_function = _configuration_distance,
                edge_step_size = 0.2)

scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) 
q_min = plant.GetPositionLowerLimits()*scaler
q_max =  plant.GetPositionUpperLimits()*scaler

col_func_handle_ = get_col_func(plant, plant_context)
sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)
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



vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
clogger = CliqueApproachLogger(f"14dof_iiwa_",f"{ap_names[approach]}", estimate_coverage=estimate_coverage, cfg_dict=cfg)
# iris_handle = partial(SNOPT_IRIS_ellipsoid, 
#                       region_obstacles = [],
#                       logger = clogger, 
#                       plant = plant, 
#                       context = diagram_context,
#                       snoptiris_options = snopt_iris_options,
#                       estimate_coverage = estimate_coverage,
#                       coverage_threshold = 1- eps)

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
print('ready')