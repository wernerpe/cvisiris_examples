#from pydrake.all import RationalForwardKinematics
from pydrake.geometry.optimization import IrisOptions#, HPolyhedron, Hyperellipsoid
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
# from pydrake.all import (PiecewisePolynomial, 
#                          InverseKinematics, 
#                          Sphere, 
#                          Rgba, 
#                          RigidTransform, 
#                          RotationMatrix, 
#                          IrisInConfigurationSpace)
#import time
#import pydrake
from ur3e_demo import UrDiagram, SetDiffuse
import visualization_utils as viz_utils
from functools import partial
import numpy as np
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from pydrake.all import SceneGraphCollisionChecker
from visibility_logging import CliqueApproachLogger
from visibility_clique_decomposition import VisCliqueDecomp
from region_generation import SNOPT_IRIS_ellipsoid, SNOPT_IRIS_ellipsoid_parallel

add_shelf = True
seed = 1
for seed in range(3,12):
    N = 1000
    eps = 0.25
    max_iterations_clique = 15
    min_clique_size = 14
    approach = 1
    ap_names = ['redu', 'greedy', 'nx']
    extend_cliques = False

    require_sample_point_is_contained = True
    iteration_limit = 1
    configuration_space_margin = 1.e-3
    termination_threshold = -1
    num_collision_infeasible_samples = 19
    relative_termination_threshold = 0.02

    pts_coverage_estimator = 5000
    cfg = {'add_shelf': add_shelf,
        'seed': seed,
        'N': N,
        'eps': eps,
        'max_iterations_clique': max_iterations_clique,
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

    def plant_builder(use_meshcat = False):
        ur = UrDiagram(num_ur = 1, weld_wrist = True, add_shelf = add_shelf,
                        add_gripper = True, use_meshcat=use_meshcat)

        if use_meshcat: meshcat = ur.meshcat
        plant = ur.plant
        diagram_context = ur.diagram.CreateDefaultContext()
        ur.diagram.ForcedPublish(diagram_context)
        diagram = ur.diagram
        plant_context = ur.plant.GetMyMutableContextFromRoot(
                diagram_context)
        # scene_graph_context = ur.scene_graph.GetMyMutableContextFromRoot(
        #     diagram_context)
        scene_graph = ur.scene_graph
        return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if use_meshcat else None

    plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(True)


    robot_instances = [plant.GetModelInstanceByName("ur0"), plant.GetModelInstanceByName("schunk0")]

    checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                    robot_model_instances = robot_instances,
                    distance_function_weights =  [1] * plant.num_positions(),
                    #configuration_distance_function = _configuration_distance,
                    edge_step_size = 0.125)

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
    clogger = CliqueApproachLogger(f"5dof_ur_{'shelf' if add_shelf else 'noshelf'}",f"{ap_names[approach]}", estimate_coverage=estimate_coverage, cfg_dict=cfg)
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
    vcd = VisCliqueDecomp(N, 
                    eps,
                    max_iterations=max_iterations_clique,
                    sample_cfree = sample_cfree,
                    col_handle= col_func_handle_,
                    build_vgraph=vgraph_handle,
                    iris_w_obstacles=iris_handle,
                    verbose = True,
                    logger=clogger,
                    approach=approach,
                    extend_cliques=extend_cliques
                    )
    regs = vcd.run()


print('ha')