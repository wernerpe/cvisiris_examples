import os
import pickle
import numpy as np
from pydrake.all import (#PiecewisePolynomial, 
                        #InverseKinematics, 
                        Sphere, 
                        Cylinder,
                        Rgba, 
                        RigidTransform, 
                        RotationMatrix, 
                        #IrisInConfigurationSpace, 
                        RollPitchYaw,
                        StartMeshcat,
                        MeshcatVisualizerParams,
                        MeshcatVisualizer,
                        Role,
                        TriangleSurfaceMesh,
                        SurfaceTriangle,
                        IrisOptions
                        )
from functools import partial
import numpy as np
from pydrake.planning import RobotDiagramBuilder
from pydrake.all import SceneGraphCollisionChecker
from region_generation import SNOPT_IRIS_obstacles
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              point_in_regions
                              )
from visibility_seeding import VisSeeder
from visibility_logging import Logger
from scipy.sparse import lil_matrix
from tqdm import tqdm
from pydrake.all import (SceneGraphCollisionChecker, 
                         StartMeshcat, 
                         RobotDiagramBuilder,
                         ProcessModelDirectives,
                         LoadModelDirectives,
                         MeshcatVisualizer)


# seed = 1
for seed in [29]:
    np.random.seed(seed)
    def plant_builder(usemeshcat = False):
        if usemeshcat:
            meshcat = StartMeshcat()
        builder = RobotDiagramBuilder()
        plant = builder.plant()
        scene_graph = builder.scene_graph()
        parser = builder.parser()
        #parser.package_map().Add("cvisirisexamples", missing directory)
        if usemeshcat:
            visualizer = MeshcatVisualizer.AddToBuilder(builder.builder(), scene_graph, meshcat)
        directives_file = "7_dof_directives_newshelf.yaml"#FindResourceOrThrow() 
        directives = LoadModelDirectives(directives_file)
        models = ProcessModelDirectives(directives, plant, parser)
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        diagram.ForcedPublish(diagram_context)
        return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None

    plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)

    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
        diagram_context)
    robot_instances = [plant.GetModelInstanceByName("iiwa"), plant.GetModelInstanceByName("wsg")]

    checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                    robot_model_instances = robot_instances,
                    distance_function_weights =  [1] * plant.num_positions(),
                    #configuration_distance_function = _configuration_distance,
                    edge_step_size = 0.125)

    diagram.ForcedPublish(diagram_context)

    scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) 
    q_min = plant.GetPositionLowerLimits()*scaler
    q_max =  plant.GetPositionUpperLimits()*scaler

    col_func_handle_ = get_col_func(plant, plant_context)
    sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)
    estimate_coverage = get_coverage_estimator(sample_cfree, pts = 5000)

    snopt_iris_options = IrisOptions()
    snopt_iris_options.require_sample_point_is_contained = True
    snopt_iris_options.iteration_limit = 6
    snopt_iris_options.configuration_space_margin = 2e-3
    #snopt_iris_options.max_faces_per_collision_pair = 60
    snopt_iris_options.termination_threshold = -1
    #snopt_iris_options.q_star = np.zeros(3)
    snopt_iris_options.num_collision_infeasible_samples = 19
    snopt_iris_options.relative_termination_threshold = 0.02


    def is_los(q1, q2, regions, step_size, checker):
        d = np.linalg.norm(q2-q1)
        num_checks = np.ceil(d/step_size) +1
        tvec = np.linspace(0,1, num_checks)
        configs = tvec.reshape(-1,1)@q1.reshape(1,-1) + (1-tvec.reshape(-1,1))@q2.reshape(1,-1)
        for c in configs:
            if point_in_regions(c, regions):
                return False
        col_free = checker.CheckEdgeCollisionFreeParallel(q1.reshape(-1,1), q2.reshape(-1,1))
        #col_free = checker.CheckConfigsCollisionFree(configs, parallelize = True)
        if np.any(np.array(col_free)==0):
            return False
        return True

    visibility_checker = partial(is_los, step_size = 0.125, checker = checker)

    def vgraph_builder(points, regions):
        n = len(points)
        adj_mat = lil_matrix((n,n))
        for i in tqdm(range(n)):
            point = points[i, :]
            for j in range(len(points[:i])):
                other = points[j]
                result = visibility_checker(point, other, regions)
                #print(result)
                if result:
                    adj_mat[i,j] = adj_mat[j,i] = 1
        return adj_mat#.toarray()

    N = 1
    eps = 0.25
    alpha = 0.05 #not used in current forumlation

    logger = Logger(f"7dof_iiwa_naive_", seed, N, alpha, eps, estimate_coverage)
    iris_handle = partial(SNOPT_IRIS_obstacles, 
                        logger = logger, 
                        plant = plant, 
                        context = diagram_context,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1- eps)

    VS = VisSeeder(N = N,
                    alpha = alpha,
                    eps = eps,
                    max_iterations = 2000,
                    sample_cfree = sample_cfree,
                    build_vgraph = vgraph_builder,
                    iris_w_obstacles = iris_handle,
                    verbose = True,
                    logger = logger
                )
    out = VS.run()