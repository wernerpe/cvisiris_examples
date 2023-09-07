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

seed = 1
for seed in range(10):
    np.random.seed(seed) 

    meshcat = StartMeshcat()
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()
    # plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    # parser = Parser(plant)
    rel_path_cvisiris = "../cvisiris_examples/"
    oneDOF_iiwa_asset = rel_path_cvisiris + "assets/oneDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf")
    twoDOF_iiwa_asset = rel_path_cvisiris + "assets/twoDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/twoDOF_iiwa7_with_box_collision.sdf")

    box_asset = rel_path_cvisiris + "assets/box_small.urdf" #FindResourceOrThrow("drake/C_Iris_Examples/assets/box_small.urdf")

    models = []
    models.append(parser.AddModelFromFile(box_asset, "box"))
    models.append(parser.AddModelFromFile(twoDOF_iiwa_asset, "iiwatwodof"))
    models.append(parser.AddModelFromFile(oneDOF_iiwa_asset, "iiwaonedof"))

    locs = [[0.,0.,0.],
            [0.,.55,0.],
            [0.,-.55,0.]]
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[0]),
        RigidTransform(locs[0]))
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("iiwa_twoDOF_link_0", models[1]), 
                    RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[1]))
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("iiwa_oneDOF_link_0", models[2]), 
                    RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[2]))

    plant.Finalize()
    inspector = scene_graph.model_inspector()

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.role = Role.kIllustration
    visualizer = MeshcatVisualizer.AddToBuilder(
            builder.builder(), scene_graph, meshcat, meshcat_params)
    # X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array([5, 4, 2]) ) # some drake.RigidTransform()
    # meshcat.SetTransform("/Cameras/default", X_WC) 
    # meshcat.SetProperty("/Background", "top_color", [0.8, 0.8, 0.6])
    # meshcat.SetProperty("/Background", "bottom_color",
    #                                 [0.9, 0.9, 0.9])
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(diagram_context)
    print(meshcat.web_url())
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
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
    estimate_coverage = get_coverage_estimator(sample_cfree, pts = 5000)

    _offset_meshcat_2 = np.array([-1,-5, 1.5])
    col_func_handle2 = get_col_func(plant, plant_context, min_dist=0.001)
    def check_collision_by_ik(q0,q1,q2, min_dist=1e-5):
        q = np.array([q0,q1,q2])
        return 1.*col_func_handle2(q) 

    snopt_iris_options = IrisOptions()
    snopt_iris_options.require_sample_point_is_contained = True
    snopt_iris_options.iteration_limit = 10
    snopt_iris_options.configuration_space_margin = 1.0e-4
    #snopt_iris_options.max_faces_per_collision_pair = 60
    snopt_iris_options.termination_threshold = -1
    #snopt_iris_options.q_star = np.zeros(3)
    snopt_iris_options.num_collision_infeasible_samples = 19
    snopt_iris_options.relative_termination_threshold = 0.02

    def is_los(q1, q2, regions, num_checks, checker):

        tvec = np.linspace(0,1, num_checks)
        configs = tvec.reshape(-1,1)@q1.reshape(1,-1) + (1-tvec.reshape(-1,1))@q2.reshape(1,-1)
        for c in configs:
            if point_in_regions(c, regions):
                return False
        col_free = checker.CheckConfigsCollisionFree(configs, parallelize = True)
        if np.any(np.array(col_free)==0):
            return False
        return True

    visibility_checker = partial(is_los, num_checks = 40, checker = checker)

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
    eps = 0.1
    alpha = 0.05 #not used in current forumlation
    #approach = 0
    #ap_names = ['redu', 'greedy', 'nx']

    logger = Logger(f"3DOf_pinball_naive", seed, N, alpha, eps, estimate_coverage)
    iris_handle = partial(SNOPT_IRIS_obstacles, 
                        logger = logger, 
                        plant = plant, 
                        context = diagram_context,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1- eps)

    #vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
    VS = VisSeeder(N = N,
                    alpha = alpha,
                    eps = eps,
                    max_iterations = 300,
                    sample_cfree = sample_cfree,
                    build_vgraph = vgraph_builder,
                    iris_w_obstacles = iris_handle,
                    verbose = True,
                    logger = logger
                )
    out = VS.run()

from visualization_utils import generate_maximally_different_colors, plot_regions, plot_ellipses

plot_regions(meshcat, VS.regions, offset=_offset_meshcat_2)
print('')