import numpy as np
from pydrake.all import (
                        IrisOptions
                        )
from functools import partial
import numpy as np
from pydrake.all import SceneGraphCollisionChecker
from region_generation import SNOPT_IRIS_obstacles
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              point_in_regions
                              )
from hidden_point_seeding import HiddenPointSeeder
from visibility_logging import Logger
from scipy.sparse import lil_matrix
from tqdm import tqdm
from environments import get_environment_builder
    
seed = 1
for seed in range(10):
    np.random.seed(seed) 
    env_builder = get_environment_builder('3DOFFLIPPER')
    plant, scene_graph, diagram, diagram_context, plant_context, meshcat = env_builder(usemeshcat=True)
    inspector = scene_graph.model_inspector()
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
    
    N = 1 #this makes the apporach IOS
    eps = 0.1
    alpha = 0.05 #not used in current forumlation

    logger = Logger(f"3DOf_pinball_naive_it", seed, N, alpha, eps, estimate_coverage)
    iris_handle = partial(SNOPT_IRIS_obstacles, 
                        logger = logger, 
                        plant = plant, 
                        context = diagram_context,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1- eps)

    #vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
    VS = HiddenPointSeeder(N = N,
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

print('done')