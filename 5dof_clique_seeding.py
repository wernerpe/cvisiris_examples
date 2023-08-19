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
from region_generation import SNOPT_IRIS_ellipsoid

add_shelf = True
seed = 0
np.random.seed(seed)

ur = UrDiagram(num_ur = 1, weld_wrist = True, add_shelf = add_shelf,
                 add_gripper = True)
meshcat = ur.meshcat
plant = ur.plant
diagram_context = ur.diagram.CreateDefaultContext()
ur.diagram.ForcedPublish(diagram_context)
diagram = ur.diagram

plant_context = ur.plant.GetMyMutableContextFromRoot(
        diagram_context)
scene_graph_context = ur.scene_graph.GetMyMutableContextFromRoot(
    diagram_context)
inspector = ur.scene_graph.model_inspector()    
robot_instances = [plant.GetModelInstanceByName("ur0"), plant.GetModelInstanceByName("schunk0")]

checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                robot_model_instances = robot_instances,
                distance_function_weights =  [1] * plant.num_positions(),
                #configuration_distance_function = _configuration_distance,
                edge_step_size = 0.125)

scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) 
q_min = ur.plant.GetPositionLowerLimits()*scaler
q_max =  ur.plant.GetPositionUpperLimits()*scaler

col_func_handle_ = get_col_func(plant, plant_context)
sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)
estimate_coverage = get_coverage_estimator(sample_cfree, pts = 5000)

snopt_iris_options = IrisOptions()
snopt_iris_options.require_sample_point_is_contained = True
snopt_iris_options.iteration_limit = 1
snopt_iris_options.configuration_space_margin = 0.5e-3
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = -1
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.num_collision_infeasible_samples = 19
snopt_iris_options.relative_termination_threshold = 0.02


N = 1000
eps = 0.05
approach = 0
ap_names = ['redu', 'greedy', 'nx']


vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
clogger = CliqueApproachLogger(f"5dof_ur_init_sp_{'shelf' if add_shelf else 'noshelf'}",f"{ap_names[approach]}", seed = seed, N = N, eps= eps, estimate_coverage=estimate_coverage)
iris_handle = partial(SNOPT_IRIS_ellipsoid, 
                      region_obstacles = [],
                      logger = clogger, 
                      plant = plant, 
                      context = diagram_context,
                      snoptiris_options = snopt_iris_options,
                      estimate_coverage = estimate_coverage,
                      coverage_threshold = 1- eps)

vcd = VisCliqueDecomp(N, 
                eps,
                max_iterations=5,
                sample_cfree = sample_cfree,
                col_handle= col_func_handle_,
                build_vgraph=vgraph_handle,
                iris_w_obstacles=iris_handle,
                verbose = True,
                logger=clogger,
                approach=approach
                )
regs = vcd.run()


print('ha')