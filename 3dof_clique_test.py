#pydrake imports
from pydrake.all import RationalForwardKinematics
import mcubes
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.all import (PiecewisePolynomial, 
                        InverseKinematics, 
                        Sphere, 
                        Rgba, 
                        RigidTransform, 
                        RotationMatrix, 
                        IrisInConfigurationSpace, 
                        RollPitchYaw,
                        StartMeshcat,
                        MeshcatVisualizerParams,
                        MeshcatVisualizer,
                        Role,
                        TriangleSurfaceMesh,
                        SurfaceTriangle)
import time
import pydrake
from ur3e_demo import UrDiagram, SetDiffuse
import visualization_utils as viz_utils
from functools import partial
import numpy as np
from pydrake.planning import RobotDiagramBuilder

from pydrake.all import VisibilityGraph
from pydrake.all import SceneGraphCollisionChecker

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
X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array([5, 4, 2]) ) # some drake.RigidTransform()
meshcat.SetTransform("/Cameras/default", X_WC) 
# meshcat.SetProperty("/Background", "top_color", [0.8, 0.8, 0.6])
# meshcat.SetProperty("/Background", "bottom_color",
#                                 [0.9, 0.9, 0.9])
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.ForcedPublish(diagram_context)
print(meshcat.web_url())
plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

ik = InverseKinematics(plant, plant_context)
collision_constraint = ik.AddMinimumDistanceConstraint(0.001, 0.001)

def eval_cons(q, c, tol):
    return 1-1*float(c.evaluator().CheckSatisfied(q, tol))


scaler = 1 #np.array([0.8, 1., 0.8, 1, 0.8, 1, 0.8]) #do you even geometry bro?
q_min = np.array([-1.7, -2., -1.7])
q_max = np.array([ 1.7,  2.,  1.7])
q_diff = q_max-q_min
col_func_handle = partial(eval_cons, c=collision_constraint, tol=0.01)

def sample_cfree_QPoint(MAXIT=1000):
        it = 0
        while it<MAXIT:
                rand = np.random.rand(3)
                q_s = q_min + rand*q_diff
                col = col_func_handle(q_s)
                if not col:
                        return q_s #Ratfk.ComputeQValue(q_s, q_star)
                it+=1

robot_instances =[plant.GetModelInstanceByName("iiwaonedof"), plant.GetModelInstanceByName("iiwatwodof")]
checker = SceneGraphCollisionChecker(model = diagram.Clone(), 
                robot_model_instances = robot_instances,
                distance_function_weights =  [1] * plant.num_positions(),
                #configuration_distance_function = _configuration_distance,
                edge_step_size = 0.125)

points = np.array([sample_cfree_QPoint()])


_offset_meshcat_2 = np.array([-1,-5, 1.5])


collision_constraint_plot = ik.AddMinimumDistanceConstraint(0.001, 0.001)
def check_collision_by_ik(q0,q1,q2, min_dist=1e-5):
        q = np.array([q0,q1,q2])
        return 1.*col_func_handle(q) 


def plot_collision_constraint(N = 50):
        import pickle
        import os
        if f"col_cons{N}.pkl" in os.listdir('tmp'):
               with open(f"tmp/col_cons{N}.pkl", 'rb') as f:
                d = pickle.load(f)
               vertices = d['vertices']
               triangles = d['triangles']
        else:  
                vertices, triangles = mcubes.marching_cubes_func(
                tuple(
                        q_min), tuple(
                        q_max), N, N, N, check_collision_by_ik, 0.5)
                with open(f"tmp/col_cons{N}.pkl", 'wb') as f:
                       d = {'vertices': vertices, 'triangles': triangles}
                       pickle.dump(d, f)

        tri_drake = [SurfaceTriangle(*t) for t in triangles]

        vertices += _offset_meshcat_2.reshape(-1,3)
        meshcat.SetObject("/collision_constraint",
                                      TriangleSurfaceMesh(tri_drake, vertices),
                                      Rgba(1, 0, 0, 1), wireframe=True)

plot_collision_constraint()

ad_mat = VisibilityGraph(checker, points.T, parallelize=True)


print('')
        