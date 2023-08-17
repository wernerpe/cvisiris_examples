#pydrake imports
#from pydrake.all import RationalForwardKinematics
import mcubes
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid
#from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
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
                        #IrisRegionsFromCliqueCover
                        )
from functools import partial
import numpy as np
from pydrake.planning import RobotDiagramBuilder
from pydrake.all import SceneGraphCollisionChecker
from region_generation import SNOPT_IRIS_ellipsoid
from visibility_clique_decomposition import VisCliqueDecomp
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from visibility_logging import CliqueApproachLogger
import os
import pickle

seed = 1
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
snopt_iris_options.iteration_limit = 1
snopt_iris_options.configuration_space_margin = 1.0e-4
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = -1
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.num_collision_infeasible_samples = 19
snopt_iris_options.relative_termination_threshold = 0.02

N = 500
eps = 0.1
approach = 0
ap_names = ['redu', 'greedy', 'nx']

iris_handle = partial(SNOPT_IRIS_ellipsoid, 
                      region_obstacles = [],
                      logger = None, 
                      plant = plant, 
                      context = diagram_context,
                      snoptiris_options = snopt_iris_options,
                      estimate_coverage = estimate_coverage,
                      coverage_threshold = 1- eps)

vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 
clogger = CliqueApproachLogger(f"3dof_flipper_",f"{ap_names[approach]}", seed = seed, N = N, eps= eps, estimate_coverage=estimate_coverage)

vcd = VisCliqueDecomp(N, 
                eps,
                max_iterations=100,
                sample_cfree = sample_cfree,
                col_handle= col_func_handle_,
                build_vgraph=vgraph_handle,
                iris_w_obstacles=iris_handle,
                verbose = True,
                logger=clogger,
                approach=0
                )
regs = vcd.run()


def plot_collision_constraint(N = 50, q_min = q_min, q_max= q_max):
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
        
def plot_points(points, name, size = 0.05, color = Rgba(0.06, 0.0, 0, 1)):
    for i, pt in enumerate(points):
        n_i = name+f"/pt{i}"
        meshcat.SetObject(n_i,
                          Sphere(size),
                          color)
        meshcat.SetTransform(n_i, 
                             RigidTransform(
                             RotationMatrix(), 
                             np.array(pt)+_offset_meshcat_2.reshape(-1)))

def compute_rotation_matrix(a, b):
    # Normalize the points
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # Calculate the rotation axis
    rotation_axis = np.cross(a, b)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Calculate the rotation angle
    dot_product = np.dot(a, b)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Construct the rotation matrix using Rodrigues' formula
    skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * skew_matrix + (1 - np.cos(rotation_angle)) * np.dot(skew_matrix, skew_matrix)
    
    return rotation_matrix

def plot_edge(pt1, pt2, name, color, size = 0.01):
    meshcat.SetObject(name,
                        Cylinder(size, np.linalg.norm(pt1-pt2)),
                        color)
    
    dir = pt2-pt1
    rot = compute_rotation_matrix(np.array([0,0,1]), dir )
    offs = rot@np.array([0,0,np.linalg.norm(pt1-pt2)/2])
    meshcat.SetTransform(name, 
                        RigidTransform(
                        RotationMatrix(rot), 
                        np.array(pt1)+offs))

def plot_edges(edges, name, color = Rgba(0,0,0,1), size = 0.01):
    for i, e in enumerate(edges):
         plot_edge(e[0], e[1], name + f"/e_{i}", color= color, size=size)

def get_edges_clique(cl, points):
    e = []
    for i,c1 in enumerate(cl[:-1]):
        for c2 in cl[i+1:]:
            e.append([points[c1, :], points[c2,:]])
    return e
#plot_edges(edges, 'evgraph', Rgba(0,1,1,1))
#plot_points(points, size=0.1, name = 'a')
plot_collision_constraint(100)

# clique_edges = []
# for cl_lst, pts_cl in zip(vcd.cliques, vcd.vgraph_points):
#     for c in cl_lst:
#         clique_edges.append(get_edges_clique(c, pts_cl))
from visualization_utils import generate_maximally_different_colors, plot_regions, plot_ellipses

# colors = [Rgba(c[0], c[1], c[2], 1.) for c in generate_maximally_different_colors(len(clique_edges))]
# for i,c in enumerate(clique_edges):
#     plot_edges(c+_offset_meshcat_2, f"cl_{i}", color=colors[i])
plot_regions(meshcat, vcd.regions, offset=_offset_meshcat_2)
ellipses = [] 
for g in vcd.metrics_iteration:
    for e in g:
        ellipses.append(e)
colors = [Rgba(c[0], c[1], c[2], 0.8) for c in generate_maximally_different_colors(len(ellipses)+5)[5:]]   


plot_ellipses(meshcat, ellipses,'LJe', colors, offset=_offset_meshcat_2)

# idx_c = 0
# ell = ellipses[idx_c]
# points = vcd.vgraph_points[0][vcd.cliques[0][idx_c]]

# pts_cl =vcd.vgraph_points[0][vcd.cliques[0][4], :]
# e_crit = Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(pts_cl.T)
# plot_points(pts_cl, size = 0.1, name ='a')
# plot_ellipses(meshcat, [e_crit],'a', [colors[0]], offset=_offset_meshcat_2)
# q_min = np.min(pts_cl, axis=0)
# q_max = np.max(pts_cl, axis=0)
# diff= q_max-q_min
# pts_e = []
# for _ in range(2000):
#     r = q_min+diff*np.random.rand(3)
#     if e_crit.PointInSet(r):
#         pts_e.append(r)
# plot_points(np.array(pts_e), size = 0.05, name = 'b')
print('')
