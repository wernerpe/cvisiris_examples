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
from visibility_clique_decomposition import VisCliqueInflation
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from visibility_logging import CliqueApproachLogger
import os
import pickle

N = 100
eps = 0.1
approach = 1
ap_names = ['redu', 'greedy', 'nx']
seed = 1

max_iterations_clique = 10
extend_cliques = False

require_sample_point_is_contained = True
iteration_limit = 1
configuration_space_margin = 1.e-4
termination_threshold = -1
num_collision_infeasible_samples = 19
relative_termination_threshold = 0.02
pts_coverage_estimator = 5000
seed = 1
cfg = {
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
    if np.any(q>q_max):
        return 1
    if np.any(q<q_min):
        return 1
    return 1.*col_func_handle2(q) 

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


#with open('logs/experiment_3dof_flipper__1_500_0.100greedy20230905161943/data/it_0.pkl', 'rb') as f:
with open('logs_icra_paper/3DOf_pinball_naive_1it_20230917190637_9_1_0.050_0.100/data/it_40.pkl', 'rb') as f:
    #3DOf_pinball_naive_1it_20230917190637_9_1_0.050_0.100
    #3DOf_pinball_naive_20230830155947_0_1_0.050_0.100
    data = pickle.load(f)

regionsIOS = []
for rga, rgb in zip(data['ra'], data['rb']):
    for ra, rb in zip(rga, rgb):
        regionsIOS.append(HPolyhedron(ra, rb))

with open('logs_icra_paper/experiment_3dof_flipper__1_500_0.100greedy20230905161943/data/it_0.pkl', 'rb') as f:
    data = pickle.load(f)

regionsCBS = []
for rga, rgb in zip(data['ra'], data['rb']):
    for ra, rb in zip(rga, rgb):
        regionsCBS.append(HPolyhedron(ra, rb))

def plot_collision_constraint(N = 50, q_min = q_min, q_max= q_max):
    if f"col_cons{N}.pkl" in os.listdir('tmp'):
        with open(f"tmp/col_cons{N}.pkl", 'rb') as f:
            d = pickle.load(f)
            vertices = d['vertices']
            triangles = d['triangles']
    else:  
        vertices, triangles = mcubes.marching_cubes_func(
        tuple(
                q_min-0.2), tuple(
                q_max+0.2), N, N, N, check_collision_by_ik, 0.5)
        with open(f"tmp/col_cons{N}.pkl", 'wb') as f:
                d = {'vertices': vertices, 'triangles': triangles}
                pickle.dump(d, f)

    tri_drake = [SurfaceTriangle(*t) for t in triangles]

    vertices += _offset_meshcat_2.reshape(-1,3)
    meshcat.SetObject("/collision_constraint/c1",
                                    TriangleSurfaceMesh(tri_drake, vertices),
                                    Rgba(0, 0.6, 0, 0.2))
    meshcat.SetObject("/collision_constraint/c2",
                                    TriangleSurfaceMesh(tri_drake, vertices),
                                    Rgba(0, 0.6, 0, 1), wireframe = True)
    
plot_collision_constraint(110)
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

def plot_edges_vgraph(ad, pts):
    edges = []
    for i in range(ad.shape[0]):
        for j in range(i+1, ad.shape[0]):
        
            if ad[i,j]==1 and np.random.rand()>0.05:
               plot_edge(pts[i,:]+_offset_meshcat_2, pts[j, :]+_offset_meshcat_2, "vgraph" + f"/e_{i}{j}", color= Rgba(0,0,0,0.9), size=0.005) #edges.append([pts[i,:], pts[j,:]])
    return edges

# clique_edges = []
# for cl_lst, pts_cl in zip(vcd.cliques, vcd.vgraph_points):
#     for c in cl_lst:
#         clique_edges.append(get_edges_clique(c, pts_cl))


# plot_points(vcd.vgraph_points[0], size=0.05, name = 'a')
# plot_edges_vgraph(vcd.vgraph_admat[0], vcd.vgraph_points[0])

from visualization_utils import generate_maximally_different_colors, plot_regions, plot_ellipses
#plot_edges(edges, 'evgraph', Rgba(0,1,1,1))
#plot_collision_constraint(110)

# colors = [Rgba(c[0], c[1], c[2], 1.) for c in generate_maximally_different_colors(len(clique_edges))]
# colors2 = [[c.r(),c.g(), c.b()] for c in colors ]
# for i,c in enumerate(clique_edges):
#     plot_edges(c+_offset_meshcat_2, f"cl_{i}", color=colors[i])
# #for i,c in enumerate(clique_edges):
#     plot_edges(c+_offset_meshcat_2, f"vg_{i}", color=colors[i])
plot_regions(meshcat, regionsIOS,region_suffix='IOS', offset=_offset_meshcat_2, opacity=.9)
plot_regions(meshcat, regionsCBS,region_suffix='CBS', offset=_offset_meshcat_2, opacity=.9)

#plot_regions(meshcat, vcd.regions,region_suffix='CBS', offset=_offset_meshcat_2, opacity=.9, colors = colors)
def conv_dummy(q):
    return q

import dijkstraspp
dspp = dijkstraspp.DijkstraSPPsolver(regionsCBS, conv_dummy)
import pydrake
from pydrake.all import Rgba, Sphere, RotationMatrix, Box
def densify_waypoints(waypoints_q):
    densify = 100
    dists = []
    dense_waypoints = []
    for idx in range(len(waypoints_q[:-1])):
        a = waypoints_q[idx]
        b = waypoints_q[idx+1]
        t = np.linspace(1,0, 10)
        locs_endeff = []
        locs_endeff2 = []
        dists_endeff = []
        for tval in t:
            qa = a*tval + b*(1-tval)
            #qa = Ratfk.ComputeQValue(ta, np.zeros(7))
            #showres(qa)
            #time.sleep(0.1)            
            plant.SetPositions(plant_context, qa)
            #visualizer.set_joint_angles(qa)
            tf_tot= plant.EvalBodyPoseInWorld(plant_context, plant.get_body(pydrake.multibody.tree.BodyIndex(12)))
            tf = tf_tot.translation() + tf_tot.GetAsMatrix4()[:3,:3][:,1] *0.15
            tf_tot= plant.EvalBodyPoseInWorld(plant_context, plant.get_body(pydrake.multibody.tree.BodyIndex(20)))
            tf2 = tf_tot.translation() + tf_tot.GetAsMatrix4()[:3,:3][:,1] *0.15
            locs_endeff.append(tf)
            locs_endeff2.append(tf2)
        for i in range(len(locs_endeff)-1):
            dists_endeff.append(0.5*np.linalg.norm(locs_endeff[i]- locs_endeff[i+1]) + 0.5*np.linalg.norm(locs_endeff2[i]- locs_endeff2[i+1]))
        d = np.sum(dists_endeff)
        #print(d * densify)
        t = np.linspace(1,0,int(densify*d))
        for tval in t:
            dense_waypoints.append(waypoints_q[idx]*tval + waypoints_q[idx+1]*(1-tval))
    return dense_waypoints

def hide_traj(name, anim, frame):
    for i in range(200):
        anim.SetTransform(frame,name + str(i), RigidTransform(
                            RotationMatrix(), 
                            np.array([0,0,1000])))
        
def prepare_traj_plotting():
    color = Rgba(1,0,0,1.0)
    color2 = Rgba(0,0,1,1.0)
    color3 = Rgba(0,1,1,1.0)
    for idx in range(200):
        meshcat.SetObject(f"/points/traj/{idx}",
                                Sphere(0.01),
                                color)
        meshcat.SetObject(f"/points/traj2/{idx}",
                               Sphere(0.01),
                               color2)
        meshcat.SetObject(f"/points/traj3/{idx}",
                               Sphere(0.025),
                               color3)

def plot_endeff_traj(dense_waypoints, frame, anim, opt = True):
    
    start_idx = 0
    for i, qa in enumerate(dense_waypoints[::2]):
        
        #showres(qa)
        #time.sleep(0.1)            
        plant.SetPositions(plant_context,qa)
        tf_tot= plant.EvalBodyPoseInWorld(plant_context, plant.get_body(pydrake.multibody.tree.BodyIndex(12)))
        tf = tf_tot.translation() + tf_tot.GetAsMatrix4()[:3,:3][:,1] *0.15

        # meshcat.SetObject(f"/points/traj/{i+start_idx}",
        #                        Sphere(0.005),
        #                        color)

        anim.SetTransform(frame, f"/points/traj/{i+start_idx}",
                                   RigidTransform(RotationMatrix(),
                                                  tf))
        
        tf_tot= plant.EvalBodyPoseInWorld(plant_context, plant.get_body(pydrake.multibody.tree.BodyIndex(20)))
        tf = tf_tot.translation() + tf_tot.GetAsMatrix4()[:3,:3][:,1] *0.15

        # meshcat.SetObject(f"/points/traj2/{i+start_idx}",
        #                        Sphere(0.005),
        #                        color2)

        anim.SetTransform(frame,  f"/points/traj2/{i+start_idx}",
                                   RigidTransform(RotationMatrix(),
                                                  tf))
        
        # meshcat.SetObject(f"/points/traj3/{i+start_idx}",
        #                        Sphere(0.02),
        #                        color3)

        anim.SetTransform(frame, f"/points/traj3/{i+start_idx}",
                                   RigidTransform(RotationMatrix(),
                                                  qa+_offset_meshcat_2))

from visibility_utils import point_in_regions
def sample_point_in_regions(regions):
    while True:
        pt = sample_cfree(1,1000, [])[0].reshape(-1,)
        if point_in_regions(pt,regions):
            return pt

meshcat.SetObject('config',
                Sphere(0.05),
                Rgba(0,1,1,1))

def showres(q, frame, anim):
    plant.SetPositions(plant_context, q)
    # meshcat.SetObject('config',
    #                       Sphere(0.05),
    #                       Rgba(0,1,1,1))
    
    anim.SetTransform(frame, 'config', 
                            RigidTransform(
                            RotationMatrix(), 
                            np.array(q)+_offset_meshcat_2.reshape(-1)))
    diagram.ForcedPublish(diagram_context)

import time
from pydrake.all import Mesh
a = Mesh('display_signs/3dofsign.gltf')
meshcat.SetObject('/instructionsign', a)
meshcat.SetTransform('/instructionsign',RigidTransform(
                            RotationMatrix.MakeZRotation(-np.pi/2)@RotationMatrix.MakeXRotation(-np.pi/2), 
                            np.array([0, 15 , 0])))

start = sample_point_in_regions(regionsCBS)

meshcat.SetProperty(f"/iris/regionsIOS", "visible", False)
meshcat.SetProperty(f"/iris/regionsCBS", "visible", True)
meshcat.SetProperty(f"/iris", "visible", False)
#meshcat.SetProperty(f"/Background", "visible", False)
#meshcat.SetProperty(f"/collision_constraint", "visible", False)
meshcat.SetProperty(f"/collision_constraint/c1", "visible", False)
meshcat.SetProperty(f"/Grid", "visible", False)
frame_time = 1/32.0
cur_time = 0

prepare_traj_plotting()
meshcat.StartRecording()
animation = meshcat.get_mutable_recording()
hide_traj('/points/traj/',animation, 0)
hide_traj('/points/traj2/',animation, 0)
hide_traj('/points/traj3/',animation, 0)

frame = 0
for idx in range(9):
    #nxt = vs.sample_in_regions() #
    while True:
        nxt = sample_point_in_regions(regionsCBS)
        if nxt[0] != start[0]:
            break
    wp, dist = dspp.solve(start, nxt, refine_path=True)#dijkstra_spp(start, nxt, node_intersections, base_ad_mat, vs.regions, point_conversion, optimize= True)
    print(dist)
    if dist >0:
        dense_waypoints = densify_waypoints(wp)
        plot_endeff_traj(dense_waypoints,frame, animation)
        for qa in dense_waypoints[::2]:
            diagram_context.SetTime(cur_time)
            showres(qa, frame, animation)
            plot_endeff_traj(dense_waypoints,frame, animation)
            # if visualizer.col_func_handle(qa):
            #     #print(col)
            #     break
            time.sleep(frame_time)
            cur_time+=frame_time
            frame+= 1
        start = nxt
        hide_traj("/points/traj/", animation ,frame)
        hide_traj("/points/traj2/", animation ,frame)
        hide_traj("/points/traj3/", animation ,frame)
        time.sleep(10*frame_time)
        cur_time+=10*frame_time
        frame+=10
        hide_traj("/points/traj/", animation ,frame)
        hide_traj("/points/traj2/", animation ,frame)
        hide_traj("/points/traj3/", animation ,frame)
    else:
        nxt = sample_point_in_regions(regionsCBS)
meshcat.StopRecording()
animation.set_autoplay(True)
meshcat.PublishRecording()
with open("static_htmls/3dofsys.html", "w+") as f:
    f.write(meshcat.StaticHtml())
print('done')

