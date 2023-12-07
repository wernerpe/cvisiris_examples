
from pydrake.geometry.optimization import IrisOptions, HPolyhedron, Hyperellipsoid
from pydrake.solvers import MosekSolver, CommonSolverOption, SolverOptions
from pydrake.all import (PiecewisePolynomial, 
                         InverseKinematics, 
                         Sphere, 
                         Rgba, 
                         RigidTransform, 
                         RotationMatrix, 
                         Solve,
                         MathematicalProgram,
                         RollPitchYaw,
                         Cylinder)
import time
import pydrake
import numpy as np
from functools import partial
from environments import get_environment_builder

from visualization_utils import plot_points, plot_regions
from pydrake.all import VPolytope, Role
from task_space_seeding_utils import (get_cvx_hulls_of_bodies,
                                      get_AABB_cvxhull,
                                      task_space_sampler_mp,
                                      task_space_sampler)
from pydrake.all import SceneGraphCollisionChecker
from visibility_utils import vgraph
from clique_covers import compute_greedy_clique_partition
import os
import pickle

from visibility_utils import get_coverage_estimator, get_sample_cfree_handle, get_col_func
from clique_covers import get_iris_metrics
from region_generation import SNOPT_IRIS_ellipsoid_parallel

Npts= 2000
offset_size = 0.05
kmeans_trials = 5
seed = 1337
SHOWSAMPLES = True
min_clique_size = 15
silhouette_method_steps = 10


plant_builder = get_environment_builder('7DOFBINS')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)

scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
    diagram_context)

def show_pose(qvis, plant, plant_context, diagram, diagram_context, endeff_frame, show_body_frame = None):
    plant.SetPositions(plant_context, qvis)
    diagram.ForcedPublish(diagram_context)
    tf =plant.EvalBodyPoseInWorld(plant_context,  plant.GetBodyByName(endeff_frame))
    transl = tf.translation()+tf.rotation()@np.array([0,0.1,0])
    if show_body_frame is not None:
        show_body_frame(RigidTransform(tf.rotation(), transl))

def show_ik_target(pose, meshcat, name):
    h = 0.2
    if 'targ' in name:
        colors = [Rgba(1,0.5,0, 0.5), Rgba(0.5,1,0, 0.5), Rgba(0.0,0.5,1, 0.5)]
    else:
        colors = [Rgba(1,0,0, 1), Rgba(0.,1,0, 1), Rgba(0.0,0.0,1, 1)]

    rot = pose.rotation()@RotationMatrix.MakeYRotation(np.pi/2)
    pos= pose.translation() +pose.rotation()@np.array([h/2, 0,0])
    meshcat.SetObject(f"/drake/ik_target{name}/triad1",
                                   Cylinder(0.01,0.2),
                                   colors[0])
    meshcat.SetTransform(f"/drake/ik_target{name}/triad1",RigidTransform(rot, pos))
    rot = pose.rotation()@RotationMatrix.MakeXRotation(-np.pi/2)
    pos= pose.translation() +pose.rotation()@np.array([0,h/2,0])

    meshcat.SetObject(f"/drake/ik_target{name}/triad2",
                                   Cylinder(0.01,0.2),
                                   colors[1])
    meshcat.SetTransform(f"/drake/ik_target{name}/triad2",RigidTransform(rot, pos))
    pos= pose.translation().copy()
    rot = pose.rotation()
    pos = pos + rot@np.array([0,0,h/2])
    meshcat.SetObject(f"/drake/ik_target{name}/triad3",
                                   Cylinder(0.01,0.2),
                                   colors[2])
    meshcat.SetTransform(f"/drake/ik_target{name}/triad3",RigidTransform(rot, pos))

show_body_frame = partial(show_ik_target, 
                          meshcat=meshcat, 
                          name='endeff_acutal', 
                          )
showres = partial(show_pose, 
                  plant = plant, 
                  plant_context = plant_context, 
                  diagram = diagram, 
                  diagram_context = diagram_context,
                  endeff_frame = 'body',
                  show_body_frame=show_body_frame)

#Determine task space regions of interest
from visibility_utils import generate_distinct_colors
geom_names = ['bin_base', 
              #'bin_base', 
              #'shelves_body'
              ]
model_names = ['binL', 
               #'binR', 
               #'shelves'
               ]
default_pos = [np.array([ 1.53294,  0.4056 ,  0.23294, -0.5944 ,0.,0.9056 ,0.]),
               #np.array([-1.53294,  0.4056 ,  0.23294, -0.5944 ,0.,  0.9056 ,0. ]),
               #np.array([ 0., -0.08940423,  0., -1.7087849,  0., 1.32867852,  0.])
               ]
approach_dir = [2, 
                #2, 
                #0
                ] 
approach_sign = [1,1,-1]
ts_samplers = []
cols = generate_distinct_colors(2*len(model_names), rgb = True)[1:]
#cols = [list(c)+[1] for c in cols]
AABB_sampling_regions = []
for i, (g, m) in enumerate(zip(geom_names, model_names)):
    cvx_hulls_of_ROI_unsc, bodies = get_cvx_hulls_of_bodies([g], [m], plant, scene_graph, scene_graph_context, scaling = 1.0)
    verts = [VPolytope(c).vertices().T for c in cvx_hulls_of_ROI_unsc]
    cvx_hulls_of_ROI = cvx_hulls_of_ROI_unsc
    cvx_hulls_of_ROI = []
    for v in verts:
        offset = approach_sign[i]*(np.sign(v[:,approach_dir[i]] - np.mean(v[:,approach_dir[i]]))==approach_sign[i])*offset_size
        v[:,approach_dir[i]] += offset #scale*(v[:,approach_dir[i]] - np.mean(v[:,approach_dir[i]])) +  np.mean(v[:,approach_dir[i]])
        cvx_hulls_of_ROI.append(HPolyhedron(VPolytope(v.T)))
    ts_min, ts_max, cvxh_hpoly = get_AABB_cvxhull(cvx_hulls_of_ROI)
    AABB_sampling_regions.append([ts_min, ts_max])
    plot_regions(meshcat, cvx_hulls_of_ROI, region_suffix=m,opacity=0.2, colors=[cols[i]])
    q0  = default_pos[i] #np.zeros(7)
    plant.SetPositions(plant_context, q0)
    plant.ForcedPublish(plant_context)
    showres(q0)
    t0 = plant.EvalBodyPoseInWorld(plant_context,  plant.GetBodyByName("body")).translation()       
    sample_handle_ts = partial(task_space_sampler_mp,
                            q0 = q0,
                            t0 = t0,
                            plant_builder = plant_builder,
                            frame_names = ['body'],
                            offsets = [np.array([0,0.1,0])],
                            cvx_hulls_of_ROI =cvx_hulls_of_ROI,
                            ts_min = ts_min,
                            ts_max = ts_max,
                            collision_free = True,
                            track_orientation = True,
                            axis_alignment = None#approach_dir[i]
                            )
    ts_samplers.append(sample_handle_ts)

robot_instances = [plant.GetModelInstanceByName("iiwa"), plant.GetModelInstanceByName("wsg")]
checker = SceneGraphCollisionChecker(model = diagram,#.Clone(), 
                    robot_model_instances = robot_instances,
                    distance_function_weights =  [1] * plant.num_positions(),
                    #configuration_distance_function = _configuration_distance,
                    edge_step_size = 0.125)
vgraph_handle = partial(vgraph, checker = checker, parallelize = True) 


if f"7dof_bin_sample_no_clus_{Npts}_{offset_size}_{seed}.pkl" in os.listdir('tmp'):
    with open(f"tmp/7dof_bin_sample_no_clus_{Npts}_{offset_size}_{seed}.pkl", 'rb') as f:
        d = pickle.load(f)
        q_obj = d['q_obj']
        t_obj = d['t_obj']
        ad_mat = d['ad_obj']
else:
    q_obj = []
    t_obj = []
    ad_obj = []
    for sh in ts_samplers:
        q, t, _, res = sh(Npts,[])
        ad_mat = vgraph_handle(q)
        q_obj +=[q]
        t_obj +=[t]
        ad_obj +=[ad_obj]
    with open(f"tmp/7dof_bin_sample_no_clus_{Npts}_{offset_size}_{seed}.pkl", 'wb') as f:
        pickle.dump({'q_obj':q_obj, 't_obj':t_obj, 'ad_obj': ad_obj}, f)

if SHOWSAMPLES:
    
    for i,(ts,qs) in enumerate(zip(t_obj,q_obj)):
        plot_points(meshcat, ts[::4], f"tval_{i}", size=0.01)
        for q in qs[::10]:
            showres(q)
            time.sleep(0.05)

#cluster points
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
cliques_obj = []
num_clusters_obj = []
cluster_sizes_obj = []
for i, (q, t) in enumerate(zip(q_obj, t_obj)):
    q_regularized = (q - np.mean(q, axis = 0))/np.std(q, axis = 0)
    sil = []
    kmsol = []
    min_k = 2#[5,5, 5]
    max_k = int(np.ceil(Npts/min_clique_size))#[20, 20, 100 ]
    clus_vals = np.arange(min_k,max_k, silhouette_method_steps)
    clus_vals_trials = []
    mean_sil = []

    # for k in clus_vals:
    #     #print(k)
    #     sil_loc = []
    #     for _ in range(kmeans_trials):
    #         km = KMeans(n_clusters=k).fit(q_regularized)
    #         kmsol.append(km)
    #         labels = km.labels_
    #         sil_score = silhouette_score(q_regularized, labels, metric='euclidean')
    #         sil.append(sil_score)
    #         sil_loc.append(sil_score)
    #         clus_vals_trials.append(k)
    #     mean_sil.append(np.max(sil_loc))
    # fig = plt.figure()
    # plt.scatter(clus_vals_trials,sil)
    # plt.plot(clus_vals, mean_sil, c = 'r')

    #best_k = clus_vals[np.argmax(mean_sil)]
    #best_clustering_idx = np.argmax(mean_sil[np.argmax(mean_sil)])
    #best_clustering = kmsol[kmeans_trials*np.where(clus_vals == best_k)[0][0] + best_clustering_idx]
    #num_clusters = best_k
    q_clus = [q]#[q[np.where(best_clustering.labels_ == l )[0], :] for l in range(num_clusters)]
    t_clus = [t]#[t[np.where(best_clustering.labels_ == l )[0], :] for l in range(num_clusters)]
    idx_clus = [np.arange(len(q))]#[np.where(best_clustering.labels_ == l )[0] for l in range(num_clusters)]
    clus_sizes = [len(qc) for qc in q_clus]
    cluster_sizes_obj.append(clus_sizes)
    num_clusters_obj.append(len(q_clus))
    vgraph_clus = [vgraph_handle(qc) for qc in q_clus]
    cliques_cluster = []
    for ad in vgraph_clus:
        smin = min_clique_size
        cliques = compute_greedy_clique_partition(ad.toarray(), min_cliuqe_size=smin)
        cl = []
        for c in cliques:
            if len(c)>=smin:
               cl.append(c)
        cliques_cluster.append(cl)
    cliques_clusters_glob = []
    for clus_idx, cliques in enumerate(cliques_cluster):
        for c in cliques:
            if len(c):
                cliques_clusters_glob.append(idx_clus[clus_idx][c])
    cliques_obj.append(cliques_clusters_glob)

print('#'*5 + " STATS ON CLUSTERS " + '#'*5 )
cliques = 0
clusters = 0
for i in range(len(cluster_sizes_obj)):
    print(f"cliques: {len(cliques_obj[i])}")
    print(f"clusters: { num_clusters_obj[i]}")
    print(f"mean cluster size: {np.mean(cluster_sizes_obj[i])}")
    print(f"mean clique size: {np.mean([len(c) for c in cliques_obj[i]])}")
    print(f"max clique size: {np.max([len(c) for c in cliques_obj[i]])}")
    print(f"min clique size: {np.min([len(c) for c in cliques_obj[i]])}")
    print(f"clique sizes: {[len(c) for c in cliques_obj[i]]}")
    cliques += len(cliques_obj[i])
    clusters += num_clusters_obj[i]

print(f"num cliques: {cliques}")
print(f"num clusters: {clusters}")
plt.show(block=False)
plt.pause(2)
t_tot = np.concatenate(tuple(t_obj), axis=0)
q_tot = np.concatenate(tuple(q_obj))
cliques_tot = []
offset = 0
for i, c in enumerate(cliques_obj):
    if i>0:
        offset+= len(q_obj[i-1])
    cliques_tot += [cl + offset for cl in c] 

ells = []
for c in cliques_tot:
    clique_points = q_tot[c]
    ells +=  [Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(clique_points.T)]
min_eigs = []
max_eigs = []
ratios = []

for e in ells:
    eigs = np.linalg.eig(e.A()@e.A().T)[0]    
    if len(eigs) <7:
        maxe = 1e4
        ratio = 1e4
        mine =1 
    else:
        maxe = np.max(eigs)
        mine = np.min(eigs)
        ratio = maxe/mine
    ratios += [ratio]
    min_eigs +=[mine]
    max_eigs +=[maxe]

import matplotlib.pyplot as plt

plt.figure()
bins = plt.hist(np.clip(ratios,a_min = 0, a_max = 1e4))
plt.xlabel('max eig / min eig')
plt.ylabel(f"number of ellipsoids (total {len(ratios)}) ")
plt.title('eigen value ratios of ellipsoids, clipped at 1e4')
plt.show(block = False)
plt.pause(2)

q_min = plant.GetPositionLowerLimits()*1
q_max =  plant.GetPositionUpperLimits()*1
col_func_handle_ = get_col_func(plant, plant_context)
sample_cfree = get_sample_cfree_handle(q_min,q_max, col_func_handle_)

require_sample_point_is_contained = True
iteration_limit = 1
configuration_space_margin = 1.e-3
termination_threshold = -1
num_collision_infeasible_samples = 100
relative_termination_threshold = 0.02
estimate_coverage = get_coverage_estimator(sample_cfree, pts = 3000)

snopt_iris_options = IrisOptions()
snopt_iris_options.require_sample_point_is_contained = require_sample_point_is_contained
snopt_iris_options.iteration_limit = iteration_limit
snopt_iris_options.configuration_space_margin = configuration_space_margin
#snopt_iris_options.max_faces_per_collision_pair = 60
snopt_iris_options.termination_threshold = termination_threshold
#snopt_iris_options.q_star = np.zeros(3)
snopt_iris_options.num_snopt_seed_guesses = 100
snopt_iris_options.num_collision_infeasible_samples = num_collision_infeasible_samples
snopt_iris_options.relative_termination_threshold = relative_termination_threshold
def col_hnd(pt):
    return 1- 1.0*checker.CheckConfigCollisionFree(pt)

iris_handle = partial(SNOPT_IRIS_ellipsoid_parallel,
                        region_obstacles = [],
                        logger = None, 
                        plant_builder = plant_builder,
                        snoptiris_options = snopt_iris_options,
                        estimate_coverage = estimate_coverage,
                        coverage_threshold = 1)

seed_points, metrics, _ = get_iris_metrics([q_tot[c] for c in cliques_tot], col_hnd)

if f"7dof_bin_regions_noapproachdir_{Npts}_{offset_size}_{seed}.pkl" in os.listdir('tmp'):
    with open(f"tmp/7dof_bin_regions_no_clus_{Npts}_{offset_size}_{seed}.pkl", 'rb') as f:
        d = pickle.load(f)
        regions_red = d['r']
        succs_sp = d['succs_sp']
        seed_points = d['sp']
        metrics = d['metrics']
else:
    regions_red, succs_sp, is_full = iris_handle(seed_points, metrics, [])
    with open(f"tmp/7dof_bin_regions_no_clus_{Npts}_{offset_size}_{seed}.pkl", 'wb') as f:
        pickle.dump({'r':regions_red, 'succs_sp':succs_sp, 'sp': seed_points, 'metrics': metrics}, f)


