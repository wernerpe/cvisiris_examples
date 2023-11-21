import numpy as np
from functools import partial
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
                         Cylinder,
                         VPolytope,
                         Role)
import time
import pydrake
from environments import get_environment_builder
from task_space_seeding_utils import solve_ik_problem, task_space_sampler_mp
from pydrake.all import SurfaceTriangle, TriangleSurfaceMesh
from visibility_utils import get_col_func
import pickle, os, mcubes
from visibility_utils import (sample_in_union_of_polytopes, 
                              get_AABB_cvxhull,
                              )
from visualization_utils import plot_points


plant_builder = get_environment_builder('2DOFFLIPPER')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)

meshcat.SetTransform('/Cameras/default', RigidTransform(RollPitchYaw(0,0,1.5*np.pi/2).ToRotationMatrix(),  np.array([4,-3,1])))
scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
    diagram_context)

inspector = scene_graph.model_inspector()
b = plant.GetBodyFrameIdIfExists(plant.GetBodyByName("roi_box", plant.GetModelInstanceByName("roi_box")).index(), )
ids = inspector.GetGeometries(b, Role.kIllustration)
vp = [VPolytope(scene_graph.get_query_output_port().Eval(scene_graph_context), id) for id in ids]
roi = HPolyhedron(vp[0])


tmin, tmax, aabb = get_AABB_cvxhull([roi])

#sample in roi using IK
starting_confs = [np.array([1.5,1.0]),
                  np.array([-0.3,-1.0])]
 
ts_samplers = []
for q0 in starting_confs:    
    plant.SetPositions(plant_context, q0)
    plant.ForcedPublish(plant_context)
    t0 = plant.EvalBodyPoseInWorld(plant_context,  plant.GetBodyByName("iiwa_twoDOF_link_7")).translation()  
    sample_handle_ts = partial(task_space_sampler_mp,
                            q0 = q0,
                            t0 = t0,
                            plant_builder = plant_builder,
                            frame_names = ['iiwa_twoDOF_link_7'],
                            offsets = [np.array([0,0.0,0.05])],
                            cvx_hulls_of_ROI = [roi],
                            ts_min = tmin,
                            ts_max = tmax,
                            collision_free = True,
                            track_orientation = False,
                            
                            )
    ts_samplers.append(sample_handle_ts)

q_obj = []
t_obj = []
Npts = 10
for sh in ts_samplers:
    q, t, _, res = sh(Npts,[])
    q_obj +=[q]
    t_obj +=[t]

q_obj = np.concatenate(tuple(q_obj), axis=0)
t_obj = np.concatenate(tuple(t_obj), axis=0)

for i,t in enumerate([t_obj]):
    plot_points(meshcat, t, f"iksol{i}", size = 0.01)


_offset_meshcat_2 = np.array([-1,-5, 1.5])
q_min = np.concatenate((np.array([-0]),plant.GetPositionLowerLimits()))
q_max =  np.concatenate((np.array([-0]),plant.GetPositionUpperLimits()))
col_func_handle_ = get_col_func(plant, plant_context)
def check_collision_by_ik(q0,q1,q2, min_dist=1e-5):
    q = np.array([q1,q2])
    if np.any(q>q_max[1:]):
        return 0
    if np.any(q<q_min[1:]):
        return 0
    if np.any(q0!=0):
        return 1
    return 1-1.*col_func_handle_(q) 

def plot_collision_constraint(N = 50, q_min = q_min, q_max= q_max):
    if f"col_cons2d2{N}.pkl" in os.listdir('tmp'):
        with open(f"tmp/col_cons2d{N}.pkl", 'rb') as f:
            d = pickle.load(f)
            vertices = d['vertices']
            triangles = d['triangles']
    else:  
        vertices, triangles = mcubes.marching_cubes_func(
        tuple(
                q_min-0.3), tuple(
                q_max+0.3), 3, N, N, check_collision_by_ik, 0.5)
        with open(f"tmp/col_cons2d{N}.pkl", 'wb') as f:
                d = {'vertices': vertices, 'triangles': triangles}
                pickle.dump(d, f)

    tri_drake = [SurfaceTriangle(*t) for t in triangles]

    vertices += _offset_meshcat_2.reshape(-1,3)
    meshcat.SetObject("/collision_constraint/c1",
                                    TriangleSurfaceMesh(tri_drake, vertices),
                                    Rgba(1, 0, 0, 1))
    meshcat.SetObject("/collision_constraint/c2",
                                    TriangleSurfaceMesh(tri_drake, vertices),
                                    Rgba(0.6, 0.0, 0, 1), wireframe = True)
    
plot_collision_constraint(60)
plant.SetPositions(plant_context, np.array([np.pi/4,0]))
diagram.ForcedPublish(diagram_context)
for i,q in enumerate([q_obj]):
    qext = np.concatenate((np.zeros((q.shape[0],1)),q), axis =1)
    plot_points(meshcat, qext+_offset_meshcat_2, f"ik _{i}", size = 0.03)

time.sleep(10)