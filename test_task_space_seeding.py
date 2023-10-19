import numpy as np
from functools import partial
import ipywidgets as widgets

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
from environments import get_environment_builder
from visualization_utils import plot_points, plot_regions
from pydrake.all import VPolytope, Role
from task_space_seeding_utils import (get_cvx_hulls_of_bodies,
                                      get_AABB_cvxhull,
                                      task_space_sampler_mp,
                                      task_space_sampler)
plant_builder = get_environment_builder('7DOFBINS')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)

scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
    diagram_context)

#Determine task space regions of interest
geom_names = ['bin_base', 'bin_base', 'shelves_body']
model_names = ['binL', 'binR', 'shelves']
cvx_hulls_of_ROI, bodies = get_cvx_hulls_of_bodies(geom_names, model_names, plant, scene_graph, scene_graph_context)
ts_min, ts_max, cvxh_hpoly = get_AABB_cvxhull(cvx_hulls_of_ROI)

plot_regions(meshcat, cvx_hulls_of_ROI, opacity=0.2)
q0  = np.zeros(7)
plant.SetPositions(plant_context, q0)
plant.ForcedPublish(plant_context)
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
                           )


q, t, _, res = sample_handle_ts(100,[])