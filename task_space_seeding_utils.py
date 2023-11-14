from pydrake.all import (RigidTransform,
                         RotationMatrix,
                         InverseKinematics,
                         Solve,
                         Role,
                         VPolytope,
                         HPolyhedron
                         )
import numpy as np
from tqdm import tqdm

from visibility_utils import (sample_in_union_of_polytopes,
                              point_in_regions,
                              get_AABB_cvxhull
                              )

import multiprocessing as mp
from functools import partial

# def get_ik_problem_solver(plant_builder, frame_names, offsets, collision_free= False, track_orientation = True):
#     plant_ik, scene_graph, diagram, diagram_context, plant_context_ik, meshcat = plant_builder()
#     frames = [plant_ik.GetFrameByName(f) for f in frame_names]
#     def solve_ik_problem(poses, q0, collision_free = collision_free):
#         ik = InverseKinematics(plant_ik, plant_context_ik)
#         for pose, f, o in zip(poses, frames, offsets):
#             ik.AddPositionConstraint(
#                 f,
#                 o,
#                 plant_ik.world_frame(),
#                 pose.translation()-0.02,
#                 pose.translation()+0.02,
#             )
#             if track_orientation:
#                 ik.AddOrientationConstraint(
#                     f,
#                     RotationMatrix(),
#                     plant_ik.world_frame(),
#                     pose.rotation(),
#                     0.1,
#                 )
#         if collision_free:
#             ik.AddMinimumDistanceConstraint(0.001, 0.1)
#         prog = ik.get_mutable_prog()
#         q = ik.q()
#         prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
#         prog.SetInitialGuess(q, q0)
#         result = Solve(ik.prog())
#         if result.is_success():
#                 return result.GetSolution(q)
#         return None
#     return solve_ik_problem

def get_cvx_hulls_of_bodies(geometry_names, model_names, plant, scene_graph, scene_graph_context, scaling = 1):
    inspector = scene_graph.model_inspector()
    bodies_of_interest = []
    cvx_hulls = []
    for g_n, m_n in zip(geometry_names, model_names):
        b = plant.GetBodyFrameIdIfExists(
                                        plant.GetBodyByName(g_n, 
                                                            plant.GetModelInstanceByName(m_n)
                                                            ).index())
        bodies_of_interest +=[b]
        ids = inspector.GetGeometries(b, Role.kProximity)
        vp = [VPolytope(scene_graph.get_query_output_port().Eval(scene_graph_context), id) for id in ids]
        verts = np.concatenate(tuple([v.vertices().T for v in vp]), axis=0)
        mean = np.mean(verts,axis=0).reshape(1,-1)
        cvx_hulls += [HPolyhedron(VPolytope(scaling*(verts.T- mean.T)+mean.T))]
    return cvx_hulls, bodies_of_interest

from pydrake.all import RotationMatrix, AngleAxis

def sample_random_orientations(N, seed = 1230):
    #np.random.seed(seed)
    vecs = np.random.randn(N,3)
    vecs = vecs/np.linalg.norm(vecs)
    angs = 2*np.pi*(np.random.rand(N)-0.5)
    rotmats = [AngleAxis(ang, ax) for ang, ax in zip(angs, vecs)]
    return rotmats


def solve_ik_problem(poses,
                     q0, 
                     frames, 
                     offsets, 
                     plant_ik, 
                     plant_context_ik, 
                     collision_free = True,
                     track_orientation = True):
    
    ik = InverseKinematics(plant_ik, plant_context_ik)
    for pose, f, o in zip(poses, frames, offsets):
        ik.AddPositionConstraint(
            f,
            o,
            plant_ik.world_frame(),
            pose.translation()-0.02,
            pose.translation()+0.02,
        )
        if track_orientation:
            ik.AddOrientationConstraint(
                f,
                RotationMatrix(),
                plant_ik.world_frame(),
                pose.rotation(),
                0.1,
            )
    if collision_free:
        ik.AddMinimumDistanceConstraint(0.01, 0.1)
    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if result.is_success():
            return result.GetSolution(q)
    return None

def task_space_sampler(num_points_seed_q0_t0_ax_al_tuple, 
                       regions,  
                       plant_builder,
                       frame_names,
                       offsets, 
                       cvx_hulls_of_ROI,
                       ts_min, #bounding box in task space to sample in
                       ts_max,
                       collision_free = True, 
                       track_orientation = True,
                       MAXIT = int(1e4)):
        n_points = num_points_seed_q0_t0_ax_al_tuple[0]
        seed = num_points_seed_q0_t0_ax_al_tuple[1]
        q0 = num_points_seed_q0_t0_ax_al_tuple[2]
        t0 = num_points_seed_q0_t0_ax_al_tuple[3]
        preferred_axis_alignment = num_points_seed_q0_t0_ax_al_tuple[4]
        plant_ik, _, _, _, plant_context_ik, _ = plant_builder()
        frames = [plant_ik.GetFrameByName(f) for f in frame_names]
    
        q_points = [q0]
        t_points = [t0]
        np.random.seed(seed)    
        if preferred_axis_alignment is not None:
            sc = np.ones(3)
            sc[preferred_axis_alignment] = 2

        for i in tqdm(range(n_points)):
            for it in range(MAXIT):
                if preferred_axis_alignment is not None:
                    vecs = np.random.randn(1,3)*sc
                    vecs = vecs/np.linalg.norm(vecs)
                    angs = 2*np.pi*(np.random.rand(1)-0.5)
                    if preferred_axis_alignment ==2:
                        rot_corr = RotationMatrix.MakeXRotation(-np.pi/2)
                    # if preferred_axis_alignment == 1:
                    #     rot_corr = RotationMatrix.MakeYRotation(np.pi/2)
                    if preferred_axis_alignment == 0:
                        rot_corr = RotationMatrix.MakeZRotation(-np.pi/2)
                    rotmat =  rot_corr@ RotationMatrix(AngleAxis(angs[0], vecs[0,:]) )
                else:  
                    rotmat = sample_random_orientations(1)[0]
                t_point = sample_in_union_of_polytopes(1, cvx_hulls_of_ROI, [ts_min, ts_max]).squeeze() #t_min + t_diff*np.random.rand(3)
                idx_closest = np.argmin(np.linalg.norm(np.array(t_points)-t_point))
                q0 = q_points[idx_closest]
                res = solve_ik_problem([RigidTransform(rotmat, t_point)], 
                                       q0= q0,
                                       plant_ik=plant_ik,
                                       plant_context_ik=plant_context_ik,
                                       frames=frames,
                                       offsets=offsets,
                                       collision_free = collision_free,
                                       track_orientation =track_orientation)
                if res is not None and not point_in_regions(res, regions):
                    q_points.append(res)
                    t_points.append(t_point)
                    #print(f"found point {i} seed {seed}")
                    break
                #else:
                #    print(f"failed seed {seed}")
                if it ==MAXIT:
                    print("[SAMPLER] CANT FIND IK SOLUTION")
                    return None, None, True
        return np.array(q_points[1:]), np.array(t_points[1:]), False    

def task_space_sampler_mp(n_points, 
                          regions,  
                          plant_builder,
                          frame_names,
                          offsets,
                          cvx_hulls_of_ROI,
                          ts_min,
                          ts_max, 
                          q0 = None, 
                          t0 = None,
                          collision_free = True, 
                          track_orientation = True,
                          axis_alignment = None
                          ):
        
        processes = mp.cpu_count()
        pool = mp.Pool(processes=processes)
        pieces = np.array_split(np.ones(n_points), processes)
        if q0 is not None:
            n_chunks = [[int(np.sum(p)), np.random.randint(1000), q0, t0, axis_alignment] for p in pieces]
        else:
             plant_ik, _, _, _, plant_context_ik, _ = plant_builder()
             qmax = plant_ik.GetPositionUpperLimits()
             qmin = plant_ik.GetPositionLowerLimits()
             dim = len(qmax)
             qdiff =qmax- qmin
             chunks = []
             for p in pieces:
                q0 = qdiff*np.random.rand(dim) +qmin
                plant_ik.SetPositions(plant_context_ik, q0)
                t0 = plant_ik.EvalBodyPoseInWorld(plant_context_ik,  plant_ik.GetBodyByName(frame_names[0])).translation()   
                chunks.append([int(np.sum(p)), np.random.randint(1000), q0, t0])
        q_pts = []
        t_pts = []
        is_full = False
        SAMPLERHANDLE = partial(task_space_sampler, 
                                regions = regions, 
                                plant_builder = plant_builder,
                                frame_names = frame_names, 
                                offsets = offsets,
                                cvx_hulls_of_ROI = cvx_hulls_of_ROI,
                                ts_min = ts_min, #bounding box in task space to sample in
                                ts_max = ts_max,
                                collision_free = collision_free,
                                track_orientation = track_orientation) 
        #print(n_chunks)
        results = pool.map(SAMPLERHANDLE, n_chunks)
        for r in results:
            if len(r[0]):
                q_pts.append(r[0])
                t_pts.append(r[1])
                is_full |= r[2]
        return np.concatenate(tuple(q_pts), axis = 0), np.concatenate(tuple(t_pts), axis = 0), is_full, results


#def task_space_sampler_mp(n_points, regions, q0 = q0, t0 = t0, collision_free = collision_free, )



