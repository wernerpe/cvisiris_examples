from pydrake.geometry.optimization import IrisInConfigurationSpace, IrisOptions
import multiprocessing as mp
from functools import partial
import pickle

def SNOPT_IRIS_ellipsoid(q_seeds, metrics, old_regions, region_obstacles, logger, plant, context, snoptiris_options, estimate_coverage, coverage_threshold):
    regions = []
    succsessful_seedpoints = []
    is_full = False
    for reg_indx, (q_seed, metric) in enumerate(zip(q_seeds, metrics)):
        #q_seed = Ratforwardkin.ComputeQValue(s_seed.reshape(-1,1), np.zeros((7,1)))
        #print('snopt iris call')
        snoptiris_options.configuration_obstacles = []
        if len(region_obstacles):
            snoptiris_options.configuration_obstacles = region_obstacles
        plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q_seed.reshape(-1,1))
        try:
            #r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
            snoptiris_options.starting_ellipse = metric
            r = IrisInConfigurationSpace(plant, plant.GetMyMutableContextFromRoot(context), snoptiris_options)
            r_red = r.ReduceInequalities()
            print(f"[SNOPT IRIS]: Region:{reg_indx} / {len(q_seeds)}")
            #run the certifier
            # cert = cspace_free_polytope.BinarySearch(set(),
            # 								r.A(),
            # 								r.b(), 
            # 								np.array(s_seed),
            # 								binary_search_options)
            if logger is not None: logger.log_region(r_red)
            # r = cert.certified_polytope
            regions.append(r)
            succsessful_seedpoints.append([q_seed, metric])
            if estimate_coverage(old_regions+regions)>= coverage_threshold:
                return regions, succsessful_seedpoints, True
        except:
            print("error, SNOPT IRIS FAILED")
    return regions, succsessful_seedpoints, False

def SNOPT_IRIS_ellipsoid_worker(q_seeds_and_metrics,
                                require_sample_point_is_contained,
                                iteration_limit,
                                configuration_space_margin,
                                termination_threshold,
                                num_collision_infeasible_samples,
                                relative_termination_threshold,
                                logdir,
                                plant_builder):
    plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder()
    default_b = np.concatenate((plant.GetPositionLowerLimits(), -plant.GetPositionUpperLimits()), axis =0) 
    snopt_iris_options = IrisOptions()
    snopt_iris_options.require_sample_point_is_contained = require_sample_point_is_contained
    snopt_iris_options.iteration_limit = iteration_limit
    snopt_iris_options.configuration_space_margin = configuration_space_margin
    #snopt_iris_options.max_faces_per_collision_pair = 60
    snopt_iris_options.termination_threshold = termination_threshold
    #snopt_iris_options.q_star = np.zeros(3)
    snopt_iris_options.num_collision_infeasible_samples = num_collision_infeasible_samples
    snopt_iris_options.relative_termination_threshold = relative_termination_threshold
    q_seeds = q_seeds_and_metrics[0]
    metrics = q_seeds_and_metrics[1] 
    regions = []
    successful_seed_ells = []
    for reg_indx, (q_seed, metric) in enumerate(zip(q_seeds, metrics)):
        plant.SetPositions(plant.GetMyMutableContextFromRoot(diagram_context), q_seed.reshape(-1,1))
        try:
            #r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
            snopt_iris_options.starting_ellipse = metric
            #r = IrisInConfigurationSpace(plant, plant.GetMyMutableContextFromRoot(diagram_context), snopt_iris_options)
            r = IrisInConfigurationSpace(plant, diagram_context, snopt_iris_options)
            ##r_red = r.ReduceInequalities()
            r_red = r
            #check if returned region is jointlimit box
            if len(r.b()) == len(default_b): 
                print(f"[SNOPT IRIS Worker] Region failed at {q_seed}")
            else:
                print(f"[SNOPT IRIS Worker]: Region:{reg_indx} / {len(q_seeds)}")
                regions.append(r_red)
                successful_seed_ells.append([q_seed, metric])
                if logdir is not None:
                    data = {'ra': r_red.A(), 'rb': r_red.b()}

                    with open(logdir+f"/regions/region_{np.random.rand():.3f}"+".pkl", 'wb') as f:
                        pickle.dump(data,f)
        except:
            print(f"[SNOPT IRIS Worker] Region failed at {q_seed}!")
    return regions, successful_seed_ells

import numpy as np

def SNOPT_IRIS_ellipsoid_parallel(q_seeds, 
                                  metrics, 
                                  old_regions, 
                                  region_obstacles, 
                                  logger, 
                                  plant_builder, 
                                  snoptiris_options, 
                                  estimate_coverage, 
                                  coverage_threshold):
    regions = []
    succ_seed_pts = []
    #is_full = False
    nr_pts=len(q_seeds)
    chunk_list = []
    
    #split  into batches 
    split = mp.cpu_count()-2
    chunks_seedpoints = np.array_split(q_seeds, split)
    chunks_metrics = np.array_split(metrics, split)
    chunks = [(c_sp, c_m) for c_sp, c_m in zip(chunks_seedpoints, chunks_metrics)]
    SNOPTIRISHANDLE = partial(SNOPT_IRIS_ellipsoid_worker,
                              require_sample_point_is_contained = True,
                              iteration_limit = snoptiris_options.iteration_limit,
                              configuration_space_margin = snoptiris_options.configuration_space_margin,
                              termination_threshold = snoptiris_options.termination_threshold,
                              num_collision_infeasible_samples = snoptiris_options.num_collision_infeasible_samples,
                              relative_termination_threshold = snoptiris_options.relative_termination_threshold,
                              logdir = logger.expdir if logger is not None else None,
                              plant_builder = plant_builder)

    pool = mp.Pool(processes= split)
    results = pool.map(SNOPTIRISHANDLE, chunks)
    for r in results:
        regions += r[0]
        succ_seed_pts += r[1]
    current_coverage_est = estimate_coverage(regions+old_regions)
    if current_coverage_est>= coverage_threshold:
        #space is already full, stop generating more regions
        return regions, succ_seed_pts, True
    else:
        return regions, succ_seed_pts, False



def SNOPT_IRIS_obstacles(q_seeds, region_obstacles, old_regions, logger, plant, context, snoptiris_options, estimate_coverage, coverage_threshold):
    regions = []
    for reg_indx, q_seed in enumerate(q_seeds):
        #q_seed = Ratforwardkin.ComputeQValue(s_seed.reshape(-1,1), np.zeros((7,1)))
        #print('snopt iris call')
        snoptiris_options.configuration_obstacles = []
        if len(region_obstacles):
            snoptiris_options.configuration_obstacles = region_obstacles
        plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q_seed.reshape(-1,1))
        try:
            #r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
            r = IrisInConfigurationSpace(plant, plant.GetMyMutableContextFromRoot(context), snoptiris_options)
            r_red = r.ReduceInequalities()
            print(f"[SNOPT IRIS]: Region:{reg_indx} / {len(q_seeds)}")
            #run the certifier
            # cert = cspace_free_polytope.BinarySearch(set(),
            # 								r.A(),
            # 								r.b(), 
            # 								np.array(s_seed),
            # 								binary_search_options)
            if logger is not None: logger.log_region(r_red)
            # r = cert.certified_polytope
            regions.append(r)
            print(estimate_coverage(old_regions+regions))
            if estimate_coverage(old_regions+regions)>= coverage_threshold:
                return regions, True
        except:
            print("error, SNOPT IRIS FAILED")
    return regions, False

def SNOPT_IRIS_obstacles_simple(q_seeds, region_obstacles, plant, context, snoptiris_options):
    regions = []
    for reg_indx, q_seed in enumerate(q_seeds):
        #q_seed = Ratforwardkin.ComputeQValue(s_seed.reshape(-1,1), np.zeros((7,1)))
        #print('snopt iris call')
        snoptiris_options.configuration_obstacles = []
        if len(region_obstacles):
            snoptiris_options.configuration_obstacles = region_obstacles
        plant.SetPositions(plant.GetMyMutableContextFromRoot(context), q_seed.reshape(-1,1))
        try:
            #r = IrisInRationalConfigurationSpace(plant, plant.GetMyContextFromRoot(context), q_star, snoptiris_options)
            r = IrisInConfigurationSpace(plant, plant.GetMyMutableContextFromRoot(context), snoptiris_options)
            r_red = r.ReduceInequalities()
            print(f"[SNOPT IRIS]: Region:{reg_indx} / {len(q_seeds)}")
            #run the certifier
            # cert = cspace_free_polytope.BinarySearch(set(),
            # 								r.A(),
            # 								r.b(), 
            # 								np.array(s_seed),
            # 								binary_search_options)
           
            regions.append(r)
        except:
            print("error, SNOPT IRIS FAILED")
    return regions











