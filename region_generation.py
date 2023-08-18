from pydrake.geometry.optimization import IrisInConfigurationSpace

def SNOPT_IRIS_ellipsoid(q_seeds, metrics, old_regions, region_obstacles, logger, plant, context, snoptiris_options, estimate_coverage, coverage_threshold):
    regions = []
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
            if estimate_coverage(old_regions+regions)>= coverage_threshold:
                return regions, True
        except:
            print("error, SNOPT IRIS FAILED")
    return regions, False

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