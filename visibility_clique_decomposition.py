import numpy as np
from independent_set_solver import solve_max_independent_set_integer
from clique_covers import compute_greedy_clique_partition, get_iris_metrics,compute_minimal_clique_partition_nx, compute_cliques_REDUVCC, extend_cliques
# from visibility_utils import shrink_regions
# import time
from time import strftime,gmtime
# import matplotlib.pyplot as plt
import pickle
from pydrake.all import HPolyhedron, Hyperellipsoid
# import networkx as nx


class VisCliqueDecomp:
    def __init__(self,
                 N = 400, #computational cost per iteration
                 eps = 0.05, #bernoulli test parameter 2
                 max_iterations = 10,
                 sample_cfree = None,
                 col_handle = None,
                 build_vgraph = None,
                 iris_w_obstacles = None,
                 verbose = False,
                 logger = None,
                 approach = 0,
                 extend_cliques = False,
                 min_clique_size = 10
                 ):
        self.min_clique_size = min_clique_size
        self.extend_cliques = extend_cliques
        self.approach = approach
        self.logger = logger
        if self.logger is not None: self.logger.time()
        self.vb = verbose
        self.sample_cfree = sample_cfree
        self.build_vgraph = build_vgraph
        self.iris_w_obstacles = iris_w_obstacles
        self.col_handle = col_handle
        self.N = N
        self.eps = eps
        self.M = 1e5 #attempts to insert a point in cfree
        self.maxit = max_iterations
        if self.vb: 
            #print(strftime("[%H:%M:%S] ", gmtime()) +'[VisCliqueDecomp] GuardInsertion attempts M:', str(self.M))
            print(strftime("[%H:%M:%S] ", gmtime()) +f"[VisCliqueDecomp] Attempting to cover {100-100*eps:.1f} '%' of Cfree ")
        
        self.vgraph_points = []
        self.vgraph_admat = []
        self.seed_points = []
        self.regions = []
        self.region_groups = []
        self.cliques = []
        self.metrics_iteration = []

    def run(self):
        done = False
        it = 0
        fails = 0
        while it<self.maxit:
            #sample N points in cfree
            points, sampling_failed = self.sample_cfree(int(self.N/(1+it*0)), self.M, self.regions)
            if sampling_failed:
                print(strftime("[%H:%M:%S] ", gmtime()) +f"[VisCliqueDecomp] Sampling failed after {self.M} attempts ")
        
            self.vgraph_points.append(points)
            if self.logger is not None: self.logger.time()

            #build visibility graph
            #self.sregs = shrink_regions(self.regions, offset_fraction=self.region_pullback)
            ad_mat = self.build_vgraph(points)
            self.vgraph_admat.append(ad_mat)
            if self.logger is not None: self.logger.time()
            
            #solve clique partition problem and sort cliques
            if self.approach == 0:
                cliques_idxs = compute_cliques_REDUVCC(ad_mat, maxtime = 30)
            elif self.approach == 1:
                cliques_idxs = compute_greedy_clique_partition(ad_mat.toarray(), min_cliuqe_size=self.min_clique_size)
            elif self.approach == 2:
                cliques_idxs = compute_minimal_clique_partition_nx(ad_mat)
            cliques_idxs_e = extend_cliques(ad_mat.toarray(), cliques_idxs) if self.extend_cliques else cliques_idxs
            # cliques_idxs = compute_cliques_REDUVCC(ad_mat)#compute_minimal_clique_partition_nx(ad_mat) if self.use_nx else compute_greedy_clique_partition(ad_mat) # #compute_greedy_clique_partition(ad_mat)
            nr_cliques = len(cliques_idxs)

            end_idx_cand = np.where(np.array([len(c) for c in cliques_idxs]) < self.min_clique_size)[0]
            if len(end_idx_cand):
                end_idx = end_idx_cand[0]
            else:
                end_idx = len(cliques_idxs)
            self.cliques_step = np.array([points[i,:] for i in cliques_idxs_e[:end_idx]])
            self.cliques.append(cliques_idxs_e[:end_idx])
            nr_cliques_big_enough = len(self.cliques_step)
            #compute seed points and initial iris metric
            self.seed_points, self.metrics, unscaled = get_iris_metrics(self.cliques_step, self.col_handle)
            self.metrics_iteration.append([Hyperellipsoid(un.A(), seed_p) for un, seed_p in zip(unscaled, self.seed_points)])
            #self.seed_points +=[points[mhs_idx, :].squeeze()]
            if self.vb : 
                print(strftime("[%H:%M:%S] ", gmtime()) +'[VisCliqueDecomp] Found ', nr_cliques_big_enough, ' cliques')
                if nr_cliques_big_enough == 0:
                    fails +=1
                    if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[VisCliqueDecomp] No cliques found')
                    if fails ==10:
                        return self.regions
                else:
                    fails = 0
            if self.logger is not None: self.logger.time()

            #grow the regions with obstacles
            regions_step, successful_seed_points, is_full_iris = self.iris_w_obstacles(self.seed_points, self.metrics, self.regions)
            if not len(self.seed_points) == len(successful_seed_points):
                #discard failed seed points
                self.seed_points = [s[0] for s in successful_seed_points]
                self.metrics = [s[1] for s in successful_seed_points]
            self.regions += regions_step
            self.region_groups.append(regions_step)
            if self.logger is not None: self.logger.time()
            
            frac_cliques_in_region = []
            for c,r in zip(self.cliques_step, regions_step):
                size = len(c)                
                nr_in_reg = 0
                for p in c:
                    if r.PointInSet(p): nr_in_reg +=1
                frac_cliques_in_region.append(1.0*nr_in_reg/size)

            if self.logger is not None: self.logger.log(self, it)
            if self.logger is not None and len(regions_step): self.logger.log_frac_contained_points(np.mean(frac_cliques_in_region), np.min(frac_cliques_in_region), np.max(frac_cliques_in_region))
            if is_full_iris:
                if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[VisCliqueDecomp] Coverage met, terminated on Iris step')
                return self.regions
            it+=1
        if self.logger is not None: self.logger.log_string(strftime("[%H:%M:%S] ", gmtime()) +'[VisCliqueDecomp] Maxit reached')
        return self.regions
    
    def save_state(self, path):
        region_groups_A = [[r.A() for r in g] for g in self.region_groups] 
        region_groups_b = [[r.b() for r in g] for g in self.region_groups]
        data = {
        'vg':self.vgraph_points,
        'vad':self.vgraph_admat,
        'sp':self.seed_points,
        'ra':region_groups_A,
        'rb':region_groups_b}
        with open(path+".pkl", 'wb') as f:
            pickle.dump(data,f)

    def load_state(self, path):
        with open(path,'rb') as f:
            data = pickle.load(f)

        self.region_groups = [[HPolyhedron(a,b) for a,b in zip(ga, gb)] for ga, gb in zip(data['ra'], data['rb'])]    
        self.regions = []
        for g in self.region_groups:
            for r in g:
                self.regions.append(r)
        self.vgraph_points = data['vg']
        self.vgraph_admat = data['vad']
        self.seed_points = data['sp']
