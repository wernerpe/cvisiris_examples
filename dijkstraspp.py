from scipy.sparse import coo_matrix
import numpy as np
from pydrake.all import MathematicalProgram, Solve, eq
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix 

class DijkstraSPP:
    def __init__(self, regions, checker, verbose = True):
        self.verbose = verbose
        self.checker = checker
        self.safe_sets = []
        self.safe_adjacencies = []
        self.dim = regions[0].ambient_dimension()
        self.regions = [r for r in regions]
        for id1, r1 in enumerate(regions):
            if (id1%10) == 0:
                if self.verbose: print('[DijkstraSPP] Pre-Building adjacency matrix ', id1,'/', len(self.regions))
            for id2, r2 in enumerate(regions):
                if id1 != id2 and id1 < id2:
                    if r1.IntersectsWith(r2):
                        self.safe_sets.append(r1.Intersection(r2))
                        self.safe_adjacencies.append([id1, id2])
                        
        reppts = np.array([s.ChebyshevCenter() for s in self.safe_sets])
        
        safe_ad = lil_matrix((len(self.safe_sets), len(self.safe_sets)))
        for id in range(len(regions)):
            safeset_idxs_in_region_id = np.where([id in s for s in self.safe_adjacencies])[0]
            if (id%10) == 0:
                if self.verbose: print('[DijkstraSPP] Pre-Building safe-adjacency matrix ', id,'/', len(self.regions))
            for i,id1 in enumerate(safeset_idxs_in_region_id[:-1]):
                for id2 in safeset_idxs_in_region_id[i:]:     
                    safe_ad[id1, id2] = 1
                    safe_ad[id2, id1] = 1
                    
        #optimize_reppts
        prog = MathematicalProgram()
        repopt = prog.NewContinuousVariables(*reppts.shape)
        if self.verbose: print(f"[DijkstraSPP] Optimizing {len(reppts)} pointlocations in safe-sets")
        for i, s in enumerate(self.safe_sets):
            s.AddPointInSetConstraints(prog, repopt[i,:])

        for i in range(len(self.safe_sets)):
            for j in range(i+1, len(self.safe_sets)):
                if safe_ad[i,j]==1:
                    t = prog.NewContinuousVariables(self.dim+1, 't'+str(i*(j+10)))
                    prog.AddConstraint(eq(t[1:], repopt[i,:]- repopt[j,:]))
                    prog.AddLorentzConeConstraint(t)
                    prog.AddCost(t[0])
        result = Solve(prog)
        print(result.is_success())
        self.reppts = result.GetSolution(repopt)
        dist_mat = lil_matrix((len(self.safe_sets), len(self.safe_sets)))
        for i in range(len(self.safe_sets)):
            for j in range(i+1, len(self.safe_sets)):
                if safe_ad[i,j]==1:
                    dist = np.linalg.norm(self.reppts[i,:]- self.reppts[j,:])
                    dist_mat[i,j] = dist_mat[j,i] = dist
        self.dist_mat = dist_mat.tocoo()
    
    def solve(self, 
              start,
              target,
              refine_path = True):
        ad_mat, fixed_idx = self.extend_adjacency_mat(start, target)
        if ad_mat is not None:
            wps, dist = self.dijkstra_in_configspace(ad_mat)
            if dist<0:
                print('[DijkstraSPP] Points not reachable')
                return [], -1
            #bezier_spline = self.smooth_phase(wps, start, target)
            if refine_path:
                location_wps_optimized, dist_optimized = self.refine_path_SOCP(wps, 
                                                                        start, 
                                                                        target,
                                                                        fixed_idx 
                                                                        )
                return location_wps_optimized, dist_optimized
            
            intermediate_nodes = [self.reppts[idx, :] for idx in wps[1:-1]]
            waypoints = [start] + intermediate_nodes + [target]
            return waypoints, dist
        else:
            print('[DijkstraSPP] Points not in regions')
            return [], -1

    def get_region_sequence(self, wps, start, target):
        region_sequence = []
        regions_start = []
        regions_target = []
        for i,r in enumerate(self.regions):
            if r.PointinSet(start):
                regions_start.append(i) 
            if r.PointinSet(target):
                regions_target.append(i)
        prev = regions_start
        for i in range(len(wps)):
            region_idx = list(set(prev)&set(self.safe_adjacencies[wps[i]]))
            if len(region_idx)==0:
                raise ValueError("This should not happen")
            region_sequence.append(region_idx[0])
            prev = self.safe_adjacencies[wps[i]]
        region_idx = list(set(prev)&set(regions_target))
        if len(region_idx)==0:
                raise ValueError("This should not happen")  
        region_sequence.append(region_idx[0])
        return region_sequence
    
    def get_distances(self, wps, start, target):
        dists = []
        prev = start
        for wp in wps:
            dists.append(np.linalg.norm(prev-self.reppts[wp,:]))
            prev = self.reppts[wp,:]
        dists.append(np.linalg.norm(prev- target))
        return dists
    
    def smooth_phase(self, wps, start, target):
        region_sequence = self.get_region_sequence(wps[1:-1], start, target)
        distances = self.get_distances(wps[1:-1], start, target)
        init_times = np.cumsum(distances)
        
    def dijkstra_in_configspace(self, adj_mat):
        # convention for start and target: source point is second to last and target is last point
        src = adj_mat.shape[0] -2
        target = adj_mat.shape[0] -1
        dist, pred = dijkstra(adj_mat, directed=False, indices=src, return_predecessors=True)
        #print(f'{len(np.argwhere(pred == -9999))} disconnected nodes'), #np.argwhere(pred == -9999))
        idxs = (pred == -9999)
        pred[idxs] = -1000000
        dist[idxs] = -1000000
        sp_list = []
        sp_length = dist[target]
        if sp_length<0:
            return [], sp_length
        current_idx = target
        sp_list.append(current_idx)
        while not current_idx == adj_mat.shape[0] - 2:
            current_idx = pred[current_idx]
            sp_list.append(current_idx)
            if current_idx==src: break
        return [idx for idx in sp_list[::-1]], sp_length


    def extend_adjacency_mat(self, start, target):
        N = self.dist_mat.shape[0] + 2
        data = list(self.dist_mat.data)
        rows = list(self.dist_mat.row)
        cols = list(self.dist_mat.col)
        start_adj_idx = N-2
        target_adj_idx = N-1
        #first check point memberships
        start_idx = []
        target_idx = []
        fixed_idxs = []
        for idx, r in enumerate(self.regions):
            if r.PointInSet(start):
                start_idx.append(idx)
            if r.PointInSet(target):
                target_idx.append(idx)
        if len(start_idx)==0:
            #distances = np.linalg.norm(start - self.reppts, axis = 1)
            #idx_sort = np.argsort(distances)[::-1]
            print('[DijkstraSPP] Attempting visibility extension')
            idx_vis_start = []
            for idx_r, rp in enumerate(self.reppts):
                if self.checker.CheckEdgeCollisionFreeParallel(start, rp):
                    idx_vis_start.append(idx_r)
                    dist = np.linalg.norm(start-rp)
                    data.append(dist)
                    rows.append(start_adj_idx)
                    cols.append(idx_r)
                    data.append(dist)
                    rows.append(idx_r)
                    cols.append(start_adj_idx)
            if len(idx_vis_start) ==0:
                return None, None
            fixed_idxs.append(1)
        if len(target_idx)==0:
            #distances = np.linalg.norm(start - self.reppts, axis = 1)
            #idx_sort = np.argsort(distances)[::-1]
            print('[DijkstraSPP] Attempting visibility extension')
            idx_vis_target = []
            for idx_r, rp in enumerate(self.reppts):
                if self.checker.CheckEdgeCollisionFreeParallel(start, rp):
                    idx_vis_target.append(idx_r)
                    dist = np.linalg.norm(target-rp)
                    data.append(dist)
                    rows.append(target_adj_idx)
                    cols.append(idx_r)
                    data.append(dist)
                    rows.append(idx_r)
                    cols.append(target_adj_idx)
            if len(idx_vis_target) ==0:
                return None, None
            fixed_idxs.append(-2)
        if len(start_idx):
            for id in start_idx:
                safeset_idxs_in_region_id = np.where([id in s for s in self.safe_adjacencies])[0]
                for idx in safeset_idxs_in_region_id:
                    dist = np.linalg.norm(start - self.reppts[idx, :])
                    data.append(dist)
                    rows.append(start_adj_idx)
                    cols.append(idx)
                    data.append(dist)
                    rows.append(idx)
                    cols.append(start_adj_idx)
        if len(target_idx):
            for id in target_idx:
                safeset_idxs_in_region_id = np.where([id in s for s in self.safe_adjacencies])[0]
                for idx in safeset_idxs_in_region_id:
                    dist = np.linalg.norm(target - self.reppts[idx, :])
                    data.append(dist)
                    rows.append(target_adj_idx)
                    cols.append(idx)
                    data.append(dist)
                    rows.append(idx)
                    cols.append(target_adj_idx)
        if len(list(set(start_idx)& set(target_idx))):
            dist = np.linalg.norm(target - start)
            data.append(dist)
            rows.append(target_adj_idx)
            cols.append(start_adj_idx)
            data.append(dist)
            rows.append(start_adj_idx)
            cols.append(target_adj_idx)     
            
        ad_mat_extend = coo_matrix((data, (rows, cols)), shape=(N, N))
        return ad_mat_extend, fixed_idxs
    
    def refine_path_SOCP(self, wps, start, target, fixed_idx):
            #intermediate_nodes = [self.node_intersections[idx] for idx in wps[1:-1]]
            dim = len(start)
            prog = MathematicalProgram()
            wps = np.array(wps)
            int_waypoints = prog.NewContinuousVariables(len(wps[1:-1]), dim)
            for i, wp in enumerate(wps[1:-1]):
                self.safe_sets[wp].AddPointInSetConstraints(prog, int_waypoints[i,:])
            #convert fixed_idx 
            fixed_idx_conv = [len(wps)+i if i<0 else i for i in fixed_idx]
            for i in fixed_idx_conv:
                prog.AddLinearConstraint(eq(int_waypoints[i-1,:], self.reppts[wps[1:-1][i-1],:]))

            prev = start
            cost = 0 
            for idx in range(len(wps[1:-1])):
                t = prog.NewContinuousVariables(dim+1, 't'+str(idx))
                prog.AddConstraint(eq(t[1:], prev-int_waypoints[idx]))
                prev = int_waypoints[idx]
                prog.AddLorentzConeConstraint(t)
                cost += t[0]
            t = prog.NewContinuousVariables(dim+1, 'tend')
            prog.AddConstraint(eq(t[1:], prev-target))
            prog.AddLorentzConeConstraint(t)
            cost += t[0]
            prog.AddCost(cost)

            res = Solve(prog)
            if res.is_success():
                path = [start]
                for i in res.GetSolution(int_waypoints):
                    path.append(i)
                path.append(target)
                wps_start = [self.reppts[idx] for idx in wps[1:-1]]
                dist_start = 0
                prev = start
                for wp in wps_start + [target]:
                    #dist_start += np.linalg.norm()#* np.array([4.0,3.5,3,2.5,2,2.5,1])
                    a = prev-wp
                    dist_start += np.sqrt(a.T@a)
                    prev = wp
                if self.verbose: print("[DijkstraSPP] optimized distance/ start-distance = {opt:.2f} / {start:.2f} = {res:.2f}".format(opt = res.get_optimal_cost(), start = dist_start, res = res.get_optimal_cost()/dist_start))
                return path, res.get_optimal_cost()
            else:
                print("[DijkstraSPP] Refine path SCOP failed")
                return None, None
            



################# COPIED FROM FPP CODEBASE##########################################
#https://github.com/cvxgrp/fastpathplanning/blob/new_retiming/fastpathplanning/bezier.py
from scipy.special import binom
from bisect import bisect

class BezierCurve:

    def __init__(self, points, a=0, b=1):

        assert b > a

        self.points = points
        self.M = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):

        c = np.array([self.berstein(t, n) for n in range(self.M + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):

        c1 = binom(self.M, n)
        c2 = (t - self.a) / self.duration 
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.M - n) 

        return value

    def start_point(self):

        return self.points[0]

    def end_point(self):

        return self.points[-1]
        
    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.M / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):

        A = l2_matrix(self.M, self.d)
        p = self.points.flatten()

        return p.dot(A.dot(p)) * self.duration
    
def l2_matrix(M, d):

    A = np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        for n in range(M + 1):
            A[m, n] = binom(M, m) * binom(M, n) / binom(2 * M, m + n)
    A /= (2 * M + 1)
    A_kron = np.kron(A, np.eye(d))

    return A_kron

class CompositeBezierCurve:

    def __init__(self, beziers):

        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def find_segment(self, t):

        return min(bisect(self.transition_times, t) - 1, self.N - 1)

    def __call__(self, t):

        i = self.find_segment(t)

        return self.beziers[i](t)

    def start_point(self):

        return self.beziers[0].start_point()

    def end_point(self):

        return self.beziers[-1].end_point()

    def derivative(self):

        return CompositeBezierCurve([b.derivative() for b in self.beziers])

    def l2_squared(self):

        return sum(bez.l2_squared() for bez in self.beziers)