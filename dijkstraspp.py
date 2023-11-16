from scipy.sparse import coo_matrix
import numpy as np
from pydrake.all import MathematicalProgram, Solve, eq
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix 

class DijkstraSPP:
    def __init__(self, regions, verbose = True):
        self.verbose = verbose
        self.safe_sets = []
        self.safe_adjacencies = []
        self.regions = [r for r in regions]
        for id1, r1 in enumerate(regions):
            for id2, r2 in enumerate(regions):
                if id1 != id2 and id1 < id2:
                    if r1.IntersectsWith(r2):
                        self.safe_sets.append(r1.Intersection(r2))
                        self.safe_adjacencies.append([id1, id2])
                        
        reppts = np.array([s.ChebyshevCenter() for s in self.safe_sets])
        
        safe_ad = lil_matrix((len(self.safe_sets), len(self.safe_sets)))
        for id in range(len(regions)):
            safeset_idxs_in_region_id = np.where([id in s for s in self.safe_adjacencies])[0]
            for i,id1 in enumerate(safeset_idxs_in_region_id[:-1]):
                for id2 in safeset_idxs_in_region_id[i:]:     
                    safe_ad[id1, id2] = 1
                    safe_ad[id2, id1] = 1
                    
        #optimize_reppts
        prog = MathematicalProgram()
        repopt = prog.NewContinuousVariables(*reppts.shape)
        for i, s in enumerate(self.safe_sets):
            s.AddPointInSetConstraints(prog, repopt[i,:])

        for i in range(len(self.safe_sets)):
            for j in range(i+1, len(self.safe_sets)):
                if safe_ad[i,j]==1:
                    prog.AddCost(np.linalg.norm(repopt[i,:]- repopt[j,:]))
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
        ad_mat = self.extend_adjacency_mat(start, target)
        if ad_mat is not None:
            wps, dist = self.dijkstra_in_configspace(adj_mat=ad_mat)
            if dist<0:
                print('[DijkstraSPP] Points not reachable')
                return [], -1
            if refine_path:
                location_wps_optimized, dist_optimized = self.refine_path_SOCP(wps, 
                                                                        start, 
                                                                        target, 
                                                                        )
                return location_wps_optimized, dist_optimized
            
            intermediate_nodes = [self.reppts[idx, :] for idx in wps[1:-1]]
            waypoints = [start] + intermediate_nodes + [target]
            return waypoints, dist
        else:
            print('[DijkstraSPP] Points not in regions')
            return [], -1


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
        #first check point memberships
        start_idx = []
        target_idx = []
        for idx, r in enumerate(self.regions):
            if r.PointInSet(start):
                start_idx.append(idx)
            if r.PointInSet(target):
                target_idx.append(idx)
        if len(start_idx)==0 or len(target_idx)==0:
            print('[DijkstraSPP] Points not in set, idxs', start_idx,', ', target_idx)
            return None
        N = self.dist_mat.shape[0] + 2
        data = list(self.dist_mat.data)
        rows = list(self.dist_mat.row)
        cols = list(self.dist_mat.col)
        start_adj_idx = N-2
        target_adj_idx = N-1
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
        return ad_mat_extend
    
    def refine_path_SOCP(self, wps, start, target):
            #intermediate_nodes = [self.node_intersections[idx] for idx in wps[1:-1]]
            dim = len(start)
            prog = MathematicalProgram()
            wps = np.array(wps)
            
            int_waypoints = prog.NewContinuousVariables(len(wps[1:-1]), dim)
            for i, wp in enumerate(wps[1:-1]):
                self.safe_sets[wp].AddPointInSetConstraints(prog, int_waypoints[i,:])
            
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