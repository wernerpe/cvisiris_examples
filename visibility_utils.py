import numpy as np
from pydrake.all import HPolyhedron, VisibilityGraph
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def shrink_regions(regions, offset_fraction = 0.25):
	shrunken_regions = []
	for r in regions:
		offset = offset_fraction*np.min(1/np.linalg.eig(r.MaximumVolumeInscribedEllipsoid().A())[0])
		rnew = HPolyhedron(r.A(), r.b()-offset)
		shrunken_regions.append(rnew)	
	return shrunken_regions

def point_in_regions(pt, regions):
    for r in regions:
        if r.PointInSet(pt.reshape(-1,1)):
            return True
    return False

def generate_distinct_colors(n):
    cmap = plt.cm.get_cmap('hsv', n)  # Choose a colormap
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]  # Convert colormap to hexadecimal colors
    return colors

def point_near_regions(pt, regions, tries = 10, eps = 0.1):
    for _ in range(tries):
        n = 2*eps*(np.random.rand(len(pt))-0.5)
        checkpt = pt+n
        for r in regions:
            if r.PointInSet(checkpt.reshape(-1,1)):
                return True
    return False

from pydrake.all import InverseKinematics
from functools import partial

def eval_cons(q, c, tol):
    return 1-1*float(c.evaluator().CheckSatisfied(q, tol))

def get_col_func(plant, plant_context, min_dist = 0.01, tol = 0.01):
    ik = InverseKinematics(plant, plant_context)
    collision_constraint = ik.AddMinimumDistanceConstraint(min_dist, 0.001)
    return partial(eval_cons, c=collision_constraint, tol=tol)

def sample_cfree(N, M, regions, q_min, q_diff, dim, col_func_handle):
    points = []
    it = 0
    for _ in range(N):
        while it<M:
            rand = np.random.rand(dim)
            q_s = q_min + rand*q_diff
            col = col_func_handle(q_s)
            if not col and not point_in_regions(q_s, regions):
                break #Ratfk.ComputeQValue(q_s, q_star)
            it+=1
        if it == M:
            return np.array(points), True
        points.append(q_s)
        it = 0
    return np.array(points), False

def get_sample_cfree_handle(q_min, q_max, col_func_handle):
    return partial(sample_cfree, q_min = q_min, q_diff = q_max-q_min, dim = len(q_min), col_func_handle = col_func_handle)

def estimate_coverage(regions, pts = 5000, sample_cfree_handle = None):
    pts_, _ = sample_cfree_handle(pts, 1000,[])
    inreg = 0
    for pt in pts_:
        if point_in_regions(pt, regions): inreg+=1
    return inreg/pts

def get_coverage_estimator(sample_cfree_handle, pts = 5000):
    return partial(estimate_coverage, sample_cfree_handle= sample_cfree_handle, pts = 5000)

def vgraph(points, checker, parallelize):
    ad_mat = VisibilityGraph(checker.Clone(), np.array(points).T, parallelize = parallelize)
    N = ad_mat.shape[0]
    for i in range(N):
        ad_mat[i,i] = False
    #TODO: need to make dense for now to avoid wierd nx bugs for saving the metis file.
    return  ad_mat