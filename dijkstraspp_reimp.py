from pydrake.all import HPolyhedron, VPolytope, Hyperellipsoid
import numpy as np
from visualization_utils import plot_HPoly
import matplotlib.pyplot as plt
from dijkstraspp import DijkstraSPP

class checker:
    def __init__(self, obstacles):
        self.obstacles = [o for o in obstacles]
        self.res = 0.01
    
    def CheckEdgeCollisionFreeParallel(self, a, b):
        tv = np.arange(0,1,self.res)
        for t in tv:
            pt = t*b + (1-t)*a
            if point_in_regions(pt, self.obstacles):
                return False
        return True
    
    def CheckConfigCollisionFree(self, pt):
        return not point_in_regions(pt, self.obstacles)

    def CheckConfigsCollisionFree(self, pts):
        return np.array([1*self.CheckConfigCollisionFree(pt) for pt in pts])

dom_L = np.array([-10, -10])
dom_U = np.array([10, 10])


num_obstacles = 20
np.random.seed(1)
obstacles = []
for _ in range(num_obstacles):
    size = np.random.randint(5)+2
    pts = np.clip(size*0.3*np.random.randn(10,2) + 0.8*(dom_U-dom_L)*np.random.rand(1,2)+0.8*dom_L, a_min= dom_L, a_max = dom_U)
    obstacles.append(HPolyhedron(VPolytope(pts.T)))


# points = np.array([[0,-1.8],[-5,5]])
points = np.array([[2,-2],[-6,5]])
center_points = np.array([[0, 0],
                          [1.3, 1.4],
                          [2.2, 4.2],
                          [-1.6, 1],
                          [-3,3.7],
                          [-4.2, 5],
                          [0,6],
                          [0,3.7]])

sizes = np.array([[1,2],
                  [2,2],
                  [0.5,5.],
                  [3,1],
                  [1,5],
                  [2.2,2],
                  [6,0.3],
                  [0.8,5]])

from visibility_utils import point_in_regions
regions =[]
for c, s in zip(center_points, sizes):
    l = c- s/2
    u = c + s/2
    regions.append(HPolyhedron.MakeBox(l,u))


sizes = np.array([[1,4],
                  [1,3],
                  #[2,2],
                  ])

center_points = np.array([[-2, 3.5],
                          [1.3, 4],
                          #[1.6, 1.4],
                          ])

# obstacles = []
# for c, s in zip(center_points, sizes):
#     l = c- s/2
#     u = c + s/2
#     obstacles.append(HPolyhedron.MakeBox(l,u))








col_check = checker(obstacles)
# dspp = DijkstraSPP(regions, col_check, verbose=True)

# wps, dist = dspp.solve(points[0], points[1], refine_path=True)
# wps = np.array(wps)
# ad_coo, fixed_idx = dspp.extend_adjacency_mat(points[0], points[1])
# ad_dense = ad_coo.toarray()
# reppoints_tot = np.concatenate((dspp.reppts, np.array(points)), axis = 0)

from irisbs import IRISBS
fig, ax = plt.subplots(figsize = (10,10))
for r in obstacles:
    plot_HPoly(ax, r, color = 'k', zorder=10)
seed_point = np.array([0, 0.5])
domain = HPolyhedron.MakeBox(dom_L, dom_U)
ell = Hyperellipsoid.MakeHypersphere(1e-3, seed_point)
import time
t1 = time.time()
reg = IRISBS(domain, ell, col_check, N= 1000, NumFails=2, backoff=1e-2, bisection_steps=10, gradient_steps=2, use_bisection_only=True, plot=ax)
t2 = time.time()
print(f"time {t2-t1}")

# ell2 = reg.MaximumVolumeInscribedEllipsoid()
# reg = IRISBS(domain, ell2, col_check, N= 100, NumFails=10, backoff=0.1, bisection_steps=20, gradient_steps=5, use_bisection_only=False, plot=ax)
# ell2 = reg.MaximumVolumeInscribedEllipsoid()
# reg = IRISBS(domain, ell2, col_check, N= 100, NumFails=10, backoff=0.1, bisection_steps=20, gradient_steps=5, use_bisection_only=False, plot=ax)
from pydrake.all import Iris, IrisOptions

opts = IrisOptions()
opts.iteration_limit = 1
t1 = time.time()
reg2 = Iris(obstacles, ell.center(),domain,opts)
t2 = time.time()
print(f"time {t2-t1}")





# for p,c in zip(points, ['k','r']):
#     ax.scatter(p[0], p[1], c = c)
plt.scatter(seed_point[0], seed_point[1], s = 100)
plot_HPoly(ax, reg)
#plot_HPoly(ax, reg2)
ax.axis('equal')
# for r in regions:
#     plot_HPoly(ax, r)
# for r in dspp.safe_sets:
#     plot_HPoly(ax, r, color = 'r')



# dm = dspp.dist_mat.toarray()
# for i in range(len(dspp.safe_sets)+len(regions)-1):
#     for j in range(i+1, len(dspp.safe_sets)+len(regions)):
#         if dm[i,j]!=0:
#             x = [reppoints_tot[i][0], reppoints_tot[j][0]]
#             y = [reppoints_tot[i][1], reppoints_tot[j][1]]
#             ax.plot(x,y, linewidth = 1, c = 'k',zorder =10)
# ax.scatter(dspp.reppts[:,0], dspp.reppts[:,1], c = 'k')
# ax.plot(wps[:,0], wps[:, 1], linewidth = 5 , c = 'r')
plt.show()