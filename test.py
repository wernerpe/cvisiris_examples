import pickle
from pydrake.all import HPolyhedron
import os
import numpy as np

exp_with = 'logs/experiment_5dof_ur_shelf_1_1000_0.200greedy20230820181043'
exp_without = 'logs/experiment_5dof_ur_shelf_1_1000_0.200greedy20230820181354'

reg_pth = os.listdir(exp_with+'/regions')
regs_with = []
for p in reg_pth:
    with open(exp_with+'/regions/'+p, 'rb') as f:
        d =pickle.load(f)
        regs_with.append(HPolyhedron(d['ra'], d['rb']))

vols_with = [r.MaximumVolumeInscribedEllipsoid().Volume() for r in regs_with]
print(np.mean(vols_with))

reg_pth = os.listdir(exp_without+'/regions')
regs_without = []
for p in reg_pth:
    with open(exp_without+'/regions/'+p, 'rb') as f:
        d =pickle.load(f)
        regs_without.append(HPolyhedron(d['ra'], d['rb']))

vols_without = [r.MaximumVolumeInscribedEllipsoid().Volume() for r in regs_without]
print(np.mean(vols_without))