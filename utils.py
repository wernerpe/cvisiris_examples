import pickle
import os
import numpy as np
from pydrake.all import HPolyhedron


def load_regions_from_experiment_dir(dir, it = None):
    # poly_names = os.listdir("logs/"+exp_name+"/regions")
    data_chkpts = os.listdir(dir+"/data")
    # poly_names.sort()
    itmax = np.max([int(x.replace('it_', '').replace('.pkl','')) for x in data_chkpts])
    regions = []
    # for p in poly_names:
    #     with open("logs/"+exp_name+"/regions/"+p, 'rb') as f:
    #         d = pickle.load(f)
    #     regions.append(HPolyhedron(d['ra'], d['rb']))
    if it is None:
        with open(dir+"/data/"+f"it_{itmax}.pkl", 'rb') as f:
            d2 = pickle.load(f)
    else:
        with open(dir+"/data/"+f"it_{it}.pkl", 'rb') as f:
            d2 = pickle.load(f)
    #seed_points = d2['sp'][-1]
    for rga, rgb in zip(d2['ra'], d2['rb']):
        for a,b in zip(rga, rgb):
            regions.append(HPolyhedron(a,b))

    return regions