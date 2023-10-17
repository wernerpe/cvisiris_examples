from functools import partial
import numpy as np
from visibility_utils import (get_col_func, 
                              get_sample_cfree_handle,
                              get_coverage_estimator,
                              vgraph)
from pydrake.all import (SceneGraphCollisionChecker)
from pydrake.geometry.optimization import IrisOptions#, HPolyhedron, Hyperellipsoid
from environments import get_environment_builder
from visibility_logging import CliqueApproachLogger
from visibility_clique_decomposition import VisCliqueInflation
from region_generation import SNOPT_IRIS_ellipsoid_parallel
    

plant_builder = get_environment_builder('7DOFBINS')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = plant_builder(usemeshcat=True)



