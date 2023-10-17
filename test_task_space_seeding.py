from environments import get_environment_builder
import numpy as np

env_builder = get_environment_builder('5DOFUR5')
plant, scene_graph, diagram, diagram_context, plant_context, meshcat = env_builder(True)




