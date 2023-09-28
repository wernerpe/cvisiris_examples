from ur3e_demo import UrDiagram
from functools import partial
from pydrake.all import (StartMeshcat,
                         RobotDiagramBuilder,
                         MeshcatVisualizer,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         RigidTransform,
                         MeshcatVisualizerParams,
                         Role,
                         RollPitchYaw
                         )
import numpy as np
import os

def plant_builder_5dof_ur5(use_meshcat = False, cfg = {'add_shelf': True, 'add_gripper': True}):
    ur = UrDiagram(num_ur = 1, weld_wrist = True, add_shelf = cfg['add_shelf'],
                    add_gripper = cfg['add_gripper'], use_meshcat=use_meshcat)

    if use_meshcat: meshcat = ur.meshcat
    plant = ur.plant
    diagram_context = ur.diagram.CreateDefaultContext()
    ur.diagram.ForcedPublish(diagram_context)
    diagram = ur.diagram
    plant_context = ur.plant.GetMyMutableContextFromRoot(
            diagram_context)
    # scene_graph_context = ur.scene_graph.GetMyMutableContextFromRoot(
    #     diagram_context)
    scene_graph = ur.scene_graph
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if use_meshcat else None


def plant_builder_3dof_flipper(usemeshcat = False):
    meshcat = StartMeshcat()
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()
    # plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    # parser = Parser(plant)
    rel_path_cvisiris = "../cvisiris_examples/"
    oneDOF_iiwa_asset = rel_path_cvisiris + "assets/oneDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf")
    twoDOF_iiwa_asset = rel_path_cvisiris + "assets/twoDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/twoDOF_iiwa7_with_box_collision.sdf")

    box_asset = rel_path_cvisiris + "assets/box_small.urdf" #FindResourceOrThrow("drake/C_Iris_Examples/assets/box_small.urdf")

    models = []
    models.append(parser.AddModelFromFile(box_asset, "box"))
    models.append(parser.AddModelFromFile(twoDOF_iiwa_asset, "iiwatwodof"))
    models.append(parser.AddModelFromFile(oneDOF_iiwa_asset, "iiwaonedof"))

    locs = [[0.,0.,0.],
            [0.,.55,0.],
            [0.,-.55,0.]]
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[0]),
        RigidTransform(locs[0]))
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("iiwa_twoDOF_link_0", models[1]), 
                    RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[1]))
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("iiwa_oneDOF_link_0", models[2]), 
                    RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[2]))

    plant.Finalize()
    inspector = scene_graph.model_inspector()

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.role = Role.kIllustration
    visualizer = MeshcatVisualizer.AddToBuilder(
            builder.builder(), scene_graph, meshcat, meshcat_params)
    # X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array([5, 4, 2]) ) # some drake.RigidTransform()
    # meshcat.SetTransform("/Cameras/default", X_WC) 
    # meshcat.SetProperty("/Background", "top_color", [0.8, 0.8, 0.6])
    # meshcat.SetProperty("/Background", "bottom_color",
    #                                 [0.9, 0.9, 0.9])
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)
    print(meshcat.web_url())
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None
    

def plant_builder_7dof_iiwa(usemeshcat = False):
    if usemeshcat:
        meshcat = StartMeshcat()
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()
    #parser.package_map().Add("cvisirisexamples", missing directory)
    if usemeshcat:
        visualizer = MeshcatVisualizer.AddToBuilder(builder.builder(), scene_graph, meshcat)
    directives_file = "7_dof_directives_newshelf.yaml"#FindResourceOrThrow() 
    path_repo = os.path.dirname(os.path.abspath('')) #os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # replace with {path to cvisirsexamples repo}
    parser.package_map().Add("cvisiris", path_repo+"/cvisiris_examples/assets")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None


def get_environment_builder(environment_name):
    valid_names = ['3DOFFLIPPER', '5DOFUR5', '7DOFIIWA']
    if not environment_name in valid_names:
        raise ValueError(f"Choose a valid environment {valid_names}")
    if environment_name == '3DOFFLIPPER':
        return plant_builder_3dof_flipper
    elif environment_name == '5DOFUR5':
        return plant_builder_5dof_ur5
    elif environment_name == '7DOFIIWA':
        return plant_builder_7dof_iiwa
    return None