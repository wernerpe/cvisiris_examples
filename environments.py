from ur3e_demo import UrDiagram
from functools import partial
from pydrake.all import (StartMeshcat,
                         RobotDiagramBuilder,
                         MeshcatVisualizer,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         RigidTransform,
                         RotationMatrix,
                         MeshcatVisualizerParams,
                         Role,
                         RollPitchYaw,
                         Meldis,
                         AddDefaultVisualization,
                         Box
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

def plant_builder_2dof_flipper_obs(usemeshcat = False):
    if usemeshcat:
        #meshcat = StartMeshcat()
        meldis = Meldis()
        meshcat = meldis.meshcat
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()
    # plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    # parser = Parser(plant)
    rel_path_cvisiris = "../cvisiris_examples/"
    #oneDOF_iiwa_asset = rel_path_cvisiris + "assets/oneDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/oneDOF_iiwa7_with_box_collision.sdf")
    twoDOF_iiwa_asset = rel_path_cvisiris + "assets/twoDOF_iiwa7_with_box_collision.sdf"#FindResourceOrThrow("drake/C_Iris_Examples/assets/twoDOF_iiwa7_with_box_collision.sdf")

    box_asset = rel_path_cvisiris + "assets/box_small.urdf" #FindResourceOrThrow("drake/C_Iris_Examples/assets/box_small.urdf")
    roi_asset = rel_path_cvisiris + "assets/roi_box.urdf"
    obs_asset = rel_path_cvisiris + "assets/2d_obs.urdf"
    path_repo = os.path.dirname(os.path.abspath('')) #os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # replace with {path to cvisirsexamples repo}
    parser.package_map().Add("cvisiris", path_repo+"/cvisiris_examples/assets")
    models = []
    models.append(parser.AddModelFromFile(box_asset, "box"))
    
    models.append(parser.AddModelFromFile(obs_asset, "obs_box"))
    models.append(parser.AddModelFromFile(obs_asset, "obs_box2"))
    models.append(parser.AddModelFromFile(obs_asset, "obs_box3"))
    models.append(parser.AddModelFromFile(twoDOF_iiwa_asset, "iiwatwodof"))
    models.append(parser.AddModelFromFile(roi_asset, "roi_box"))
    #models.append(parser.AddModelFromFile(oneDOF_iiwa_asset, "iiwaonedof"))

    locs = [[0.,0.,0.],
            [0,0,0,],
            [0,0.6,0.19],
            [0.,.55,0.],
            ]
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[0]),
        RigidTransform(locs[0]))
    from pydrake.geometry import ProximityProperties, AddContactMaterial
    from pydrake.all import CoulombFriction
    proximity_properties = ProximityProperties()
    AddContactMaterial(dissipation=0.1,
                        point_stiffness=250.0,
                        friction=CoulombFriction(0.9, 0.5),
                        properties=proximity_properties)
    box_shape = Box(0.1, 0.1, 0.5)
    frame = plant.GetBodyByName("box1")
    #X1 = RigidTransform(np.array([0,1.1,0.1]))
    X1 = RigidTransform(RollPitchYaw(np.pi/10,0,0).ToRotationMatrix(),np.array([0,1.1,0.10]))
    plant.RegisterVisualGeometry(frame,X1,
                                              box_shape, "obs1",
                                              np.array([1, 0., 0., 1]))
    obs1 = plant.RegisterCollisionGeometry(frame, X1,
                                              box_shape, "obs1",
                                               proximity_properties)
    box_shape = Box(0.05, 0.2, 0.05)
    frame = plant.GetBodyByName("box1")
    # X2 = RigidTransform(np.array([0,0.75,-0.75]))
    X2 = RigidTransform(np.array([0,0.75,10.75]))
    plant.RegisterVisualGeometry(frame,X2,
                                              box_shape, "obs2",
                                              np.array([1, 0., 0., 1]))
    obs2 = plant.RegisterCollisionGeometry(frame, X2,
                                              box_shape, "obs2",
                                               proximity_properties)
    box_shape = Box(0.05, 0.1, 0.05)
    # X3 = RigidTransform(np.array([0,1.0,-0.85]))
    X3 = RigidTransform(np.array([0,1.0,10.85]))
    plant.RegisterVisualGeometry(frame,X3,
                                              box_shape, "obs3",
                                              np.array([1, 0., 0., 1]))
    obs3 = plant.RegisterCollisionGeometry(frame, X3,
                                              box_shape, "obs3",
                                               proximity_properties)
    
    box_shape = Box(0.1, 0.1, 0.4)
    # X3 = RigidTransform(RollPitchYaw(np.pi/2.5,0,0).ToRotationMatrix(), np.array([0,1.1,-0.78]))
    X3 = RigidTransform(RollPitchYaw(np.pi/3,0,0).ToRotationMatrix(), np.array([0,1.1,0.15]))
    plant.RegisterVisualGeometry(frame,X3,
                                              box_shape, "obs4",
                                              np.array([1, 0., 0., 1]))
    obs4 = plant.RegisterCollisionGeometry(frame, X3,
                                              box_shape, "obs4",
                                               proximity_properties)
    
    #obsbox
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[1]),
        RigidTransform(np.array([0,0,10])))
    #obsbox2
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[2]),
        RigidTransform(RollPitchYaw([-np.pi/4,0, 0]).ToRotationMatrix(), np.array([0,-0.46,10.43])))
    #obsbox3
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[3]),
        RigidTransform(RollPitchYaw([-np.pi/2,0, 0]).ToRotationMatrix(), np.array([0,-0.3,10.6])))
    plant.WeldFrames(plant.world_frame(), 
        plant.GetFrameByName("base", models[-1]),
        RigidTransform(RollPitchYaw(0,0,0).ToRotationMatrix(), np.array([0,-0.05,0.15])))
    plant.WeldFrames(plant.world_frame(), 
                    plant.GetFrameByName("iiwa_twoDOF_link_0", models[-2]), 
                    RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[-1]))
    # plant.WeldFrames(plant.world_frame(), 
    #                 plant.GetFrameByName("iiwa_oneDOF_link_0", models[2]), 
    #                 RigidTransform(RollPitchYaw([0,0, -np.pi/2]).ToRotationMatrix(), locs[2]))

    # roi_shape = Box(0.1, 0.1, 1)

    # plant.RegisterVisualGeometry(plant.GetBodyByName("base"), 
    #                              RigidTransform(RollPitchYaw([-np.pi/4,0, 0]).ToRotationMatrix(), np.array([0,0,1])),
    #                                         roi_shape, "roi_area",
    #                                         np.array([0, 1., 0., 0.5]))
    
    plant.Finalize()
    inspector = scene_graph.model_inspector()
    if usemeshcat:
        meshcat_params = MeshcatVisualizerParams()
        meshcat_params.role = Role.kIllustration
        visualizer = AddDefaultVisualization(builder.builder(), meshcat)
    # visualizer = MeshcatVisualizer.AddToBuilder(
    #         builder.builder(), scene_graph, meshcat, meshcat_params)
    # X_WC = RigidTransform(RollPitchYaw(0,0,0),np.array([5, 4, 2]) ) # some drake.RigidTransform()
    # meshcat.SetTransform("/Cameras/default", X_WC) 
    # meshcat.SetProperty("/Background", "top_color", [0.8, 0.8, 0.6])
    # meshcat.SetProperty("/Background", "bottom_color",
    #                                 [0.9, 0.9, 0.9])
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)
    if usemeshcat:
        print(meshcat.web_url())
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None

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

def plant_builder_7dof_bins(usemeshcat = False):
    if usemeshcat:
        #meshcat = StartMeshcat()
        meldis = Meldis()
        meshcat = meldis.meshcat
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()
    #parser.package_map().Add("cvisirisexamples", missing directory)
    
    directives_file = "7dof_bins_example.yaml"#FindResourceOrThrow() 
    path_repo = os.path.dirname(os.path.abspath('')) #os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # replace with {path to cvisirsexamples repo}
    parser.package_map().Add("cvisiris", path_repo+"/cvisiris_examples/assets")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    if usemeshcat:
        #par = MeshcatVisualizerParams()
        #par.role = Role.kIllustration
        #visualizer = MeshcatVisualizer.AddToBuilder(builder.builder(), scene_graph, meshcat, par)
        visualizer = AddDefaultVisualization(builder.builder(), meshcat)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None


def environment_builder_14dof_iiwas(usemeshcat = False):
    builder = RobotDiagramBuilder()
    if usemeshcat: meshcat = StartMeshcat()
    # if export_sg_input:
    #     scene_graph = builder.AddSystem(SceneGraph())
    #     plant = MultibodyPlant(time_step=0.0)
    #     plant.RegisterAsSourceForSceneGraph(scene_graph)
    #     builder.ExportInput(scene_graph.get_source_pose_port(plant.get_source_id()), "source_pose")
    # else:
    plant = builder.plant()
    scene_graph = builder.scene_graph()# AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser = builder.parser()
    parser.package_map().Add("bimanual", os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/cvisiris_examples/assets_bimanual")

    directives = LoadModelDirectives("assets_bimanual/models/bimanual_iiwa_with_shelves.yaml")
    models = ProcessModelDirectives(directives, plant, parser)

    plant.Finalize()
    if usemeshcat:
        meshcat_visual_params = MeshcatVisualizerParams()
        meshcat_visual_params.delete_on_initialization_event = False
        meshcat_visual_params.role = Role.kIllustration
        meshcat_visual_params.prefix = "visual"
        meshcat_visual = MeshcatVisualizer.AddToBuilder(
            builder.builder(), scene_graph, meshcat, meshcat_visual_params)

        meshcat_collision_params = MeshcatVisualizerParams()
        meshcat_collision_params.delete_on_initialization_event = False
        meshcat_collision_params.role = Role.kProximity
        meshcat_collision_params.prefix = "collision"
        meshcat_collision_params.visible_by_default = False
        meshcat_collision = MeshcatVisualizer.AddToBuilder(
            builder.builder(), scene_graph, meshcat, meshcat_collision_params)


    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(
        diagram_context)
    return plant, scene_graph, diagram, diagram_context, plant_context, meshcat if usemeshcat else None

def get_environment_builder(environment_name):
    valid_names = ['2DOFFLIPPER','3DOFFLIPPER', '5DOFUR5', '7DOFIIWA', '7DOFBINS', '14DOFIIWAS']
    if not environment_name in valid_names:
        raise ValueError(f"Choose a valid environment {valid_names}")
    if environment_name == '2DOFFLIPPER':
        return plant_builder_2dof_flipper_obs
    if environment_name == '3DOFFLIPPER':
        return plant_builder_3dof_flipper
    elif environment_name == '5DOFUR5':
        return plant_builder_5dof_ur5
    elif environment_name == '7DOFIIWA':
        return plant_builder_7dof_iiwa
    elif environment_name == '7DOFBINS':
        return plant_builder_7dof_bins
    elif environment_name == '14DOFIIWAS':
        return environment_builder_14dof_iiwas
    return None