# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from . import mdp
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import CurriculumTermCfg as CurrTerm
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import CameraCfg

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the PickPlace scene with a robot and a PickPlace.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    tote_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ToteTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.75, -0.45, 0.45], 
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            scale=(2.0, 1.0, 0.8),
        ),
    )

    head_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HeadTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.75, -0.45, 1.0], 
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
            scale=(2.0, 1.0, 0.01),
        ),
    )

    # Target tote from demo - static as rigid body
    target_tote: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetTote",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.05, 0.5, 0.2],
            rot=[0, 0, 0, 1],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.14, 2.22, 2.31),
        ),
    )


    # camera_head = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link6/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.3, 0.05, 0.015), rot=(0.61237, 0.35355, -0.61237, 0.35355), convention="ros"),
    # )

    # camera_overhead = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/OverheadCamera",
    #     update_period=0.1,  # 10Hz update rate
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     debug_vis=True,  # 启用相机位置调试可视化
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.3, 2.5), rot=(0.0, 0.1, 0.0, 0.0),  # 相机向下俯视桌面
    #     ),
    # )


    # ====================================================================YCB Objects on table====================================================================
    cracker_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.57 - 0.15, 0 + 0.15, 0.55],
            rot=[0, 0, 0.7071, -0.7071],  # X轴旋转90度
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(0.8, 0.8, 0.8),
        ),
    )
    
    # sugar_box: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/sugar_box",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.57, 0 + 0.15, 0.55],
    #         rot=[0, 0, 0.7071, -0.7071],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    #         scale=(0.8, 0.8, 0.8),
    #     ),
    # )
    
    # tomato_soup_can: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/tomato_soup_can",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.57 + 0.15, 0 + 0.15, 0.55],
    #         rot=[0, 0, 0.7071, -0.7071],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    #         scale=(0.8, 0.8, 0.8),
    #     ),
    # )
    
    # mustard_bottle: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/mustard_bottle",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.57 - 0.15, 0, 0.55],
    #         rot=[0, 0, 0.7071, -0.7071],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    #         scale=(0.8, 0.8, 0.8),
    #     ),
    # )
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",  # will be set by agent env cfg
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.05, 0.06), pos_y=(0.5, 0.6), pos_z=(0.2, 0.3), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=2.0)

    # close_gripper_smooth = RewTerm(
    #     func=mdp.close_gripper_when_near_object_smooth,
    #     weight=15.0,
    #     params={
    #         "threshold": 0.035,  # 平滑过渡的距离尺度
    #         "open_joint_pos": 0.08,
    #         "object_cfg": SceneEntityCfg("cracker_box"),
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "robot_cfg": SceneEntityCfg("robot", joint_ids=[7, 8]),
    #     },
    # )

    # 使用硬阈值版本
    close_gripper = RewTerm(
        func=mdp.close_gripper_when_near_object,
        weight=20.0,
        params={
            "threshold": 0.035,  # 5cm以内开始关闭夹爪
            "open_joint_pos": 0.08,  # 夹爪完全打开时的关节位置
            "object_cfg": SceneEntityCfg("cracker_box"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot", joint_ids=[7, 8]),  # 假设7,8是夹爪关节
        },
    )


    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.60}, weight=1.0)


    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "release_distance": 0.18, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "release_distance": 0.18, "command_name": "object_pose"},
        weight=5.0,
    )


    open_gripper_near_goal = RewTerm(
        func=mdp.open_gripper_near_goal,
        weight=10.0,  # 调整权重
        params={
            "distance_threshold": 0.18,  # 5cm内开始松手
            "std": 0.1,
            "command_name": "object_pose",
            "gripper_close_threshold": 0.03,
            "max_ee_object_distance": 0.04,
            "open_joint_pos": 0.08,
        }
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("PickPlace", joint_names=["drawer_top_joint"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("PickPlace", joint_names=["drawer_top_joint"])},
    # )

    # # 4. Penalize actions for cosmetic reasons
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("cracker_box")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )

    object_goal_tracking_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_goal_tracking", "weight": 60, "num_steps": 10000}
    )

    object_goal_tracking_fine_grained_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "object_goal_tracking_fine_grained", "weight": 60, "num_steps": 10000}
    )

    open_gripper_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "open_gripper_near_goal", "weight": 60000, "num_steps": 150000}
    )
##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the PickPlace environment."""

    # Scene settings
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
