# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import random

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCapsule, VisualSphere, DynamicCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.storage.native import get_assets_root_path


class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path()

        # Use correct sortbot housing path + add table for totes
        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Props/Sortbot_Housing/sortbot_housing.usd"
        )
        self.tote_table_usd = (
            self.assets_root_path + "/Isaac/Props/Mounts/ThorlabsTable/table_instanceable.usd"
        )
        # self.ur10_robot_usd = (
        #     self.assets_root_path + "/Isaac/Samples/Cortex/UR10/Basic/cortex_ur10_basic.usd"
        # )
        self.ur10_robot_usd = (
            self.assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        )
        # Tote containers (KLT bins used as totes)
        self.tote_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        
        # SKU objects to be picked and placed (cracker boxes) 
        self.sku_usd = self.assets_root_path + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.rubiks_cube_usd = self.assets_root_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"


def print_diagnostics(diagnostic):
    print("=========== logical state ==========")
    if diagnostic.bin_name:
        print("active bin info:")
        print("- bin_obj.name: {}".format(diagnostic.bin_name))
        print("- bin_base: {}".format(diagnostic.bin_base))
        print("- grasp_T:\n{}".format(diagnostic.grasp))
        print("- is_grasp_reached: {}".format(diagnostic.grasp_reached))
        print("- is_attached:  {}".format(diagnostic.attached))
        print("- needs_flip:  {}".format(diagnostic.needs_flip))
    else:
        print("<no active bin>")

    print("------------------------------------")


def random_sku_spawn_position():
    # Spawn in source tote for YCB cracker box (~16x21x7cm)
    # Source tote center: (0.225, 1.25, -0.31), scaled to 0.6x0.4x0.3m
    tote_center_x = 0.225
    tote_center_y = 1.25  
    tote_center_z = -0.31
    
    # Spawn within tote bounds (adjusted for cracker box dimensions)
    x = tote_center_x + random.uniform(-0.15, 0.15)  # ±15cm within 60cm tote
    y = tote_center_y + random.uniform(-0.08, 0.08)  # ±8cm within 40cm tote  
    z = tote_center_z + 0.25  # Above tote bottom (cracker box height ~7cm)
    return np.array([x, y, z])


class SimplePickPlaceTask(BaseTask):
    def __init__(self, env_path, assets):
        super().__init__("simple_pick_place")
        self.assets = assets
        self.env_path = "/World/Ur10Table"
        self.skus = []
        self.current_sku = None
        self.max_skus = 1  # Only one SKU at a time for simple behavior
        
    def post_reset(self) -> None:
        # Clean up existing SKUs
        # Clean up USD objects (XFormPrim doesn't require scene.remove_object)
        if len(self.skus) > 0:
            for sku in self.skus:
                try:
                    # USD objects will be cleaned up automatically on scene reset
                    pass
                except:
                    pass
            self.skus.clear()
        self.current_sku = None
        
    def pre_step(self, time_step_index, simulation_time) -> None:
        """Spawn a single SKU in source tote when needed"""
        # Only spawn if we don't have an active SKU
        if self.current_sku is None and len(self.skus) < self.max_skus:
            self._spawn_sku()
            
    def _spawn_sku(self):
        """Spawn a YCB cracker box using proper Isaac Sim USD + physics pattern"""
        name = f"sku_{len(self.skus)}"
        prim_path = f"{self.env_path}/skus/{name}"
        
        # Step 1: Add YCB cracker box USD to stage (like Cortex pattern)
        add_reference_to_stage(usd_path=self.assets.sku_usd, prim_path=prim_path)
        
        # Step 2: Create XFormPrim with position (following working examples pattern)
        position = random_sku_spawn_position()
        sku_xform = XFormPrim(prim_path, positions=np.array([position]))
        
        # Store reference for position tracking
        self.current_sku = sku_xform
        self.skus.append(sku_xform)
        print(f"Spawned YCB cracker box {name} at position {position}")
        
    def get_current_sku_position(self):
        """Get position of current SKU for pick-and-place controller"""
        if self.current_sku is not None:
            # XFormPrim uses get_world_poses() (plural) for position tracking
            positions, orientations = self.current_sku.get_world_poses()
            return positions[0]  # Return first (and only) position
        return None


def main():
    # Use standard World instead of CortexWorld
    my_world = World(stage_units_in_meters=1.0)

    env_path = "/World/Ur10Table"
    ur10_assets = Ur10Assets()
    
    # Load robot first as the main component
    add_reference_to_stage(usd_path=ur10_assets.ur10_robot_usd, prim_path=f"{env_path}/ur10")
    
    # Select surface gripper variant on the UR10
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(f"{env_path}/ur10")
    if robot_prim:
        variant_set = robot_prim.GetVariantSet("Gripper")
        if variant_set:
            variant_set.SetVariantSelection("Short_Suction")
            print("Surface gripper variant selected")
    
    # Position the robot
    robot_xform = XFormPrim(
        f"{env_path}/ur10",
        positions=np.array([[0.02, 0.28, 0.03]]),  # Robot position - adjust as needed
        orientations=np.array([[1, 0, 0, 0]]),  # Robot orientation
    )
    
    # Create SurfaceGripper and SingleManipulator (following ur10_pick_up.py pattern)
    gripper = SurfaceGripper(
        end_effector_prim_path=f"{env_path}/ur10/ee_link", 
        surface_gripper_path=f"{env_path}/ur10/ee_link/SurfaceGripper"
    )
    
    ur10_robot = my_world.scene.add(
        SingleManipulator(
            prim_path=f"{env_path}/ur10", 
            name="my_ur10", 
            end_effector_prim_path=f"{env_path}/ur10/ee_link", 
            gripper=gripper
        )
    )
    
    # Set default joint positions (upright pose)
    ur10_robot.set_joints_default_state(
        positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
    )
    ur10_robot.gripper.set_default_state(opened=True)
    
    # Load table housing as a child component
    add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=f"{env_path}/sortbot_housing")
    table_prim = XFormPrim(
        f"{env_path}/sortbot_housing",
        positions=np.array([[0.0, 0.0, -1.15]]),  # Table at origin
        orientations=np.array([[1, 0, 0, 0]]),  # Default orientation
    )
    
    # Add separate table for totes
    add_reference_to_stage(usd_path=ur10_assets.tote_table_usd, prim_path=f"{env_path}/tote_table")
    tote_table_prim = XFormPrim(
        f"{env_path}/tote_table",
        positions=np.array([[0.55, 1.25, -0.48]]),  # Position for totes
        orientations=np.array([[0, 0, 0, 1]]),  # Default orientation
        scales=np.array([[2.0, 1.0, 0.8]]),  # Scale 2x along X-axis
    )
    
    add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
    background_prim = XFormPrim(
        "/World/Background",
        positions=np.array([[10.00, 2.00, -1.18180]]),
        orientations=np.array([[0.7071, 0, 0, 0.7071]]),
    )
    
    # Add source and target totes for tote-to-tote transfer (on the tote table)
    add_reference_to_stage(usd_path=ur10_assets.tote_usd, prim_path="/World/SourceTote")
    source_tote = XFormPrim(
        "/World/SourceTote",
        positions=np.array([[0.225, 1.25, -0.31]]),  # On the tote table (left side)
        orientations=np.array([[0, 0, 0, 1]]),
        scales=np.array([[2.14, 2.22, 2.31]]),  # Scale to 0.6x0.4x0.3m
    )
    
    add_reference_to_stage(usd_path=ur10_assets.tote_usd, prim_path="/World/TargetTote") 
    target_tote = XFormPrim(
        "/World/TargetTote",
        positions=np.array([[-0.225, 1.25, -0.31]]),  # On the tote table (right side)
        orientations=np.array([[0, 0, 0, 1]]),
        scales=np.array([[2.14, 2.22, 2.31]]),  # Scale to 0.6x0.4x0.3m
    )
    # ========================================================

    # Skip complex obstacle registration for simple demo

    # Add simple pick-place task
    my_task = SimplePickPlaceTask(env_path, ur10_assets)
    my_world.add_task(my_task)
    my_world.reset()
    
    # Create PickPlaceController (following ur10_pick_up.py pattern)
    my_controller = PickPlaceController(
        name="pick_place_controller", 
        gripper=ur10_robot.gripper, 
        robot_articulation=ur10_robot
    )
    articulation_controller = ur10_robot.get_articulation_controller()
    
    # Target position for placing (target tote center, adjusted for cracker box height)
    target_position = np.array([-0.225, 1.25, -0.15])  # Target tote position (higher for box)
    
    print("=== Simple Pick-and-Place Demo Started ===")
    print("Robot will pick YCB cracker boxes from source tote and place in target tote")
    
    reset_needed = False
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False
                
            # Get current SKU position from task
            sku_position = my_task.get_current_sku_position()
            
            if sku_position is not None:
                # Execute pick-and-place using controller
                actions = my_controller.forward(
                    picking_position=sku_position,
                    placing_position=target_position,
                    current_joint_positions=ur10_robot.get_joint_positions(),
                    end_effector_offset=np.array([0, 0, 0.04]),  # Higher offset for cracker box
                )
                
                if my_controller.is_done():
                    print("✅ Pick-and-place cycle completed!")
                    # Reset for next cycle
                    my_task.current_sku = None
                    
                articulation_controller.apply_action(actions)
    
    simulation_app.close()


if __name__ == "__main__":
    main()