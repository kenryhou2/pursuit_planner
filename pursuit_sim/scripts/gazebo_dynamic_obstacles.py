#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose2D
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelState
import os
import yaml

class GazeboDynamicObstacles(object):
    def __init__(self):
        # Map/grid → world parameters
        self.resolution = rospy.get_param("~resolution", 1.0)  # meters per cell
        self.origin_x   = rospy.get_param("~origin_x", 0.0)
        self.origin_y   = rospy.get_param("~origin_y", 0.0)
        self.wall_height = rospy.get_param("~wall_height", 2.0)

        # Optional: load YAML to get footprints (circle/box sizes)
        dyn_yaml_path = rospy.get_param("~dyno_yaml", "")
        self.obstacle_defs = {}
        if dyn_yaml_path and os.path.isfile(dyn_yaml_path):
            rospy.loginfo("Loading dynamic obstacle definitions from %s", dyn_yaml_path)
            with open(dyn_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            for entry in data.get("dynamic_obstacles", []):
                self.obstacle_defs[entry["id"]] = entry

        # Wait for Gazebo services
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # Spawn models & set up subscribers
        self.subscribers = {}
        self.spawned_models = set()

        # obstacle IDs can come from YAML or param
        obstacle_ids = rospy.get_param("~obstacle_ids", [])
        if not obstacle_ids and self.obstacle_defs:
            obstacle_ids = list(self.obstacle_defs.keys())

        if not obstacle_ids:
            rospy.logwarn("No obstacle_ids provided and no dyno_yaml, nothing to track.")
            return

        for oid in obstacle_ids:
            topic = "/dynamic_obstacles/{}/pose".format(oid)
            rospy.loginfo("Subscribing to %s", topic)
            self.subscribers[oid] = rospy.Subscriber(
                topic, Pose2D, self.pose_cb, callback_args=oid, queue_size=1
            )

    def pose_cb(self, msg, oid):
        # Ensure model is spawned
        if oid not in self.spawned_models:
            self.spawn_obstacle_model(oid)

        # Convert grid coords → world coords
        gx, gy = msg.x, msg.y
        wx = self.origin_x + (gx - 0.5) * self.resolution
        wy = self.origin_y + (gy - 0.5) * self.resolution
        wz = self.wall_height

        state = ModelState()
        state.model_name = oid
        state.pose.position.x = wx
        state.pose.position.y = wy
        state.pose.position.z = wz
        state.pose.orientation.w = 1.0  # no rotation for now

        req = SetModelStateRequest()
        req.model_state = state

        try:
            self.set_state_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to set model state for %s: %s", oid, str(e))

    def spawn_obstacle_model(self, oid):
        # Decide shape from YAML if available
        entry = self.obstacle_defs.get(oid, {})
        footprint = entry.get("footprint", {})
        kind = footprint.get("kind", "box")

        # Choose one of two simple SDF templates (box / cylinder)
        sdf_template = self.make_box_sdf(oid, footprint) if kind == "box" else self.make_cylinder_sdf(oid, footprint)

        try:
            self.spawn_srv(
                model_name=oid,
                model_xml=sdf_template,
                robot_namespace=oid,
                initial_pose=self.initial_pose_msg(),
                reference_frame="world"
            )
            self.spawned_models.add(oid)
            rospy.loginfo("Spawned obstacle model '%s' (%s)", oid, kind)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to spawn model %s: %s", oid, str(e))

    def initial_pose_msg(self):
        from geometry_msgs.msg import Pose
        p = Pose()
        p.position.z = self.wall_height
        p.orientation.w = 1.0
        return p

    def make_box_sdf(self, oid, footprint):
        width  = float(footprint.get("width", 10))
        height = float(footprint.get("height", 10))
        # Treat grid cells as meters here; you can scale by self.resolution if desired
        size_x = width * self.resolution
        size_y = height * self.resolution
        size_z = self.wall_height

        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>10.0</mass>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=oid, sx=size_x, sy=size_y, sz=size_z)

    def make_cylinder_sdf(self, oid, footprint):
        radius = float(footprint.get("radius", 5.0)) * self.resolution
        length = self.wall_height

        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>10.0</mass>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{r}</radius>
            <length>{l}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{r}</radius>
            <length>{l}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=oid, r=radius, l=length)

def main():
    rospy.init_node("gazebo_dynamic_obstacles")
    GazeboDynamicObstacles()
    rospy.spin()

if __name__ == "__main__":
    main()
