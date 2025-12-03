#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose2D, Pose
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelState
import os
import yaml


class GazeboPursuitViz(object):
    def __init__(self):
        # ==============================
        # Map/grid → world parameters
        # ==============================
        self.resolution = rospy.get_param("~resolution", 1.0)  # meters per cell
        self.origin_x   = rospy.get_param("~origin_x", 0.0)
        self.origin_y   = rospy.get_param("~origin_y", 0.0)

        # Height of obstacle "walls"
        self.wall_height = rospy.get_param("~wall_height", 2.0)

        # Robot/target vertical position (lower so they sit on ground)
        self.agent_height = rospy.get_param("~agent_height", 0.2)

        # Names of robot/target models in Gazebo
        self.robot_name  = rospy.get_param("~robot_name",  "pursuit_robot")
        self.target_name = rospy.get_param("~target_name", "pursuit_target")

        # Topics to subscribe for robot/target (Pose2D)
        # In your launch you can set these to match the planner ns, e.g.
        #   robot_topic:  "/pursuit_planner/robot_pose"
        #   target_topic: "/pursuit_planner/target_pose"
        self.robot_topic  = rospy.get_param("~robot_topic",  "robot_pose")
        self.target_topic = rospy.get_param("~target_topic", "target_pose")

        # ==============================
        # Optional: load YAML for dynamic obstacles
        # ==============================
        dyn_yaml_path = rospy.get_param("~dyno_yaml", "")
        self.obstacle_defs = {}
        if dyn_yaml_path and os.path.isfile(dyn_yaml_path):
            rospy.loginfo("Loading dynamic obstacle definitions from %s", dyn_yaml_path)
            with open(dyn_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            for entry in data.get("dynamic_obstacles", []):
                self.obstacle_defs[entry["id"]] = entry

        # ==============================
        # Gazebo services
        # ==============================
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # Track what we've spawned already
        self.spawned_models = set()

        # ==============================
        # Dynamic obstacle subscribers
        # ==============================
        self.subscribers = {}
        obstacle_ids = rospy.get_param("~obstacle_ids", [])
        if not obstacle_ids and self.obstacle_defs:
            obstacle_ids = list(self.obstacle_defs.keys())

        if obstacle_ids:
            for oid in obstacle_ids:
                topic = "/dynamic_obstacles/{}/pose".format(oid)
                rospy.loginfo("Subscribing to dynamic obstacle topic %s", topic)
                self.subscribers[oid] = rospy.Subscriber(
                    topic, Pose2D, self.obstacle_cb, callback_args=oid, queue_size=1
                )
        else:
            rospy.logwarn("No obstacle_ids provided and no dyno_yaml, no dynamic obstacles will be shown.")

        # ==============================
        # Robot + target subscribers
        # ==============================
        rospy.loginfo("Subscribing to robot topic:  %s", self.robot_topic)
        rospy.loginfo("Subscribing to target topic: %s", self.target_topic)

        self.robot_sub = rospy.Subscriber(
            self.robot_topic, Pose2D, self.robot_cb, queue_size=1
        )
        self.target_sub = rospy.Subscriber(
            self.target_topic, Pose2D, self.target_cb, queue_size=1
        )

    # ------------------------------------------------
    # Generic helpers
    # ------------------------------------------------
    def grid_to_world(self, gx, gy, z):
        """
        Convert 1-based grid coordinates (gx, gy) to world coordinates.
        The planner uses 1..x_size, 1..y_size; we place models at
        the center of each grid cell.
        """
        wx = self.origin_x + (gx - 0.5) * self.resolution
        wy = self.origin_y + (gy - 0.5) * self.resolution
        return wx, wy, z

    def set_model_state(self, model_name, gx, gy, z, yaw=None):
        wx, wy, wz = self.grid_to_world(gx, gy, z)

        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = wx
        state.pose.position.y = wy
        state.pose.position.z = wz

        # Simple yaw-only rotation from Pose2D.theta if desired
        if yaw is not None:
            import tf.transformations as tft
            q = tft.quaternion_from_euler(0.0, 0.0, yaw)
            state.pose.orientation.x = q[0]
            state.pose.orientation.y = q[1]
            state.pose.orientation.z = q[2]
            state.pose.orientation.w = q[3]
        else:
            state.pose.orientation.w = 1.0

        req = SetModelStateRequest()
        req.model_state = state

        try:
            self.set_state_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to set model state for %s: %s", model_name, str(e))

    def initial_pose_msg(self, z):
        p = Pose()
        p.position.z = z
        p.orientation.w = 1.0
        return p

    # ------------------------------------------------
    # Spawning models
    # ------------------------------------------------
    def ensure_obstacle_model(self, oid):
        if oid in self.spawned_models:
            return

        entry = self.obstacle_defs.get(oid, {})
        footprint = entry.get("footprint", {})
        kind = footprint.get("kind", "box")

        if kind == "box":
            sdf_xml = self.make_box_sdf(oid, footprint)
        else:
            # "circle" or anything else → cylinder
            sdf_xml = self.make_cylinder_sdf(oid, footprint)

        self.spawn_model(oid, sdf_xml, z=self.wall_height, desc="obstacle (%s)" % kind)

    def ensure_robot_model(self):
        if self.robot_name in self.spawned_models:
            return
        sdf_xml = self.make_agent_sdf(self.robot_name, color="0 1 0 1")  # green
        self.spawn_model(self.robot_name, sdf_xml, z=self.agent_height, desc="robot")

    def ensure_target_model(self):
        if self.target_name in self.spawned_models:
            return
        sdf_xml = self.make_agent_sdf(self.target_name, color="1 1 0 1")  # yellow
        self.spawn_model(self.target_name, sdf_xml, z=self.agent_height, desc="target")

    def spawn_model(self, name, sdf_xml, z, desc="model"):
        try:
            self.spawn_srv(
                model_name=name,
                model_xml=sdf_xml,
                robot_namespace=name,
                initial_pose=self.initial_pose_msg(z),
                reference_frame="world"
            )
            self.spawned_models.add(name)
            rospy.loginfo("Spawned %s '%s'", desc, name)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to spawn %s %s: %s", desc, name, str(e))

    # ------------------------------------------------
    # SDF templates
    # ------------------------------------------------
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

    def make_agent_sdf(self, name, color="0 1 0 1"):
        """
        Simple small cylinder used for robot/target. Color is 'r g b a'.
        """
        radius = 0.3 * self.resolution
        length = 0.4  # height

        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
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
          <ambient>{c}</ambient>
          <diffuse>{c}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=name, r=radius, l=length, c=color)

    # ------------------------------------------------
    # Callbacks
    # ------------------------------------------------
    def obstacle_cb(self, msg, oid):
        # Ensure model exists
        if oid not in self.spawned_models:
            self.ensure_obstacle_model(oid)

        gx, gy = msg.x, msg.y
        # Obstacles are rendered as tall walls (z = wall_height)
        self.set_model_state(oid, gx, gy, self.wall_height)

    def robot_cb(self, msg):
        if self.robot_name not in self.spawned_models:
            self.ensure_robot_model()

        gx, gy, yaw = msg.x, msg.y, msg.theta
        self.set_model_state(self.robot_name, gx, gy, self.agent_height, yaw=yaw)

    def target_cb(self, msg):
        if self.target_name not in self.spawned_models:
            self.ensure_target_model()

        gx, gy, yaw = msg.x, msg.y, msg.theta
        self.set_model_state(self.target_name, gx, gy, self.agent_height, yaw=yaw)


def main():
    rospy.init_node("gazebo_pursuit_viz")
    GazeboPursuitViz()
    rospy.spin()


if __name__ == "__main__":
    main()
