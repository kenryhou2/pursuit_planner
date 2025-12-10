#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose2D, Pose
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelState
import os
import yaml
import math
import threading


class GazeboPursuitViz(object):
    class TrajectoryTrack(object):
        """
        Simple piecewise-linear trajectory track for grid poses (gx, gy, yaw).

        - Keeps the last pose and the current interpolation segment.
        - New waypoints become segment endpoints.
        - Interpolation speed is controlled via segment_duration and replay_speed.
        """

        def __init__(self, name, z, parent):
            self.name = name
            self.z = z
            self.parent = parent  # GazeboPursuitViz instance (for params & set_model_state)

            self.has_pose = False
            # self.start_x = rospy.get_param("~start_x",rospy.get_param("/eigenbot_start_x", 0.0))
            # self.start_y = rospy.get_param("~start_y",rospy.get_param("/eigenbot_start_y", 0.0))
            # self.start_z = rospy.get_param("~start_z",rospy.get_param("/eigenbot_start_z", 0.0))
            # self.start_yaw = rospy.get_param("~start_yaw",rospy.get_param("/eigenbot_start_yaw", 0.0))
            # rospy.loginfo(f"[Viz Init] Using start pose x={self.start_x}, "
            # f"y={self.start_y}, z={self.start_z}, yaw={self.start_yaw}"
        )
            # Last "committed" pose (after previous segment).
            self.last_gx = 0.0
            self.last_gy = 0.0
            self.last_yaw = 0.0

            # Active segment: start → goal
            self.seg_start_time = None
            self.seg_duration = None
            self.seg_start_gx = 0.0
            self.seg_start_gy = 0.0
            self.seg_start_yaw = 0.0
            self.seg_goal_gx = 0.0
            self.seg_goal_gy = 0.0
            self.seg_goal_yaw = 0.0

        @staticmethod
        def _interp_angle(a0, a1, s):
            """
            Interpolate angle along shortest arc.
            """
            def wrap(a):
                return (a + math.pi) % (2.0 * math.pi) - math.pi

            da = wrap(a1 - a0)
            return wrap(a0 + s * da)

        def _current_pose(self, now):
            """
            Compute the interpolated pose at 'now' without mutating state.
            """
            if not self.has_pose:
                return None

            if self.seg_start_time is None or self.seg_duration is None or self.seg_duration <= 0.0:
                # No active segment, just hold last pose.
                return self.last_gx, self.last_gy, self.last_yaw

            dt = (now - self.seg_start_time).to_sec()
            if dt <= 0.0:
                s = 0.0
            else:
                s = dt / self.seg_duration

            if s >= 1.0:
                # Segment finished → exactly at goal pose.
                return self.seg_goal_gx, self.seg_goal_gy, self.seg_goal_yaw

            # Interpolate
            gx = self.seg_start_gx + s * (self.seg_goal_gx - self.seg_start_gx)
            gy = self.seg_start_gy + s * (self.seg_goal_gy - self.seg_start_gy)
            yaw = self._interp_angle(self.seg_start_yaw, self.seg_goal_yaw, s)
            return gx, gy, yaw

        def set_waypoint(self, gx, gy, yaw, now):
            """
            Add a new waypoint (gx, gy, yaw) to the track.

            - On first waypoint: snap to pose, no interpolation.
            - Otherwise: start a new interpolation segment from the
              *current interpolated pose* to the new waypoint.
            """
            if yaw is None:
                yaw = 0.0

            if not self.has_pose:
                # First pose: no interpolation, just store & show.
                self.has_pose = True
                self.last_gx = gx
                self.last_gy = gy
                self.last_yaw = yaw

                # Immediately push to Gazebo once to avoid delay.
                self.parent.set_model_state(self.name, gx, gy, self.z, yaw=yaw)
                return

            # Use current interpolated pose as new segment start
            cur = self._current_pose(now)
            if cur is None:
                cur_gx, cur_gy, cur_yaw = gx, gy, yaw
            else:
                cur_gx, cur_gy, cur_yaw = cur

            self.seg_start_gx = cur_gx
            self.seg_start_gy = cur_gy
            self.seg_start_yaw = cur_yaw

            self.seg_goal_gx = gx
            self.seg_goal_gy = gy
            self.seg_goal_yaw = yaw

            self.seg_start_time = now

            # Replay speed: >1.0 → faster, <1.0 → slower
            base = self.parent.segment_duration
            speed = max(self.parent.replay_speed, 1e-3)
            self.seg_duration = base / speed

        def update_and_apply(self, now):
            """
            Called from timer: compute current pose & send to Gazebo.
            """
            if not self.has_pose:
                return

            gx, gy, yaw = self._current_pose(now)

            # If the segment finished, "commit" last pose so the next
            # segment starts from a clean state.
            if (
                self.seg_start_time is not None
                and self.seg_duration is not None
                and (now - self.seg_start_time).to_sec() >= self.seg_duration
            ):
                self.last_gx = self.seg_goal_gx
                self.last_gy = self.seg_goal_gy
                self.last_yaw = self.seg_goal_yaw
                # Keep seg_* in case new arrivals use current pose.

            # Send pose to Gazebo
            self.parent.set_model_state(self.name, gx, gy, self.z, yaw=yaw)

    # ==============================================================
    # GazeboPursuitViz
    # ==============================================================
    def __init__(self):
        # ==============================
        # Map/grid → world parameters
        # ==============================
        self.resolution = rospy.get_param("~resolution", 1.0)  # meters per cell
        self.origin_x   = rospy.get_param("~origin_x", 0.0)
        self.origin_y   = rospy.get_param("~origin_y", 0.0)

        # Height of obstacle "walls"
        self.wall_height = rospy.get_param("~wall_height", 2.0)

        # Robot/target vertical position
        self.agent_height = rospy.get_param("~agent_height", 0.2)

        # Names of robot/target models in Gazebo
        self.robot_name  = rospy.get_param("~robot_name",  "pursuit_robot")
        self.target_name = rospy.get_param("~target_name", "pursuit_target")

        # Topics to subscribe for robot/target (Pose2D)
        self.robot_topic  = rospy.get_param("~robot_topic",  "robot_pose")
        self.target_topic = rospy.get_param("~target_topic", "target_pose")

        # ==============================
        # Trajectory / visualization parameters
        # ==============================
        self.update_rate = rospy.get_param("~update_rate", 30.0)           # Hz
        self.segment_duration = rospy.get_param("~segment_duration", 0.2)  # sec at replay_speed=1
        self.replay_speed = rospy.get_param("~replay_speed", 1.0)          # >1 faster, <1 slower

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

        # Tracks (for interpolation)
        self.robot_track = None
        self.target_track = None

        # Obstacle tracks + lock for thread-safe access
        self.obstacle_tracks = {}  # oid -> TrajectoryTrack
        self._obstacle_tracks_lock = threading.Lock()

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

        # ==============================
        # Timer for smooth updates
        # ==============================
        if self.update_rate > 0.0:
            self.timer = rospy.Timer(
                rospy.Duration(1.0 / self.update_rate),
                self.update_timer_cb
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

        # Create trajectory track for this obstacle
        with self._obstacle_tracks_lock:
            if oid not in self.obstacle_tracks:
                self.obstacle_tracks[oid] = self.TrajectoryTrack(oid, self.wall_height, self)

    def ensure_robot_model(self):
        if self.robot_name in self.spawned_models:
            return
        sdf_xml = self.make_agent_sdf(self.robot_name, color="0 1 0 1")  # green
        self.spawn_model(self.robot_name, sdf_xml, z=self.agent_height, desc="robot")
        self.robot_track = self.TrajectoryTrack(self.robot_name, self.agent_height, self)

    def ensure_target_model(self):
        if self.target_name in self.spawned_models:
            return
        sdf_xml = self.make_agent_sdf(self.target_name, color="1 1 0 1")  # yellow
        self.spawn_model(self.target_name, sdf_xml, z=self.agent_height, desc="target")
        self.target_track = self.TrajectoryTrack(self.target_name, self.agent_height, self)

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
        Tall, skinny cylinder used for robot/target.
        Color is 'r g b a'.
        """
        radius = 1 * self.resolution
        height = self.wall_height

        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>2.0</mass>
        <inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{r}</radius>
            <length>{h}</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{r}</radius>
            <length>{h}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{c}</ambient>
          <diffuse>{c}</diffuse>
        </material>
      </visual>

    </link>
  </model>
</sdf>""".format(name=name, r=radius, h=height, c=color)

    # ------------------------------------------------
    # Callbacks
    # ------------------------------------------------
    def obstacle_cb(self, msg, oid):
        # Ensure model & track exist
        if oid not in self.spawned_models:
            self.ensure_obstacle_model(oid)

        gx, gy = msg.x, -msg.y
        yaw = 0.0  # obstacles don't use theta right now
        now = rospy.Time.now()

        with self._obstacle_tracks_lock:
            track = self.obstacle_tracks.get(oid)
            if track is None:
                self.obstacle_tracks[oid] = self.TrajectoryTrack(oid, self.wall_height, self)
                track = self.obstacle_tracks[oid]

        # Use track outside the lock
        track.set_waypoint(gx, gy, yaw, now)

    def robot_cb(self, msg):
        if self.robot_name not in self.spawned_models:
            self.ensure_robot_model()
        gx, gy, yaw = msg.x, -msg.y, msg.theta
        now = rospy.Time.now()
        if self.robot_track is None:
            self.robot_track = self.TrajectoryTrack(self.robot_name, self.agent_height, self)
        self.robot_track.set_waypoint(gx, gy, yaw, now)

    def target_cb(self, msg):
        if self.target_name not in self.spawned_models:
            self.ensure_target_model()
        gx, gy, yaw = msg.x, -msg.y, msg.theta
        now = rospy.Time.now()
        if self.target_track is None:
            self.target_track = self.TrajectoryTrack(self.target_name, self.agent_height, self)
        self.target_track.set_waypoint(gx, gy, yaw, now)

    # ------------------------------------------------
    # Timer callback: smooth rollout of trajectories
    # ------------------------------------------------
    def update_timer_cb(self, event):
        now = rospy.Time.now()

        if self.robot_track is not None:
            self.robot_track.update_and_apply(now)

        if self.target_track is not None:
            self.target_track.update_and_apply(now)

        with self._obstacle_tracks_lock:
            tracks = list(self.obstacle_tracks.values())

        for track in tracks:
            track.update_and_apply(now)


def main():
    rospy.init_node("gazebo_pursuit_viz")
    GazeboPursuitViz()
    rospy.spin()


if __name__ == "__main__":
    main()
