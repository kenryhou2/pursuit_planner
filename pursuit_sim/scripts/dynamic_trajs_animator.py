#!/usr/bin/env python3
import os
import csv
import rospy
import rospkg
import yaml

from gazebo_msgs.srv import SetModelState, SpawnModel, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty


class DynamicTrajectoryReplay(object):
    def __init__(self):
        rospy.init_node("dynamic_trajectory_animator")

        rospack = rospkg.RosPack()
        sim_pkg_path = rospack.get_path("pursuit_sim")
        planner_pkg_path = rospack.get_path("pursuit_planner")

        # Trajectory file (CSV)
        default_traj_file = os.path.join(
            sim_pkg_path, "map_trajectories", "map3", "dynamic_trajectories.txt"
        )
        self.traj_file = rospy.get_param("~trajectory_file", default_traj_file)

        # YAML with obstacle footprints (your dyno_map3.yaml)
        default_dyno_yaml = os.path.join(
            planner_pkg_path, "config", "dyno_map3.yaml"
        )
        self.dyno_yaml = rospy.get_param("~dyno_yaml", default_dyno_yaml)

        # === Mapping: planner (gx, gy) -> world (x, y) ===
        # Must match generate_heightmap.py:
        #   x_world = (gx + 0.5) * meters_per_cell
        #   y_world = -(gy + 0.5) * meters_per_cell
        #
        # We keep the old param name "resolution" for convenience, but treat it as meters_per_cell.
        self.meters_per_cell = rospy.get_param(
            "~meters_per_cell",
            rospy.get_param("~resolution", 1.0)
        )

        # Vertical placement and height of obstacles
        self.z_base          = rospy.get_param("~z_height", 0.0)
        self.obstacle_height = rospy.get_param("~obstacle_height", 5.0)

        # Robot visualization parameters
        self.robot_name   = rospy.get_param("~robot_model_name", "robot")
        # Sphere radius in *world meters* (1m diameter)
        self.robot_radius = rospy.get_param("~robot_radius", 0.5)

        # Interpolation update rate (Hz)
        self.interp_rate = rospy.get_param("~interp_rate", 30.0)

        # Replay speed: >1 = faster, <1 = slower
        self.replay_speed = rospy.get_param("~replay_speed", 1.0)
        if self.replay_speed <= 0.0:
            rospy.logwarn("replay_speed must be > 0. Using 1.0 instead.")
            self.replay_speed = 1.0

        self.loop = rospy.get_param("~loop", False)

        rospy.loginfo("DynamicTrajectoryReplay (Python3)")
        rospy.loginfo("Trajectory file: %s", self.traj_file)
        rospy.loginfo("Dyno YAML: %s", self.dyno_yaml)
        rospy.loginfo("meters_per_cell (resolution) = %.4f", self.meters_per_cell)
        rospy.loginfo("interp_rate = %.1f Hz", self.interp_rate)
        rospy.loginfo("replay_speed = %.2f x", self.replay_speed)

        # Data containers
        self.time_stamps    = []
        self.robot_traj     = []      # list of (gx, gy)
        self.obstacle_names = []      # ["obs0", "obs1", ...]
        self.obstacle_data  = {}      # name -> [(gx,gy), ...]
        self.footprints     = {}      # name -> footprint dict

        # Load trajectory + footprints
        self._load_csv()
        self._load_footprints()

        # Gazebo services
        rospy.loginfo("Waiting for /gazebo/set_model_state ...")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.loginfo("Waiting for /gazebo/spawn_sdf_model ...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Waiting for /gazebo/get_model_state ...")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        # Optional: unpause physics
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)
            unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
            unpause()
        except (rospy.ROSException, rospy.ServiceException):
            rospy.logwarn("Could not unpause Gazebo physics.")

        # Spawn geometry-matching obstacle models + robot if needed
        self._ensure_models_exist()

    # ==========================================================
    # LOAD TRAJECTORIES
    # ==========================================================
    def _load_csv(self):
        if not os.path.exists(self.traj_file):
            rospy.logerr("Trajectory file does not exist: %s", self.traj_file)
            rospy.signal_shutdown("Missing trajectory file")
            return

        with open(self.traj_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Expect: time,robot_x,robot_y,obs0_x,obs0_y,obs1_x,obs1_y,...
            if len(header) < 5:
                rospy.logerr("CSV header too short.")
                rospy.signal_shutdown("Bad trajectory header")
                return

            # Map obstacle columns
            obs_cols = {}
            i = 3
            while i + 1 < len(header):
                x_name = header[i]
                y_name = header[i + 1]

                if x_name.endswith("_x") and y_name.endswith("_y"):
                    base_x = x_name[:-2]
                    base_y = y_name[:-2]
                    if base_x == base_y:
                        obs_name = base_x
                    else:
                        obs_name = f"obs{len(obs_cols)}"
                else:
                    obs_name = f"obs{len(obs_cols)}"

                obs_cols[obs_name] = (i, i + 1)
                i += 2

            self.obstacle_names = list(obs_cols.keys())
            rospy.loginfo("Detected obstacles from CSV: %s", self.obstacle_names)

            for name in self.obstacle_names:
                self.obstacle_data[name] = []

            for row in reader:
                if not row:
                    continue

                t = float(row[0])
                self.time_stamps.append(t)

                # Robot trajectory
                gx_r = float(row[1])
                gy_r = float(row[2])
                self.robot_traj.append((gx_r, gy_r))

                # Obstacles trajectory
                for name, (ix, iy) in obs_cols.items():
                    gx = float(row[ix])
                    gy = float(row[iy])
                    self.obstacle_data[name].append((gx, gy))

        rospy.loginfo("Loaded %d trajectory steps", len(self.time_stamps))
        if len(self.robot_traj) != len(self.time_stamps):
            rospy.logwarn("Robot trajectory length (%d) != time_stamps length (%d)",
                          len(self.robot_traj), len(self.time_stamps))

    # ==========================================================
    # LOAD FOOTPRINTS FROM dyno_map3.yaml
    # ==========================================================
    def _load_footprints(self):
        if not os.path.exists(self.dyno_yaml):
            rospy.logwarn("Dyno YAML not found: %s (using default cubes)", self.dyno_yaml)
            return

        with open(self.dyno_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        for entry in cfg.get("dynamic_obstacles", []):
            name = entry.get("id")
            fp   = entry.get("footprint", {})
            if not name or not fp:
                continue

            kind = fp.get("kind", "box")
            data = {"kind": kind}

            if kind == "circle":
                data["radius"] = float(fp.get("radius", 1.0))
            elif kind == "box":
                data["width"]  = float(fp.get("width", 1.0))
                data["height"] = float(fp.get("height", 1.0))

            self.footprints[name] = data

        rospy.loginfo("Loaded footprints for obstacles: %s", list(self.footprints.keys()))

    # ==========================================================
    # GRID → WORLD
    # ==========================================================
    def grid_to_world(self, gx, gy):
        """
        Match generate_heightmap.py mapping:

          x_world = (gx + 0.5) * meters_per_cell
          y_world = -(gy + 0.5) * meters_per_cell
        """
        m = self.meters_per_cell
        wx = (gx + 0.5) * m
        wy = -(gy + 0.5) * m
        return wx, wy

    # ==========================================================
    # SDF GENERATORS (circle / box / robot sphere)
    # ==========================================================
    def _make_circle_sdf(self, name, radius_cells):
        radius = radius_cells * self.meters_per_cell
        height = self.obstacle_height

        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
          <ixy>0.0</ixy><ixz>0.0</ixz><iyz>0.0</iyz>
        </inertia>
      </inertial>
     
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        return sdf

    def _make_box_sdf(self, name, width_cells, height_cells):
        size_x = width_cells  * self.meters_per_cell
        size_y = height_cells * self.meters_per_cell
        size_z = self.obstacle_height + 0.5  # small extra to avoid z-fighting

        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
          <ixy>0.0</ixy><ixz>0.0</ixz><iyz>0.0</iyz>
        </inertia>
      </inertial>
      
      <visual name="visual">
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        return sdf

    def _make_robot_sdf(self, name):
        """
        Green sphere robot, 1m diameter (radius = 0.5m by default).
        """
        r = self.robot_radius
        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
          <ixy>0.0</ixy><ixz>0.0</ixz><iyz>0.0</iyz>
        </inertia>
      </inertial>
     
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{r}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        return sdf

    # ==========================================================
    # ENSURE MODELS EXIST WITH CORRECT GEOMETRY
    # ==========================================================
    def _ensure_models_exist(self):
        if not self.time_stamps:
            rospy.logwarn("No trajectory loaded; cannot ensure models.")
            return

        # Obstacles
        for name in self.obstacle_names:
            try:
                resp = self.get_model_state(name, "world")
                if resp.success:
                    rospy.loginfo("Model '%s' already exists in Gazebo.", name)
                    continue
            except rospy.ServiceException:
                pass

            gx0, gy0 = self.obstacle_data[name][0]
            wx0, wy0 = self.grid_to_world(gx0, gy0)

            fp = self.footprints.get(name, {"kind": "box"})
            kind = fp.get("kind", "box")

            if kind == "circle":
                radius_cells = fp.get("radius", 1.0)
                sdf_xml = self._make_circle_sdf(name, radius_cells)
                rospy.loginfo("Spawning CIRCLE '%s' (radius_cells=%.1f) at (%.3f, %.3f)",
                              name, radius_cells, wx0, wy0)
            elif kind == "box":
                width_cells  = fp.get("width", 1.0)
                height_cells = fp.get("height", 1.0)
                sdf_xml = self._make_box_sdf(name, width_cells, height_cells)
                rospy.loginfo("Spawning BOX '%s' (w=%g, h=%g cells) at (%.3f, %.3f)",
                              name, width_cells, height_cells, wx0, wy0)
            else:
                sdf_xml = self._make_box_sdf(name, 1.0, 1.0)
                rospy.loginfo("Spawning DEFAULT BOX '%s' at (%.3f, %.3f)", name, wx0, wy0)

            pose = Pose()
            pose.position.x = wx0
            pose.position.y = wy0
            pose.position.z = self.z_base + self.obstacle_height / 2.0
            pose.orientation.w = 1.0

            try:
                self.spawn_model(
                    model_name=name,
                    model_xml=sdf_xml,
                    robot_namespace=name,
                    initial_pose=pose,
                    reference_frame="world",
                )
            except rospy.ServiceException as e:
                rospy.logerr("Failed to spawn model '%s': %s", name, str(e))

        # Robot
        if self.robot_traj:
            try:
                resp = self.get_model_state(self.robot_name, "world")
                if resp.success:
                    rospy.loginfo("Robot model '%s' already exists in Gazebo.", self.robot_name)
                    return
            except rospy.ServiceException:
                pass

            gx0, gy0 = self.robot_traj[0]
            wx0, wy0 = self.grid_to_world(gx0, gy0)

            sdf_xml = self._make_robot_sdf(self.robot_name)
            rospy.loginfo("Spawning ROBOT '%s' (radius=%.2f m) at (%.3f, %.3f)",
                          self.robot_name, self.robot_radius, wx0, wy0)

            pose = Pose()
            pose.position.x = wx0
            pose.position.y = wy0
            pose.position.z = self.z_base + self.robot_radius
            pose.orientation.w = 1.0

            try:
                self.spawn_model(
                    model_name=self.robot_name,
                    model_xml=sdf_xml,
                    robot_namespace=self.robot_name,
                    initial_pose=pose,
                    reference_frame="world",
                )
            except rospy.ServiceException as e:
                rospy.logerr("Failed to spawn robot '%s': %s", self.robot_name, str(e))

    # ==========================================================
    # SEND MODEL STATE TO GAZEBO
    # ==========================================================
    def send_obstacle_state(self, model_name, gx, gy):
        wx, wy = self.grid_to_world(gx, gy)

        state = ModelState()
        state.model_name = model_name
        state.pose = Pose()
        state.pose.position.x = wx
        state.pose.position.y = wy
        state.pose.position.z = self.z_base + self.obstacle_height / 2.0
        state.pose.orientation.w = 1.0
        state.twist = Twist()
        state.reference_frame = "world"

        try:
            self.set_model_state(state)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to move %s: %s", model_name, str(e))

    def send_robot_state(self, gx, gy):
        wx, wy = self.grid_to_world(gx, gy)

        state = ModelState()
        state.model_name = self.robot_name
        state.pose = Pose()
        state.pose.position.x = wx
        state.pose.position.y = wy
        state.pose.position.z = self.z_base + self.robot_radius
        state.pose.orientation.w = 1.0
        state.twist = Twist()
        state.reference_frame = "world"

        try:
            self.set_model_state(state)
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to move robot '%s': %s", self.robot_name, str(e))

    # ==========================================================
    # REPLAY WITH INTERPOLATION + REPLAY SPEED
    # ==========================================================
    def replay_once(self):
        if not self.time_stamps:
            rospy.logwarn("No trajectory loaded.")
            return

        rospy.loginfo("Starting dynamic obstacle + robot replay (interpolated, speed=%.2fx)",
                      self.replay_speed)
        n_steps = len(self.time_stamps)

        if n_steps < 2:
            rospy.logwarn("Not enough steps for interpolation.")
            return

        for i in range(n_steps - 1):
            if rospy.is_shutdown():
                break

            t0 = self.time_stamps[i]
            t1 = self.time_stamps[i + 1]
            seg_real = t1 - t0
            if seg_real <= 0.0:
                seg_real = 0.02  # fallback

            # Adjust segment time by replay_speed
            seg_sim = seg_real / self.replay_speed

            # Number of interpolation substeps based on simulated segment duration
            n_sub = max(1, int(seg_sim * self.interp_rate))
            sub_dt = seg_sim / float(n_sub)

            # Pre-fetch endpoints in grid coordinates
            # Robot
            if i < len(self.robot_traj) - 1:
                gx_r0, gy_r0 = self.robot_traj[i]
                gx_r1, gy_r1 = self.robot_traj[i + 1]
            else:
                gx_r0, gy_r0 = self.robot_traj[i]
                gx_r1, gy_r1 = gx_r0, gy_r0

            # Obstacles: for each, get (gx0, gy0) and (gx1, gy1)
            obs_p0 = {}
            obs_p1 = {}
            for name in self.obstacle_names:
                data = self.obstacle_data[name]
                if i < len(data) - 1:
                    (gx0, gy0) = data[i]
                    (gx1, gy1) = data[i + 1]
                else:
                    (gx0, gy0) = data[i]
                    (gx1, gy1) = (gx0, gy0)
                obs_p0[name] = (gx0, gy0)
                obs_p1[name] = (gx1, gy1)

            # Interpolate along this segment
            for j in range(n_sub):
                if rospy.is_shutdown():
                    break

                alpha = float(j) / float(n_sub)  # in [0,1)

                # Robot interpolation
                gx_r = gx_r0 + alpha * (gx_r1 - gx_r0)
                gy_r = gy_r0 + alpha * (gy_r1 - gy_r0)
                self.send_robot_state(gx_r, gy_r)

                # Obstacles interpolation
                for name in self.obstacle_names:
                    gx0, gy0 = obs_p0[name]
                    gx1, gy1 = obs_p1[name]
                    gx = gx0 + alpha * (gx1 - gx0)
                    gy = gy0 + alpha * (gy1 - gy0)
                    self.send_obstacle_state(name, gx, gy)

                rospy.sleep(sub_dt)

        # Make sure we end exactly at the final positions
        final_idx = n_steps - 1
        if final_idx < len(self.robot_traj):
            gx_r, gy_r = self.robot_traj[final_idx]
            self.send_robot_state(gx_r, gy_r)

        for name in self.obstacle_names:
            data = self.obstacle_data[name]
            if final_idx < len(data):
                gx, gy = data[final_idx]
            else:
                gx, gy = data[-1]
            self.send_obstacle_state(name, gx, gy)

        rospy.loginfo("Replay finished.")

    def run(self):
        while not rospy.is_shutdown():
            self.replay_once()
            if not self.loop:
                break
            rospy.loginfo("Looping replay...")


if __name__ == "__main__":
    node = DynamicTrajectoryReplay()
    node.run()
