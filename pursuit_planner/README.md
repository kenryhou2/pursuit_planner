# pursuit_planner

A ROS1 planner node for a simple **pursuit / evasion** scenario on a 2D grid map.

The node reads a static occupancy map, an optional set of dynamic obstacles, and
the robot/target poses and trajectory. It then continuously publishes a
**next waypoint** and reports when the target is considered "caught".

---

## 1. Package Overview

- **Node executable**: `pursuit_planner_node`
- **Launch file**: `pursuit_planner.launch`
- **Default namespace**: `pursuit_planner`

---

## 2. Launch File

Example:

```xml
<launch>
  <arg name="map_file"        default="$(find pursuit_planner)/maps/map3.txt" />
  <arg name="dyno_yaml"       default="$(find pursuit_planner)/config/dyno_map3.yaml" />
  <arg name="catch_threshold" default="0.5" />

  <arg name="planner_ns" default="pursuit_planner" />

  <node pkg="pursuit_planner"
        type="pursuit_planner_node"
        name="pursuit_planner_node"
        ns="$(arg planner_ns)"
        output="screen">

    <param name="map_file"        value="$(arg map_file)" />
    <param name="dyno_yaml"       value="$(arg dyno_yaml)" />
    <param name="catch_threshold" value="$(arg catch_threshold)" />
  </node>
</launch>
```

---

## 3. Parameters

### `~map_file`
Path to the grid map.

### `~dyno_yaml`
YAML file describing dynamic obstacles.

### `~catch_threshold`
Distance at which target is considered caught.

---

## 4. Topics

### Subscribed
- `robot_pose` (`geometry_msgs/Pose2D`)
- `target_traj` (`std_msgs/Int32MultiArray`)
- `target_pose` (`geometry_msgs/Pose2D`)

### Published
- `next_waypoint` (`geometry_msgs/Pose2D`)
- `planner_status` (`std_msgs/String`)
- `target_caught` (`std_msgs/Bool`)
- `/dynamic_obstacles/<id>/pose` (`geometry_msgs/Pose2D`)

---

## 5. Building

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 6. Running

```bash
roslaunch pursuit_planner pursuit_planner.launch
```

Override:

```bash
roslaunch pursuit_planner pursuit_planner.launch   map_file:=/path/to/map.txt   dyno_yaml:=/path/to/dyno.yaml   catch_threshold:=1.0
```

---

## 7. Manual Testing

```bash
rostopic pub /pursuit_planner/robot_pose geometry_msgs/Pose2D "{x: 0, y: 0, theta: 0}" -r 10
rostopic pub /pursuit_planner/target_pose geometry_msgs/Pose2D "{x: 10, y: 10, theta: 0}" -r 10
rostopic pub /pursuit_planner/target_traj std_msgs/Int32MultiArray "data: [0,1,2,3]" -1
```

Monitor:

```bash
rostopic echo /pursuit_planner/next_waypoint
rostopic echo /pursuit_planner/planner_status
rostopic echo /pursuit_planner/target_caught
```

---

## 8. Debugging

```bash
rosparam get /pursuit_planner/pursuit_planner_node
```

---

## 9. Quick Start Checklist
1. Build workspace  
2. Launch planner  
3. Publish robot and target states  
4. Observe planner outputs  
5. Tune thresholds and map/YAML configs  