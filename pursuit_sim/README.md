# pursuit_sim: Terrain Generation and Simulation Launch

This package provides tooling to convert a 2D planner map file (from CMU 16782 format) into a Gazebo-compatible heightmap and automatically launch a simulation world with a simple cube robot.

---
### Run the Simulation

```bash
roslaunch pursuit_sim terrain_with_cube.launch
```

This will:
- Create the heightmap PNG at  
  `pursuit_sim/worlds/heightmap_map3.png`
- Generate the world file  
  `pursuit_sim/worlds/terrain_world.world`
- Launch Gazebo with the terrain
- Insert a cube robot at coordinates `(5, 5, 2)`

---


## Overview

### What `generate_heightmap.py` Does

```markdown
- Parses a 16782-style planner map file (N/C/R/T/M format)
- Converts the 2D costmap into a padded square heightmap (size = 2^n + 1)
- Normalizes cost values into terrain elevation [0, max_height]
- Writes an 8‑bit or 16‑bit grayscale PNG heightmap
- Generates a Gazebo `.world` file with an inlined heightmap model
- Adds texture & material layers for proper visualization in Gazebo
```

The resulting terrain accurately reflects the costmap’s structure and can be used to visualize planning or dynamic obstacle simulations.

---

## Launch File: `terrain_with_cube.launch`

This launch file runs the full pipeline:

1. **Generates the heightmap PNG + world file**  
   using the `generate_heightmap.py` script.

2. **Starts Gazebo (`gzserver` + `gzclient`)**  
   loading the newly created terrain world.

3. **Spawns a simple cube robot**  
   using the SDF file in `pursuit_sim/models/cube_robot/model.sdf`.


## Modifying the Terrain Generation

You can adjust height scaling:

```bash
--max_height=10.0
```

Change PNG resolution/bit depth:

```bash
--bit_depth=8   # recommended for Gazebo stability
```

Change terrain color tint:

```bash
--color 0.7 0.7 0.7
```

---

## Example Standalone Command

If you want to generate the terrain manually:

```bash
rosrun pursuit_sim generate_heightmap.py   $(rospack find pursuit_planner)/maps/map3.txt   --heightmap_png=$(rospack find pursuit_sim)/worlds/heightmap_map3.png   --world_path=$(rospack find pursuit_sim)/worlds/terrain_world.world   --max_height=5.0   --bit_depth=8
```

---

## Directory Structure

Expected layout inside `pursuit_sim`:

```
pursuit_sim/
├── launch/
│   └── terrain_with_cube.launch
├── models/
│   └── cube_robot/
│       └── model.sdf
├── scripts/
│   └── generate_heightmap.py
├── worlds/
│   ├── heightmap_map3.png
│   └── terrain_world.world
└── README.md 
```

---

## Notes

- Gazebo Classic requires heightmap images to be **square** and **2^n + 1** on a side.
- If you modify the PNG, Gazebo may cache stale heightmap tiles in `~/.gazebo/paging/`.
- Removing that directory forces a fresh rebuild of the terrain.

---

For further debugging, launch with verbose output:

```bash
roslaunch pursuit_sim terrain_with_cube.launch --screen
```
