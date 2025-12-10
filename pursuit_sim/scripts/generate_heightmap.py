#!/usr/bin/env python3
import numpy as np
from PIL import Image
import argparse
import os
import textwrap

# =========================
# Helpers
# =========================

def parse_mapfile(filename):
    """
    Parse 16782-style map file with format:

      N
      x_size,y_size
      C
      collision_threshold
      R
      robotX,robotY
      T
      x,y
      ...
      M
      costmap rows ...

    Returns:
        x_size, y_size, collision_threshold, robotX, robotY, target_trajectory, costmap
    """
    with open(filename, 'r') as file:
        assert file.readline().strip() == 'N', "Expected 'N' in the first line"
        x_size, y_size = map(int, file.readline().strip().split(','))

        assert file.readline().strip() == 'C', "Expected 'C' in the third line"
        collision_threshold = int(file.readline().strip())

        assert file.readline().strip() == 'R', "Expected 'R' in the fifth line"
        robotX, robotY = map(int, file.readline().strip().split(','))

        assert file.readline().strip() == 'T', "Expected 'T' in the seventh line"
        target_trajectory = []
        line = file.readline().strip()
        while line != 'M':
            x, y = map(float, line.split(','))
            target_trajectory.append({'x': x, 'y': y})
            line = file.readline().strip()

        costmap = []
        for line in file:
            row = list(map(float, line.strip().split(',')))
            costmap.append(row)

        # Match visualizer convention: columns are x, rows are y, then transpose
        costmap = np.asarray(costmap).T

    return x_size, y_size, collision_threshold, robotX, robotY, target_trajectory, costmap


def next_2n_plus1(n: int) -> int:
    """
    Return the smallest value of the form (2^k + 1) that is >= n.

    Gazebo classic heightmaps require image width == height == 2^n + 1.
    """
    if n <= 2:
        return 2  # Minimum reasonable size
    k = 0
    while (1 << k) < (n - 1):
        k += 1
    return (1 << k) + 1


def pad_costmap_for_heightmap(arr: np.ndarray):
    """
    Gazebo classic heightmap requirements:
      - Image must be square
      - Side length must be 2^n + 1

    We:
      1) Take max(h, w) of the original array
      2) Compute the smallest side >= max(h, w) of the form 2^n + 1
      3) Pad with the minimum cost value

    IMPORTANT:
      - Padding is applied ONLY to the bottom and right.
      - Top-left pixel (0,0) of the original map stays at (0,0) of the image.

    Returns:
      padded_arr, pad_x_after, pad_y_after
    """
    h, w = arr.shape  # (rows, cols) = (y, x)
    c_min = float(arr.min())

    target_side = next_2n_plus1(max(h, w))

    pad_y_before = 0
    pad_x_before = 0
    pad_y_after = target_side - h
    pad_x_after = target_side - w

    padded = np.pad(
        arr,
        ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
        mode="constant",
        constant_values=c_min,
    )

    print(f"[heightmap] Padded costmap from {h}x{w} to {target_side}x{target_side} (2^n+1)")
    print(f"[heightmap] Pad y: before={pad_y_before}, after={pad_y_after}")
    print(f"[heightmap] Pad x: before={pad_x_before}, after={pad_x_after}")

    return padded, pad_x_after, pad_y_after


def costmap_to_heightmap_png(costmap: np.ndarray,
                             out_path: str = "heightmap.png",
                             max_height: float = 5.0,
                             bit_depth: int = 8):
    """
    Convert a 2D costmap array into a PNG suitable for Gazebo heightmap.

    - Pads to square 2^n+1 × 2^n+1
      (padding only on bottom and right → origin at top-left preserved)
    - Normalizes cost range [c_min, c_max] to [0, 1]
    - Encodes as:
        * 8-bit grayscale (mode 'L') if bit_depth == 8  [RECOMMENDED]
        * 16-bit grayscale (mode 'I;16') if bit_depth == 16

    Returns:
        img_width, img_height, pad_x_after, pad_y_after
    """
    arr = np.array(costmap, dtype=np.float32)
    original_h, original_w = arr.shape
    arr, pad_x_after, pad_y_after = pad_costmap_for_heightmap(arr)

    padded_h, padded_w = arr.shape
    c_min, c_max = float(arr.min()), float(arr.max())

    if c_max > c_min:
        norm = (arr - c_min) / (c_max - c_min)   # 0..1
    else:
        norm = np.zeros_like(arr)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if bit_depth == 16:
        img_vals = (norm * 65535.0).clip(0, 65535).astype(np.uint16)
        img = Image.fromarray(img_vals, mode="I;16")
    else:
        img_vals = (norm * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_vals, mode="L")

    img.save(out_path)
    print(f"[heightmap] Saved {out_path}, "
          f"original={original_w}x{original_h}, padded={padded_w}x{padded_h}, "
          f"bit_depth={bit_depth}, cost range [{c_min}, {c_max}], "
          f"height [0, {max_height}] m (via <size> in world)")

    return padded_w, padded_h, pad_x_after, pad_y_after


def compute_heightmap_center_top_left(world_size_x: float, world_size_y: float):
    """
    Compute the <pos> (center) of the heightmap so that its TOP-LEFT corner
    lies at world (0,0).

    Heightmap extents in world will be:
      x ∈ [0, world_size_x]
      y ∈ [-world_size_y, 0]

    Gazebo interprets <pos> as the *center* of the heightmap geometry.
    To get top-left at (0,0), we place the center at:

      center_x = world_size_x / 2
      center_y = -world_size_y / 2
    """
    center_x = world_size_x / 2.0
    center_y = -world_size_y / 2.0
    return center_x, center_y


def write_world_with_inlined_heightmap(world_path: str,
                                       png_path: str,
                                       world_size_x: float,
                                       world_size_y: float,
                                       img_width: int,
                                       img_height: int,
                                       max_height: float,
                                       model_name: str = "terrain_from_map",
                                       color_rgb=(0.5, 0.5, 0.5)):
    """
    Create a Gazebo world that inlines a heightmap model referencing the given PNG.

    - Heightmap image is square 2^n+1×2^n+1.
    - <size> in SDF is the physical size in meters that the image spans.

    Here we set:
      world_size_x = img_width  * meters_per_cell
      world_size_y = img_height * meters_per_cell

    With meters_per_cell = 1.0:
      1 image pixel = 1 meter in both x and y.

    We place the heightmap so that its TOP-LEFT corner is at (0,0) in world
    coordinates by centering it at (world_size_x/2, -world_size_y/2).
    That means the image covers:
      x ∈ [0, world_size_x]
      y ∈ [-world_size_y, 0]

    Physics is effectively disabled:
      - gravity = 0
      - physics type=ode but max_step_size=0 and real_time_update_rate=0
      - no <ode> solver/contact tuning → no ODE stepping used
    """
    world_dir = os.path.dirname(world_path)
    if world_dir:
        os.makedirs(world_dir, exist_ok=True)

    png_abs = os.path.abspath(png_path)
    png_uri = f"file://{png_abs}"

    r, g, b = color_rgb

    res_x = world_size_x / float(img_width)
    res_y = world_size_y / float(img_height)
    print(f"[world] Physical size: {world_size_x} x {world_size_y} m")
    print(f"[world] Image size:    {img_width} x {img_height} px")
    print(f"[world] Resolution:    {res_x:.4f} m/px (x), {res_y:.4f} m/px (y)")

    center_x, center_y = compute_heightmap_center_top_left(world_size_x, world_size_y)
    print(f"[world] Heightmap center placed at ({center_x}, {center_y}) "
          f"so top-left is at (0,0) in world.")

    # Use lower thresholds so high ground visually pops
    # max_height might be ~0.5–1.0; we emphasize heights above ~0.1 m
    wall_height_start = max_height * 0.2  # start "building/wall" texture at 20% of max

    world_xml = textwrap.dedent(f"""\
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <world name="terrain_world">

            
            <gravity>0 0 -9.81</gravity>
            <physics type="ode">
              <max_step_size>0.001</max_step_size>
              <real_time_update_rate>1000.0</real_time_update_rate>
              <real_time_factor>1.0</real_time_factor>
            </physics>

            <!-- Point lights only: no sun -->
            <light type="point" name="point_light_center">
              <pose>{world_size_x/2.0} {-world_size_y/2.0} {max_height*3.0} 0 0 0</pose>
              <diffuse>1 1 1 1</diffuse>
              <specular>0.5 0.5 0.5 1</specular>
              <attenuation>
                <range>{max(world_size_x, world_size_y)}</range>
                <constant>0.4</constant>
                <linear>0.01</linear>
                <quadratic>0.001</quadratic>
              </attenuation>
              <cast_shadows>true</cast_shadows>
            </light>

            <light type="point" name="point_light_north">
              <pose>{world_size_x/2.0} 0 {max_height*4.0} 0 0 0</pose>
              <diffuse>0.9 0.9 1 1</diffuse>
              <specular>0.4 0.4 0.6 1</specular>
              <attenuation>
                <range>{max(world_size_x, world_size_y)}</range>
                <constant>0.5</constant>
                <linear>0.02</linear>
                <quadratic>0.002</quadratic>
              </attenuation>
              <cast_shadows>true</cast_shadows>
            </light>

            <light type="point" name="point_light_south">
              <pose>{world_size_x/2.0} {-world_size_y} {max_height*4.0} 0 0 0</pose>
              <diffuse>1 0.95 0.9 1</diffuse>
              <specular>0.6 0.5 0.4 1</specular>
              <attenuation>
                <range>{max(world_size_x, world_size_y)}</range>
                <constant>0.5</constant>
                <linear>0.02</linear>
                <quadratic>0.002</quadratic>
              </attenuation>
              <cast_shadows>true</cast_shadows>
            </light>

            <model name="{model_name}">
              <static>true</static>
              <link name="terrain_link">

                <!-- Collision is still defined but physics stepping is disabled, so this is effectively visual only -->
                <collision name="terrain_collision">
                  <geometry>
                    <heightmap>
                      <uri>{png_uri}</uri>
                      <size>{world_size_x} {world_size_y} {max_height}</size>
                      <pos>{center_x} {center_y} 0</pos>
                    </heightmap>
                  </geometry>
                </collision>

                <visual name="terrain_visual">
                  <geometry>
                    <heightmap>
                      <uri>{png_uri}</uri>
                      <size>{world_size_x} {world_size_y} {max_height}</size>
                      <pos>{center_x} {center_y} 0</pos>

                      <!-- Base ground texture -->
                      <texture>
                        <diffuse>file://media/materials/textures/dirt_diffusespecular.png</diffuse>
                        <normal>file://media/materials/textures/flat_normal.png</normal>
                        <size>10</size>
                      </texture>

                      <!-- Higher-elevation "building/wall" texture -->
                      <texture>
                        <diffuse>file://media/materials/textures/terrain_detail.jpg</diffuse>
                        <normal>file://media/materials/textures/flat_normal.png</normal>
                        <size>4</size>
                      </texture>

                      <!-- Ground: everything from 0 up -->
                      <blend>
                        <min_height>0.0</min_height>
                        <fade_dist>0.05</fade_dist>
                      </blend>

                      <!-- Walls/buildings: anything above ~20% of max_height -->
                      <blend>
                        <min_height>{wall_height_start}</min_height>
                        <fade_dist>0.05</fade_dist>
                      </blend>
                    </heightmap>
                  </geometry>

                  <!-- Tint on top of the textures -->
                  <material>
                    <ambient>{r} {g} {b} 1</ambient>
                    <diffuse>{r} {g} {b} 1</diffuse>
                    <specular>0.1 0.1 0.1 1</specular>
                    <emissive>0 0 0 1</emissive>
                  </material>
                </visual>

              </link>
            </model>

          </world>
        </sdf>
    """)
    with open(world_path, "w") as f:
        f.write(world_xml)
    print(f"[world] Wrote {world_path}")
    print(f"[world] Heightmap URI: {png_uri}")
    print(f"[world] Color (RGB): {r}, {g}, {b}")


# =========================
# main()
# =========================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert planner map txt into a Gazebo heightmap PNG and purely-visual world file."
    )
    parser.add_argument("map_file", help="Map txt file (N/C/R/T/M format)")
    parser.add_argument("--heightmap_png",
                        default="heightmap.png",
                        help="Output PNG path for heightmap (default: heightmap.png)")
    parser.add_argument("--world_path",
                        required=True,
                        help="Output world file path")
    parser.add_argument("--max_height",
                        type=float,
                        default=1.0,
                        help="Max terrain height in meters (default: 1.0)")
    parser.add_argument("--model_name",
                        default="terrain_from_map",
                        help="Name of the inlined model in the world")
    parser.add_argument(
        "--color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=[0.7, 0.7, 0.7],
        help="RGB color for the terrain visual (0–1 each).",
    )
    parser.add_argument(
        "--bit_depth",
        choices=[8, 16],
        type=int,
        default=8,
        help="PNG bit depth: 8 or 16 (default: 8)",
    )
    parser.add_argument(
        "--meters_per_cell",
        type=float,
        default=1.0,
        help="Physical size of each image pixel in meters (default: 1.0).",
    )

    if argv is None:
        argv = []
    args, unknown = parser.parse_known_args(argv)

    x_size, y_size, _, robotX, robotY, target_traj, costmap = parse_mapfile(args.map_file)
    print(f"[map] Parsed {args.map_file} -> "
          f"declared size=({x_size}, {y_size}), costmap.shape={costmap.shape}")
    print(f"[map] Robot start in grid coords: ({robotX}, {robotY})")

    # Generate heightmap PNG (square 2^n+1, bottom/right padding)
    img_width, img_height, pad_x_after, pad_y_after = costmap_to_heightmap_png(
        costmap,
        out_path=args.heightmap_png,
        max_height=args.max_height,
        bit_depth=args.bit_depth
    )

    world_size_x = img_width  * args.meters_per_cell
    world_size_y = img_height * args.meters_per_cell

    # Mapping info: planner (gx, gy) -> world (x, y) with top-left origin and y-down in planner
    print("\n[mapping] Planner grid origin (0,0) is TOP-LEFT of ORIGINAL map.")
    print(f"[mapping] Padding (bottom,right) = (pad_y_after={pad_y_after}, pad_x_after={pad_x_after})")
    print("[mapping] With meters_per_cell = 1.0 and top-left of heightmap at world (0,0):")
    print("          x_world = (gx + 0.5)")
    print("          y_world = -(gy + 0.5)")
    print("        i.e. planner y-down corresponds to world y-negative.")

    gx, gy = robotX, robotY
    x_world_robot = (gx + 0.5) * args.meters_per_cell
    y_world_robot = -(gy + 0.5) * args.meters_per_cell
    print(f"[mapping] Example: robot grid ({gx},{gy}) -> world ≈ ({x_world_robot:.3f}, {y_world_robot:.3f})")

    # Write world file
    write_world_with_inlined_heightmap(
        world_path=args.world_path,
        png_path=args.heightmap_png,
        world_size_x=world_size_x,
        world_size_y=world_size_y,
        img_width=img_width,
        img_height=img_height,
        max_height=args.max_height,
        model_name=args.model_name,
        color_rgb=args.color,
    )

    print("\n[done]")
    print("Launch Gazebo with:")
    print(f"  roslaunch gazebo_ros empty_world.launch \\")
    print(f"    world_name:={os.path.abspath(args.world_path)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
