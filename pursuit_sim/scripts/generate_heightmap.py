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

        # Match your visualizer convention: columns are x, rows are y, then transpose
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
    # We want 2^k + 1 >= n  =>  2^k >= n - 1
    while (1 << k) < (n - 1):
        k += 1
    return (1 << k) + 1


def pad_costmap_for_heightmap(arr: np.ndarray) -> np.ndarray:
    """
    Gazebo classic heightmap requirements:
      - Image must be square
      - Side length must be 2^n + 1

    We:
      1) Take max(h, w) of the original array
      2) Compute the smallest side >= max(h, w) of the form 2^n + 1
      3) Pad symmetrically with the minimum cost value
    """
    h, w = arr.shape  # (rows, cols) = (y, x)
    c_min = float(arr.min())

    target_side = next_2n_plus1(max(h, w))

    pad_y_total = target_side - h
    pad_x_total = target_side - w

    pad_y_before = pad_y_total // 2
    pad_y_after  = pad_y_total - pad_y_before
    pad_x_before = pad_x_total // 2
    pad_x_after  = pad_x_total - pad_x_before

    padded = np.pad(
        arr,
        ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
        mode="constant",
        constant_values=c_min,
    )

    print(f"[heightmap] Padded costmap from {h}x{w} to {target_side}x{target_side} (2^n+1)")

    return padded


def costmap_to_heightmap_png(costmap: np.ndarray,
                             out_path: str = "heightmap.png",
                             max_height: float = 5.0):
    """
    Convert a 2D costmap array into a 16-bit PNG suitable for Gazebo heightmap.

    - Pads to square 2^n+1 × 2^n+1
    - Normalizes cost range [c_min, c_max] to height [0, max_height]
    - Encodes as 16-bit grayscale (I;16)
    """
    # 1) Pad to square with side length 2^n + 1
    arr = np.array(costmap, dtype=np.float32)
    arr = pad_costmap_for_heightmap(arr)

    c_min, c_max = float(arr.min()), float(arr.max())

    if c_max > c_min:
        norm = (arr - c_min) / (c_max - c_min)
    else:
        norm = np.zeros_like(arr)

    height = norm * max_height
    img16 = (height / max_height * 65535.0).clip(0, 65535).astype(np.uint16)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img = Image.fromarray(img16, mode="I;16")
    img.save(out_path)
    print(f"[heightmap] Saved {out_path}, padded shape={arr.shape}, "
          f"cost range [{c_min}, {c_max}], height [0, {max_height}] m")


def write_world_with_inlined_heightmap(world_path: str,
                                       png_path: str,
                                       x_size: int,
                                       y_size: int,
                                       max_height: float,
                                       model_name: str = "terrain_from_map"):
    """
    Create a Gazebo world that inlines a heightmap model referencing the given PNG.

    - The heightmap image itself is square 2^n+1×2^n+1.
    - <size> in SDF is the physical size in meters that the image spans.
      We set it to (x_size, y_size, max_height) so it matches your planner grid.
    """
    world_dir = os.path.dirname(world_path)
    if world_dir:
        os.makedirs(world_dir, exist_ok=True)

    png_abs = os.path.abspath(png_path)
    png_uri = f"file://{png_abs}"

    world_xml = textwrap.dedent(f"""\
        <?xml version="1.0" ?>
        <sdf version="1.6">
          <world name="terrain_world">

            <include>
              <uri>model://sun</uri>
            </include>

            <!-- Inlined heightmap model -->
            <model name="{model_name}">
              <static>true</static>
              <link name="terrain_link">
                <collision name="terrain_collision">
                  <geometry>
                    <heightmap>
                      <uri>{png_uri}</uri>
                      <size>{x_size} {y_size} {max_height}</size>
                      <pos>0 0 0</pos>
                    </heightmap>
                  </geometry>
                </collision>

                <visual name="terrain_visual">
                  <geometry>
                    <heightmap>
                      <uri>{png_uri}</uri>
                      <size>{x_size} {y_size} {max_height}</size>
                      <pos>0 0 0</pos>
                    </heightmap>
                  </geometry>
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


# =========================
# main()
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Convert planner map txt into a Gazebo heightmap PNG and world file."
    )
    parser.add_argument("map_file", help="Map txt file (N/C/R/T/M format)")
    parser.add_argument("--heightmap_png",
                        default="heightmap.png",
                        help="Output PNG path for heightmap (default: heightmap.png)")
    parser.add_argument("--world_path",
                        required=True,
                        help="Output world file path, e.g. $(rospack find pursuit_sim)/worlds/terrain_world.world")
    parser.add_argument("--max_height",
                        type=float,
                        default=5.0,
                        help="Max terrain height in meters (default: 5.0)")
    parser.add_argument("--model_name",
                        default="terrain_from_map",
                        help="Name of the inlined model in the world (default: terrain_from_map)")
    args = parser.parse_args()

    # 1) Parse map
    x_size, y_size, _, _, _, _, costmap = parse_mapfile(args.map_file)
    print(f"[map] Parsed {args.map_file} -> size=({x_size}, {y_size}), costmap.shape={costmap.shape}")

    # 2) Generate heightmap PNG (with square 2^n+1 padding)
    costmap_to_heightmap_png(costmap,
                             out_path=args.heightmap_png,
                             max_height=args.max_height)

    # 3) Generate world file that inlines the heightmap
    write_world_with_inlined_heightmap(
        world_path=args.world_path,
        png_path=args.heightmap_png,
        x_size=x_size,       # physical X extent (meters)
        y_size=y_size,       # physical Y extent (meters)
        max_height=args.max_height,
        model_name=args.model_name
    )

    print("\n[done]")
    print("Launch Gazebo with:")
    print(f"  roslaunch gazebo_ros empty_world.launch \\")
    print(f"    world_name:={os.path.abspath(args.world_path)}")

if __name__ == "__main__":
    main()
