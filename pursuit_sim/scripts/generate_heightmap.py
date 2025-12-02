#!/usr/bin/env python3
import numpy as np
from PIL import Image
import argparse

def parse_mapfile(filename):
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

        costmap = np.asarray(costmap).T  # same as your visualizer

    return x_size, y_size, collision_threshold, robotX, robotY, target_trajectory, costmap

def costmap_to_heightmap_png(costmap: np.ndarray,
                             out_path: str = "heightmap.png",
                             max_height: float = 5.0):
    arr = np.array(costmap, dtype=np.float32)
    c_min, c_max = float(arr.min()), float(arr.max())

    if c_max > c_min:
        norm = (arr - c_min) / (c_max - c_min)
    else:
        norm = np.zeros_like(arr)

    height = norm * max_height
    img16 = (height / max_height * 65535.0).clip(0, 65535).astype(np.uint16)

    img = Image.fromarray(img16, mode="I;16")
    img.save(out_path)
    print(f"[heightmap] Saved {out_path}, cost range [{c_min}, {c_max}], "
          f"height [0, {max_height}] m")

def main():
    parser = argparse.ArgumentParser(
        description="Convert planner map txt into a Gazebo heightmap PNG."
    )
    parser.add_argument("map_file", help="Map txt file (N/C/R/T/M format)")
    parser.add_argument("--heightmap_png", default="heightmap.png")
    parser.add_argument("--max_height", type=float, default=5.0)
    args = parser.parse_args()

    (_, _, _, _, _, _, costmap) = parse_mapfile(args.map_file)
    costmap_to_heightmap_png(costmap,
                             out_path=args.heightmap_png,
                             max_height=args.max_height)

if __name__ == "__main__":
    main()
