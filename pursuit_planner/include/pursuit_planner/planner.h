#ifndef PLANNER_H
#define PLANNER_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

// Obstacle structure definitions
struct Footprint {
    std::string kind;
    double radius = 0.0;
    int width = 1;
    int height = 1;
};

struct ObstacleState {
    int x, y;
    Footprint footprint;
};

struct CompactDynamicObstacles {
    std::vector<std::vector<ObstacleState>> timesteps;
    int max_time;
    
    // Helper function to check if a position is occupied at time t
    bool isOccupied(int x, int y, int t) const {
        if (t < 0 || t >= (int)timesteps.size()) {
            // Use last timestep if beyond max
            t = std::min((int)timesteps.size() - 1, std::max(0, t));
        }
        if (timesteps.empty() || t >= (int)timesteps.size()) return false;
        
        for (const auto& obs : timesteps[t]) {
            // Check if (x,y) is within this obstacle's footprint
            if (obs.footprint.kind == "point") {
                if (obs.x == x && obs.y == y) {
                    return true;
                }
            }
            else if (obs.footprint.kind == "circle") {
                int dx = x - obs.x;
                int dy = y - obs.y;
                double dist_sq = dx*dx + dy*dy;
                double R_sq = obs.footprint.radius * obs.footprint.radius;
                
                if (dist_sq <= R_sq) {
                    return true;
                }
            }
            else if (obs.footprint.kind == "box") {
                int half_w = obs.footprint.width / 2;
                int half_h = obs.footprint.height / 2;
                if (std::abs(x - obs.x) <= half_w && std::abs(y - obs.y) <= half_h) {
                    return true;
                }
            }
        }
        
        return false;
    }
};

// Declare the plan function
void planner(
    int* map,
    CompactDynamicObstacles& compactObs,
    int collision_thresh,
    int x_size,
    int y_size,
    int robotposeX,
    int robotposeY,
    int target_steps,
    int* target_traj,
    int targetposeX,
    int targetposeY,
    int curr_time,
    int* action_ptr
    );

#endif // PLANNER_H