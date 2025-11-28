/*=================================================================
 *
 * planner.cpp
 *
 *=================================================================*/
#include "pursuit_planner/planner.h"
#include <math.h>
#include <iostream>
#include <cstdio>
#include <queue>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <algorithm>
using namespace std;

#define GETMAPINDEX(X, Y, XSIZE, YSIZE) ((Y-1)*XSIZE + (X-1))

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

#define NUMOFDIRS 8

// Navigation directions for 8-connected grid plus staying in place
int dX[NUMOFDIRS + 1] = {-1, -1, -1,  0,  0,  1, 1, 1, 0};
int dY[NUMOFDIRS + 1] = {-1,  0,  1, -1,  1, -1, 0, 1, 0};

// Global state management
struct PlannerState {
    vector<int> robotToAllCosts;
    vector<int> goalToAllCosts;
    int* timeSteps;
    bool initialized;
    int gridWidth, gridHeight, obstacleThreshold;
    int* terrainMap;
    int selectedGoalX, selectedGoalY;
    int maxTimeHorizon;
    
    PlannerState() : initialized(false), timeSteps(nullptr) {}
    
    void cleanup() {
        if (timeSteps) {
            delete[] timeSteps;
            timeSteps = nullptr;
        }
    }
};

static PlannerState globalState;


// Validate grid coordinates and time constraints
inline bool isValidPosition(int x, int y, int time, int maxTime) {
    return (x > 0) && (x <= globalState.gridWidth) && 
           (y > 0) && (y <= globalState.gridHeight) && 
           (time <= maxTime);
}

// Node representation for pathfinding with temporal dimension
struct PathNode {
private:
    int position_x, position_y;
    int timestamp;
    int path_cost, estimated_cost;
    int parent_action;

public:
    // Constructor with member initializer list
    PathNode(int px, int py, int time, int g_val, int f_val, int action) 
        : position_x(px), position_y(py), timestamp(time), 
          path_cost(g_val), estimated_cost(f_val), parent_action(action) {}
    
    // Accessors
    int getX() const { return position_x; }
    int getY() const { return position_y; }
    int getTime() const { return timestamp; }
    int getCost() const { return path_cost; }
    int getEstimate() const { return estimated_cost; }
    int getAction() const { return parent_action; }
    
    // Equality comparison for spatial-temporal coordinates
    bool isSameLocation(const PathNode& other) const {
        return (position_x == other.position_x && 
                position_y == other.position_y && 
                timestamp == other.timestamp);
    }
};

struct HeuristicState {  
    int x, y, steps, cost;
    
    HeuristicState(int pos_x, int pos_y, int step_count, int heuristic_cost) 
        : x(pos_x), y(pos_y), steps(step_count), cost(heuristic_cost) {}
};

struct GridPosition {
    int x, y, step_num;
    
    GridPosition(int pos_x, int pos_y, int step_number) 
        : x(pos_x), y(pos_y), step_num(step_number) {}
};



vector<int> backwardHeuristicSearch(int goalX, int goalY, int curr_time, int target_steps) {
    const int total_cells = globalState.gridWidth * globalState.gridHeight;
    
    // search data structures using heap-based priority queue
    vector<HeuristicState> search_heap;
    vector<int> distance_values(total_cells, -1);
    vector<bool> processed(total_cells, false);
    
    // initialize with goal state
    HeuristicState goal_state(goalX, goalY, 0, 0);
    const int goal_idx = GETMAPINDEX(goalX, goalY, globalState.gridWidth, globalState.gridHeight);
    distance_values[goal_idx] = 0;
    
    // heap operations
    search_heap.push_back(goal_state);
    make_heap(search_heap.begin(), search_heap.end(), [](const HeuristicState& a, const HeuristicState& b) {
        return a.cost > b.cost;
    });
    
    while (!search_heap.empty()) {
        // extract minimum
        pop_heap(search_heap.begin(), search_heap.end(), [](const HeuristicState& a, const HeuristicState& b) {
            return a.cost > b.cost;
        });
        HeuristicState current_node = search_heap.back();
        search_heap.pop_back();
        
        const int current_idx = GETMAPINDEX(current_node.x, current_node.y, globalState.gridWidth, globalState.gridHeight);
        
        if (!processed[current_idx]) {
            processed[current_idx] = true;
            distance_values[current_idx] = current_node.cost;
            
            // check all 8 directions
            for (int direction = 0; direction < 8; ++direction) {
                const int neighbor_x = current_node.x + dX[direction];
                const int neighbor_y = current_node.y + dY[direction];
                const int time_step = current_node.steps + 1;
                
                if (isValidPosition(neighbor_x, neighbor_y, time_step, target_steps)) {
                    const int neighbor_idx = GETMAPINDEX(neighbor_x, neighbor_y, globalState.gridWidth, globalState.gridHeight);
                    const int cell_cost = globalState.terrainMap[neighbor_idx];
                    
                    if (cell_cost < globalState.obstacleThreshold) {
                        const int new_distance = cell_cost + current_node.cost;
                        
                        if ((distance_values[neighbor_idx] > new_distance) || (!processed[neighbor_idx])) {
                            distance_values[neighbor_idx] = new_distance;
                            // heap insertion
                            search_heap.push_back(HeuristicState(neighbor_x, neighbor_y, time_step, new_distance));
                            push_heap(search_heap.begin(), search_heap.end(), [](const HeuristicState& a, const HeuristicState& b) {
                                return a.cost > b.cost;
                            });
                            globalState.timeSteps[neighbor_idx] = time_step;
                        }
                    }
                }
            }
        }
    }
    
    return distance_values;
}

GridPosition AStarSearch(int start_x, int start_y, int target_x, int target_y, int current_time, int time_limit) {
    const int total_cells = globalState.gridWidth * globalState.gridHeight;
    
    // heap-based search data structures
    vector<int> actual_costs(total_cells, -1);
    vector<bool> visited(total_cells, false);
    vector<PathNode> frontier_heap;
    vector<PathNode> path_history;
    
    // initialize start node
    const int start_idx = GETMAPINDEX(start_x, start_y, globalState.gridWidth, globalState.gridHeight);
    const int start_estimate = globalState.goalToAllCosts[start_idx];
    PathNode initial_node(start_x, start_y, current_time, 0, start_estimate, -1);
    
    frontier_heap.push_back(initial_node);
    make_heap(frontier_heap.begin(), frontier_heap.end(), [](const PathNode& a, const PathNode& b) {
        return a.getEstimate() > b.getEstimate();
    });
    actual_costs[start_idx] = 0;
    
    // main a* search loop
    bool destination_reached = false;
    while (!frontier_heap.empty() && !destination_reached) {
        // extract minimum from heap
        pop_heap(frontier_heap.begin(), frontier_heap.end(), [](const PathNode& a, const PathNode& b) {
            return a.getEstimate() > b.getEstimate();
        });
        PathNode current_node = frontier_heap.back();
        frontier_heap.pop_back();
        
        const int current_idx = GETMAPINDEX(current_node.getX(), current_node.getY(), globalState.gridWidth, globalState.gridHeight);
        
        if (!visited[current_idx]) {
            visited[current_idx] = true;
            path_history.push_back(current_node);
            
            // explore all 9 directions (including staying in place)
            for (int move_dir = 0; move_dir < 9; ++move_dir) {
                const int next_x = current_node.getX() + dX[move_dir];
                const int next_y = current_node.getY() + dY[move_dir];
                const int next_time = current_node.getTime() + 1;
                
                if (!isValidPosition(next_x, next_y, next_time, time_limit)) continue;
                
                const int next_idx = GETMAPINDEX(next_x, next_y, globalState.gridWidth, globalState.gridHeight);
                const int move_cost = globalState.terrainMap[next_idx];
                
                if (move_cost >= globalState.obstacleThreshold) continue;
                if (visited[next_idx]) continue;
                
                const int total_cost = current_node.getCost() + move_cost;
                
                if (actual_costs[next_idx] == -1 || total_cost < actual_costs[next_idx]) {
                    actual_costs[next_idx] = total_cost;
                    if (globalState.goalToAllCosts[next_idx] < 0) {
                        continue; // skip if heuristic is invalid
                    }
                    const int estimated_total = total_cost + globalState.goalToAllCosts[next_idx];
                    // heap insertion
                    frontier_heap.push_back(PathNode(next_x, next_y, next_time, total_cost, estimated_total, move_dir));
                    push_heap(frontier_heap.begin(), frontier_heap.end(), [](const PathNode& a, const PathNode& b) {
                        return a.getEstimate() > b.getEstimate();
                    });
                }
            }
            
            // check if goal reached
            if (current_node.getX() == target_x && current_node.getY() == target_y) {
                destination_reached = true;
                
                // backtrack to find next move using vector operations
                while (!path_history.empty() && !current_node.isSameLocation(path_history.back())) {
                    path_history.pop_back();
                }
                
                if (path_history.empty()) return GridPosition(current_node.getX(), current_node.getY(), 0);
                
                PathNode trace_node = path_history.back();
                path_history.pop_back();
                
                // trace back to find first move
                while (!path_history.empty()) {
                    const int prev_x = trace_node.getX() - dX[trace_node.getAction()];
                    const int prev_y = trace_node.getY() - dY[trace_node.getAction()];
                    PathNode previous_node(prev_x, prev_y, trace_node.getTime() - 1, 0, 0, 0);
                    
                    while (!path_history.empty() && !previous_node.isSameLocation(path_history.back())) {
                        path_history.pop_back();
                    }
                    
                    if (path_history.empty()) break;
                    
                    previous_node = path_history.back();
                    path_history.pop_back();
                    
                    if (previous_node.getAction() == -1) {
                        return GridPosition(trace_node.getX(), trace_node.getY(), 0);
                    }
                    trace_node = previous_node;
                }
                return GridPosition(current_node.getX(), current_node.getY(), 0);
            }
        }
    }
    
    // no path found, stay in place
    return GridPosition(start_x, start_y, 0);
}

void planner(
    int* map,
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
)
{
    // setup environment parameters
    globalState.gridWidth = x_size;
    globalState.gridHeight = y_size;
    globalState.obstacleThreshold = collision_thresh;
    globalState.terrainMap = map;
   
    // allocate timing array and setup goal candidates
    globalState.timeSteps = new int[x_size * y_size];
    vector<tuple<int, int, double, int>> candidate_interceptions;
    
    static int selected_goal_x, selected_goal_y, time_horizon;
    
    // perform goal selection on first call
    if (curr_time == 0) {
        globalState.initialized = true;
        
        // compute heuristic from robot to all cells
        globalState.robotToAllCosts = backwardHeuristicSearch(robotposeX, robotposeY, curr_time, target_steps);
        
        // analyze trajectory for optimal interception points
        for (int i = 0; i < target_steps; i++) {
            int pos_x = target_traj[i];
            int pos_y = target_traj[i + target_steps];
            int cell_index = GETMAPINDEX(pos_x, pos_y, x_size, y_size);
            
            bool reachable = (globalState.timeSteps[cell_index] <= i);
            bool valid_path = (globalState.robotToAllCosts[cell_index] > -1);
            
            if (reachable && valid_path) {
                int delay_time = i - globalState.timeSteps[cell_index];
                int wait_cost = delay_time * map[cell_index];
                double total_cost = globalState.robotToAllCosts[cell_index] + wait_cost;
                
                candidate_interceptions.push_back(make_tuple(pos_x, pos_y, total_cost, i));
            }
        }
        
        // select minimum cost interception from candidates
        auto best_option = *min_element(candidate_interceptions.begin(), candidate_interceptions.end(),
            [](const auto& a, const auto& b) { return get<2>(a) < get<2>(b); });
        
        selected_goal_x = get<0>(best_option);
        selected_goal_y = get<1>(best_option);
        globalState.selectedGoalX = selected_goal_x;
        globalState.selectedGoalY = selected_goal_y;
        
        int goal_cell = GETMAPINDEX(selected_goal_x, selected_goal_y, x_size, y_size);
        time_horizon = globalState.timeSteps[goal_cell];
        globalState.maxTimeHorizon = time_horizon;
        
        globalState.goalToAllCosts = backwardHeuristicSearch(selected_goal_x, selected_goal_y, curr_time, time_horizon);
    }
   
   
    // handle destination reached case
    if ((robotposeX == selected_goal_x) && (robotposeY == selected_goal_y)) {
        action_ptr[0] = robotposeX;
        action_ptr[1] = robotposeY;
        return;
    }
    
    // compute next robot movement using pathfinding
    GridPosition robot_next = AStarSearch(robotposeX, robotposeY, selected_goal_x, selected_goal_y, curr_time, time_horizon);

    // set action outputs
    action_ptr[0] = robot_next.x;
    action_ptr[1] = robot_next.y;
    
    return;
}