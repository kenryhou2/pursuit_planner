/*=================================================================
 * planner.cpp - Hierarchical Waypoint-Based Planner
 *=================================================================*/
#include "../include/pursuit_planner/planner.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <queue>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <cmath>

using namespace std;

#define GETMAPINDEX(X, Y, XSIZE, YSIZE) ((Y-1)*XSIZE + (X-1))

#if !defined(MAX)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

#define NUMOFDIRS 8
static const double SQRT2 = 1.41421356237;

int dX[NUMOFDIRS + 1] = {-1, -1, -1,  0,  0,  1, 1, 1, 0};
int dY[NUMOFDIRS + 1] = {-1,  0,  1, -1,  1, -1, 0, 1, 0};

double movementCost(int dir) {
    if (dir == 0 || dir == 2 || dir == 5 || dir == 7) {
        return SQRT2;
    }
    return 1.0;
}

struct Waypoint {
    int x, y;
    int estimated_arrival_time;
    bool is_detour;
    
    Waypoint() : x(0), y(0), estimated_arrival_time(0), is_detour(false) {}
    Waypoint(int px, int py, int t, bool detour = false) 
        : x(px), y(py), estimated_arrival_time(t), is_detour(detour) {}
};

struct PlannerState {
    CompactDynamicObstacles compactObstacles;
    int gridWidth, gridHeight, obstacleThreshold;
    int* terrainMap;
    vector<Waypoint> waypoints;
    int currentWaypointIdx;
    vector<pair<int,int>> globalPath;
    int selectedGoalX, selectedGoalY;
    int maxTimeHorizon;
    double waypointRadius;
    bool initialized;
    
    PlannerState() : initialized(false), currentWaypointIdx(0), waypointRadius(4.0) {}
};

static PlannerState globalState;

inline bool isValidCell(int x, int y) {
    return (x > 0) && (x <= globalState.gridWidth) && 
           (y > 0) && (y <= globalState.gridHeight);
}

inline bool isTraversable(int x, int y) {
    if (!isValidCell(x, y)) return false;
    int idx = GETMAPINDEX(x, y, globalState.gridWidth, globalState.gridHeight);
    return globalState.terrainMap[idx] < globalState.obstacleThreshold;
}

double euclideanDist(int x1, int y1, int x2, int y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}

struct DijkstraNode {
    int x, y;
    double cost;
    
    DijkstraNode(int px, int py, double c) : x(px), y(py), cost(c) {}
    
    bool operator>(const DijkstraNode& other) const {
        return cost > other.cost;
    }
};

vector<pair<int,int>> globalPathSearch(int startX, int startY, int goalX, int goalY) {
    const int total_cells = globalState.gridWidth * globalState.gridHeight;
    
    vector<double> dist(total_cells, -1);
    vector<pair<int,int>> parent(total_cells, {-1, -1});
    
    priority_queue<DijkstraNode, vector<DijkstraNode>, greater<DijkstraNode>> pq;
    
    int start_idx = GETMAPINDEX(startX, startY, globalState.gridWidth, globalState.gridHeight);
    dist[start_idx] = 0;
    pq.push(DijkstraNode(startX, startY, 0));
    
    while (!pq.empty()) {
        DijkstraNode curr = pq.top();
        pq.pop();
        
        if (curr.x == goalX && curr.y == goalY) break;
        
        int curr_idx = GETMAPINDEX(curr.x, curr.y, globalState.gridWidth, globalState.gridHeight);
        if (dist[curr_idx] >= 0 && curr.cost > dist[curr_idx]) continue;
        
        for (int dir = 0; dir < 8; dir++) {
            int nx = curr.x + dX[dir];
            int ny = curr.y + dY[dir];
            
            if (!isTraversable(nx, ny)) continue;
            
            int next_idx = GETMAPINDEX(nx, ny, globalState.gridWidth, globalState.gridHeight);
            double terrain_cost = globalState.terrainMap[next_idx];
            double move_dist = movementCost(dir);
            double new_cost = curr.cost + terrain_cost * move_dist;
            
            if (dist[next_idx] < 0 || new_cost < dist[next_idx]) {
                dist[next_idx] = new_cost;
                parent[next_idx] = {curr.x, curr.y};
                pq.push(DijkstraNode(nx, ny, new_cost));
            }
        }
    }
    
    vector<pair<int,int>> path;
    int goal_idx = GETMAPINDEX(goalX, goalY, globalState.gridWidth, globalState.gridHeight);
    
    if (dist[goal_idx] < 0) {
        cout << "WARNING: No global path found!" << endl;
        return path;
    }
    
    int cx = goalX, cy = goalY;
    while (cx != -1 && cy != -1) {
        path.push_back({cx, cy});
        int idx = GETMAPINDEX(cx, cy, globalState.gridWidth, globalState.gridHeight);
        auto p = parent[idx];
        cx = p.first;
        cy = p.second;
    }
    
    reverse(path.begin(), path.end());
    return path;
}

vector<Waypoint> extractWaypoints(const vector<pair<int,int>>& path, double N) {
    vector<Waypoint> waypoints;
    if (path.empty()) return waypoints;
    
    double accumulated_dist = 0.0;
    int estimated_time = 0;
    
    waypoints.push_back(Waypoint(path[0].first, path[0].second, 0, false));
    
    for (size_t i = 1; i < path.size(); i++) {
        double dx = path[i].first - path[i-1].first;
        double dy = path[i].second - path[i-1].second;
        double step_dist = sqrt(dx*dx + dy*dy);
        
        accumulated_dist += step_dist;
        estimated_time++;
        
        if (accumulated_dist >= N) {
            waypoints.push_back(Waypoint(path[i].first, path[i].second, estimated_time, false));
            accumulated_dist = 0.0;
        }
    }
    
    if (waypoints.back().x != path.back().first || waypoints.back().y != path.back().second) {
        waypoints.push_back(Waypoint(path.back().first, path.back().second, path.size() - 1, false));
    }
    
    return waypoints;
}

vector<Waypoint> findDetourWaypoints(int fromX, int fromY, int blockedX, int blockedY, 
                                      int toX, int toY, int curr_time, double visibility_range) {
    vector<Waypoint> detours;
    
    double pathDx = blockedX - fromX;
    double pathDy = blockedY - fromY;
    double pathLen = sqrt(pathDx * pathDx + pathDy * pathDy);
    
    if (pathLen < 0.001) {
        pathDx = toX - blockedX;
        pathDy = toY - blockedY;
        pathLen = sqrt(pathDx * pathDx + pathDy * pathDy);
    }
    
    if (pathLen < 0.001) return detours;
    
    double perpX = -pathDy / pathLen;
    double perpY = pathDx / pathLen;
    
    double detour_dist = visibility_range * 0.5;
    
    int detour1X = (int)round(blockedX + perpX * detour_dist);
    int detour1Y = (int)round(blockedY + perpY * detour_dist);
    
    int detour2X = (int)round(blockedX - perpX * detour_dist);
    int detour2Y = (int)round(blockedY - perpY * detour_dist);
    
    int estimated_arrival = curr_time + (int)ceil(euclideanDist(fromX, fromY, blockedX, blockedY));
    
    auto isDetourValid = [&](int dx, int dy) -> bool {
        if (!isValidCell(dx, dy)) return false;
        if (!isTraversable(dx, dy)) return false;
        
        if (globalState.compactObstacles.timesteps.size() > 0) {
            for (int t = estimated_arrival - 2; t <= estimated_arrival + 3; t++) {
                if (t >= 0 && globalState.compactObstacles.isOccupied(dx, dy, t)) {
                    return false;
                }
            }
        }
        return true;
    };
    
    bool side1_valid = isDetourValid(detour1X, detour1Y);
    bool side2_valid = isDetourValid(detour2X, detour2Y);
    
    double dist1_to_goal = euclideanDist(detour1X, detour1Y, toX, toY);
    double dist2_to_goal = euclideanDist(detour2X, detour2Y, toX, toY);
    
    int chosenX, chosenY;
    if (side1_valid && side2_valid) {
        if (dist1_to_goal < dist2_to_goal) {
            chosenX = detour1X;
            chosenY = detour1Y;
        } else {
            chosenX = detour2X;
            chosenY = detour2Y;
        }
    } else if (side1_valid) {
        chosenX = detour1X;
        chosenY = detour1Y;
    } else if (side2_valid) {
        chosenX = detour2X;
        chosenY = detour2Y;
    } else {
        detour_dist = visibility_range * 0.25;
        detour1X = (int)round(blockedX + perpX * detour_dist);
        detour1Y = (int)round(blockedY + perpY * detour_dist);
        detour2X = (int)round(blockedX - perpX * detour_dist);
        detour2Y = (int)round(blockedY - perpY * detour_dist);
        
        side1_valid = isDetourValid(detour1X, detour1Y);
        side2_valid = isDetourValid(detour2X, detour2Y);
        
        if (side1_valid) {
            chosenX = detour1X;
            chosenY = detour1Y;
        } else if (side2_valid) {
            chosenX = detour2X;
            chosenY = detour2Y;
        } else {
            return detours;
        }
    }
    
    int detour_arrival = curr_time + (int)ceil(euclideanDist(fromX, fromY, chosenX, chosenY));
    detours.push_back(Waypoint(chosenX, chosenY, detour_arrival, true));
    
    return detours;
}

struct TimeSpaceNode {
    int x, y, time;
    double g_cost, f_cost;
    
    TimeSpaceNode(int px, int py, int t, double g, double f)
        : x(px), y(py), time(t), g_cost(g), f_cost(f) {}
    
    bool operator>(const TimeSpaceNode& other) const {
        return f_cost > other.f_cost;
    }
};

struct StateHash {
    size_t operator()(const tuple<int,int,int>& state) const {
        auto h1 = hash<int>{}(get<0>(state));
        auto h2 = hash<int>{}(get<1>(state));
        auto h3 = hash<int>{}(get<2>(state));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

pair<int,int> segmentAStarSearch(int startX, int startY, int goalX, int goalY, 
                                  int current_time, int time_limit) {
    
    auto heuristic = [goalX, goalY](int x, int y) -> double {
        return euclideanDist(x, y, goalX, goalY);
    };
    
    priority_queue<TimeSpaceNode, vector<TimeSpaceNode>, greater<TimeSpaceNode>> open_set;
    unordered_map<tuple<int,int,int>, double, StateHash> g_scores;
    unordered_map<tuple<int,int,int>, tuple<int,int,int>, StateHash> came_from;
    
    // Reserve space to avoid repeated reallocations
    int estimated_states = globalState.gridWidth * globalState.gridHeight * min(10, time_limit - current_time);
    g_scores.reserve(estimated_states);
    came_from.reserve(estimated_states);
    
    double start_h = heuristic(startX, startY);
    open_set.push(TimeSpaceNode(startX, startY, current_time, 0, start_h));
    g_scores[{startX, startY, current_time}] = 0;
    
    int goal_x = -1, goal_y = -1, goal_t = -1;
    bool found = false;
    // Limit iterations based on state space size to prevent memory explosion
    int max_states = globalState.gridWidth * globalState.gridHeight * (time_limit - current_time);
    int max_iterations = min(max_states * 3, 50000);  // Cap at 50k to prevent memory issues
    int iterations = 0;
    
    while (!open_set.empty() && iterations < max_iterations) {
        iterations++;
        
        TimeSpaceNode curr = open_set.top();
        open_set.pop();
        
        if (curr.x == goalX && curr.y == goalY) {
            goal_x = curr.x;
            goal_y = curr.y;
            goal_t = curr.time;
            found = true;
            break;
        }
        
        if (curr.time >= time_limit) continue;
        
        auto curr_state = make_tuple(curr.x, curr.y, curr.time);
        if (g_scores.count(curr_state) && curr.g_cost > g_scores[curr_state]) continue;
        
        for (int dir = 0; dir < 9; dir++) {
            int nx = curr.x + dX[dir];
            int ny = curr.y + dY[dir];
            int nt = curr.time + 1;
            
            if (!isValidCell(nx, ny)) continue;
            if (!isTraversable(nx, ny)) continue;
            
            if (globalState.compactObstacles.timesteps.size() > 0) {
                if (globalState.compactObstacles.isOccupied(nx, ny, nt)) continue;
            }
            
            int terrain_idx = GETMAPINDEX(nx, ny, globalState.gridWidth, globalState.gridHeight);
            double terrain_cost = globalState.terrainMap[terrain_idx];
            double move_cost = (dir == 8) ? 1.0 : movementCost(dir);
            double new_g = curr.g_cost + terrain_cost * move_cost;
            
            auto next_state = make_tuple(nx, ny, nt);
            
            if (!g_scores.count(next_state) || new_g < g_scores[next_state]) {
                g_scores[next_state] = new_g;
                came_from[next_state] = make_tuple(curr.x, curr.y, curr.time);
                double h = heuristic(nx, ny);
                open_set.push(TimeSpaceNode(nx, ny, nt, new_g, new_g + h));
            }
        }
    }
    
    if (!found) {
        cout << "  Segment A* failed" << endl;
        return {startX, startY};
    }
    
    vector<tuple<int,int,int>> path;
    auto state = make_tuple(goal_x, goal_y, goal_t);
    
    while (came_from.count(state)) {
        path.push_back(state);
        state = came_from[state];
    }
    path.push_back(state);
    reverse(path.begin(), path.end());
    
    if (path.size() < 2) return {startX, startY};
    return {get<0>(path[1]), get<1>(path[1])};
}

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
)
{
    const double WAYPOINT_DISTANCE = x_size / 10.0;
    
    globalState.gridWidth = x_size;
    globalState.gridHeight = y_size;
    globalState.obstacleThreshold = collision_thresh;
    globalState.terrainMap = map;
    globalState.compactObstacles = compactObs;
    globalState.waypointRadius = WAYPOINT_DISTANCE;
    
    if (curr_time == 0) {
        globalState.initialized = true;
        globalState.currentWaypointIdx = 0;
        
        cout << "\n=== HIERARCHICAL WAYPOINT PLANNER ===" << endl;
        cout << "Robot: (" << robotposeX << ", " << robotposeY << "), N=" << WAYPOINT_DISTANCE << endl;
        
        int goal_x = target_traj[target_steps - 1];
        int goal_y = target_traj[target_steps - 1 + target_steps];
        
        globalState.selectedGoalX = goal_x;
        globalState.selectedGoalY = goal_y;
        globalState.maxTimeHorizon = target_steps - 1;
        
        cout << "Goal: (" << goal_x << ", " << goal_y << ") [final target position]" << endl;
        
        globalState.globalPath = globalPathSearch(robotposeX, robotposeY, goal_x, goal_y);
        
        if (globalState.globalPath.empty()) {
            action_ptr[0] = robotposeX;
            action_ptr[1] = robotposeY;
            return;
        }
        
        cout << "Path length: " << globalState.globalPath.size() << endl;
        
        globalState.waypoints = extractWaypoints(globalState.globalPath, WAYPOINT_DISTANCE);
        
        cout << "Waypoints (" << globalState.waypoints.size() << "):" << endl;
        for (size_t i = 0; i < globalState.waypoints.size(); i++) {
            cout << "  WP" << i << ": (" << globalState.waypoints[i].x << "," 
                 << globalState.waypoints[i].y << ")" << endl;
        }
        
        ofstream wp_file("../output/waypoints.txt");
        if (wp_file.is_open()) {
            for (size_t i = 0; i < globalState.waypoints.size(); i++) {
                wp_file << globalState.waypoints[i].x << "," 
                        << globalState.waypoints[i].y << ","
                        << (globalState.waypoints[i].is_detour ? 1 : 0) << endl;
            }
            wp_file.close();
        }
        
        globalState.currentWaypointIdx = 1;
    }
    
    if (robotposeX == globalState.selectedGoalX && robotposeY == globalState.selectedGoalY) {
        action_ptr[0] = robotposeX;
        action_ptr[1] = robotposeY;
        return;
    }
    
    if (globalState.currentWaypointIdx < (int)globalState.waypoints.size()) {
        Waypoint& curr_wp = globalState.waypoints[globalState.currentWaypointIdx];
        double dist_to_wp = euclideanDist(robotposeX, robotposeY, curr_wp.x, curr_wp.y);
        if (dist_to_wp < 1.42) {
            globalState.currentWaypointIdx++;
        }
    }
    
    int target_wp_idx = globalState.currentWaypointIdx;
    target_wp_idx = MIN(target_wp_idx, (int)globalState.waypoints.size() - 1);
    
    Waypoint& target_wp = globalState.waypoints[target_wp_idx];
    
    int estimated_arrival = curr_time + (int)ceil(euclideanDist(robotposeX, robotposeY, target_wp.x, target_wp.y));
    bool blocked = false;
    
    if (globalState.compactObstacles.timesteps.size() > 0) {
        for (int t = estimated_arrival - 2; t <= estimated_arrival + 3; t++) {
            if (t >= 0 && globalState.compactObstacles.isOccupied(target_wp.x, target_wp.y, t)) {
                blocked = true;
                break;
            }
        }
    }
    
    if (blocked && !target_wp.is_detour) {
        int next_wp_idx = MIN(target_wp_idx + 1, (int)globalState.waypoints.size() - 1);
        Waypoint& next_wp = globalState.waypoints[next_wp_idx];
        
        vector<Waypoint> detours = findDetourWaypoints(
            robotposeX, robotposeY,
            target_wp.x, target_wp.y,
            next_wp.x, next_wp.y,
            curr_time, globalState.waypointRadius
        );
        
        if (!detours.empty()) {
            cout << "  WP" << target_wp_idx << " (" << target_wp.x << "," << target_wp.y 
                 << ") blocked at t~" << estimated_arrival << ", creating detour via ("
                 << detours[0].x << "," << detours[0].y << ")" << endl;
            
            globalState.waypoints[target_wp_idx] = detours[0];
            
            ofstream wp_file("../output/waypoints.txt");
            if (wp_file.is_open()) {
                for (size_t i = 0; i < globalState.waypoints.size(); i++) {
                    wp_file << globalState.waypoints[i].x << "," 
                            << globalState.waypoints[i].y << ","
                            << (globalState.waypoints[i].is_detour ? 1 : 0) << endl;
                }
                wp_file.close();
            }
        } else {
            cout << "  WP" << target_wp_idx << " blocked, no valid detour found, skipping" << endl;
            globalState.waypoints[target_wp_idx].is_detour = true;
            globalState.currentWaypointIdx++;
        }
    }
    
    target_wp_idx = MIN(target_wp_idx, (int)globalState.waypoints.size() - 1);
    Waypoint& final_target_wp = globalState.waypoints[target_wp_idx];
    
    int estimated_segment_time = (int)ceil(euclideanDist(robotposeX, robotposeY, final_target_wp.x, final_target_wp.y));
    int segment_time_limit = curr_time + estimated_segment_time * 3 + 20;
    
    pair<int,int> next_move = segmentAStarSearch(
        robotposeX, robotposeY, final_target_wp.x, final_target_wp.y,
        curr_time, segment_time_limit
    );
    
    action_ptr[0] = next_move.first;
    action_ptr[1] = next_move.second;
}
