#include <ros/ros.h>
#include <ros/package.h>

#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseArray.h>

#include <yaml-cpp/yaml.h>

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "pursuit_planner/planner.h"

#ifndef MAPS_DIR
#define MAPS_DIR "maps"
#endif
#ifndef OUTPUT_DIR
#define OUTPUT_DIR "output"
#endif

// ============================================================================
// Dynamic obstacle types & helpers (adapted from runtest.cpp)
// ============================================================================

struct Waypoint {
    double x, y;
    int duration;
};

// Footprint struct is defined in planner.h

struct DynamicObstacle {
    std::string id;
    Footprint footprint;
    std::vector<Waypoint> waypoints;
};

static std::vector<DynamicObstacle> loadDynamicObstacles(const std::string& yaml_path)
{
    std::vector<DynamicObstacle> obstacles;

    // Check if file exists; if not, just return empty
    std::ifstream test_file(yaml_path);
    if (!test_file.good()) {
        ROS_WARN_STREAM("No dynamic obstacles file found at: " << yaml_path);
        return obstacles;
    }
    test_file.close();

    YAML::Node config = YAML::LoadFile(yaml_path);
    if (!config["dynamic_obstacles"]) {
        return obstacles;
    }

    for (const auto& entry : config["dynamic_obstacles"]) {
        DynamicObstacle ob;
        ob.id = entry["id"].as<std::string>();

        // footprint
        auto fp = entry["footprint"];
        ob.footprint.kind = fp["kind"].as<std::string>();

        if (ob.footprint.kind == "circle") {
            ob.footprint.radius = fp["radius"].as<double>();
        } else if (ob.footprint.kind == "box") {
            ob.footprint.width  = fp["width"].as<int>();
            ob.footprint.height = fp["height"].as<int>();
        }

        // waypoints
        for (const auto& w : entry["waypoints"]) {
            Waypoint wp;
            wp.x        = w["x"].as<double>();
            wp.y        = w["y"].as<double>();
            wp.duration = w["duration"].as<int>();
            ob.waypoints.push_back(wp);
        }

        obstacles.push_back(ob);
    }

    return obstacles;
}

// For each obstacle: simulate its motion for t = 0..max_t
static std::vector<std::vector<std::pair<int,int>>>
simulateObstacle(const DynamicObstacle& ob, int max_t)
{
    std::vector<std::vector<std::pair<int,int>>> traj(max_t + 1);

    if (ob.waypoints.empty()) {
        return traj;
    }

    double x = ob.waypoints[0].x;
    double y = ob.waypoints[0].y;
    int t = 0;

    // First waypoint: stay for duration
    for (int k = 0; k < ob.waypoints[0].duration && t <= max_t; ++k) {
        traj[t++] = { { (int)std::round(x), (int)std::round(y) } };
    }

    for (size_t i = 1; i < ob.waypoints.size(); ++i) {
        const auto& prev = ob.waypoints[i - 1];
        const auto& wp   = ob.waypoints[i];

        int D = wp.duration;
        if (D == 0) {
            x = wp.x;
            y = wp.y;
            if (t <= max_t) {
                traj[t] = { { (int)std::round(x), (int)std::round(y) } };
            }
            continue;
        }

        double dx = (wp.x - prev.x) / D;
        double dy = (wp.y - prev.y) / D;

        for (int k = 0; k < D && t <= max_t; ++k) {
            traj[t] = { { (int)std::round(x), (int)std::round(y) } };
            x += dx;
            y += dy;
            ++t;
        }
    }

    // Pad remaining time with final pose
    while (t <= max_t) {
        traj[t] = { { (int)std::round(x), (int)std::round(y) } };
        ++t;
    }

    return traj;
}

// 3D occupancy grid (unused by planner now, but kept for completeness)
static std::vector<std::vector<std::vector<bool>>>
buildDynamicOccupancyGrid(
    const std::vector<DynamicObstacle>& obstacles,
    int x_size, int y_size, int max_t)
{
    std::vector<std::vector<std::vector<bool>>> occ(
        max_t + 1,
        std::vector<std::vector<bool>>(
            y_size + 1, std::vector<bool>(x_size + 1, false)));

    for (const auto& ob : obstacles) {
        auto traj = simulateObstacle(ob, max_t);

        for (int t = 0; t <= max_t; ++t) {
            if (traj[t].empty()) continue;
            int px = traj[t][0].first;
            int py = traj[t][0].second;

            if (ob.footprint.kind == "point") {
                if (px >= 1 && px <= x_size && py >= 1 && py <= y_size)
                    occ[t][py][px] = true;
            }
            else if (ob.footprint.kind == "circle") {
                int R = (int)std::ceil(ob.footprint.radius);
                for (int dx = -R; dx <= R; ++dx) {
                    for (int dy = -R; dy <= R; ++dy) {
                        if (dx*dx + dy*dy <= R*R) {
                            int nx = px + dx;
                            int ny = py + dy;
                            if (nx >= 1 && nx <= x_size &&
                                ny >= 1 && ny <= y_size)
                            {
                                occ[t][ny][nx] = true;
                            }
                        }
                    }
                }
            }
            else if (ob.footprint.kind == "box") {
                for (int dx = -ob.footprint.width / 2;
                     dx <=  ob.footprint.width / 2; ++dx)
                {
                    for (int dy = -ob.footprint.height / 2;
                         dy <=  ob.footprint.height / 2; ++dy)
                    {
                        int nx = px + dx;
                        int ny = py + dy;
                        if (nx >= 1 && nx <= x_size &&
                            ny >= 1 && ny <= y_size)
                        {
                            occ[t][ny][nx] = true;
                        }
                    }
                }
            }
        }
    }

    return occ;
}

using ObstacleTraj = std::vector<std::vector<std::pair<int,int>>>;

// ============================================================================
// PlannerNode: ROS wrapper around runtest.cpp behavior
// ============================================================================

class PlannerNode {
public:
    PlannerNode(ros::NodeHandle& nh)
        : nh_(nh),
          map_(nullptr),
          target_traj_(nullptr),
          action_ptr_(nullptr),
          x_size_(0),
          y_size_(0),
          collision_thresh_(0),
          robotposeX_(0),
          robotposeY_(0),
          curr_time_(0),
          target_steps_(0),
          goalX_(0),
          goalY_(0),
          numofmoves_(0),
          caught_(false),
          pathcost_(0),
          finished_(false)
    {
        // Get map file + dyno yaml from params (with defaults)
        std::string default_map =
            ros::package::getPath("pursuit_planner") + "/maps/map11.txt";
        std::string default_yaml =
            ros::package::getPath("pursuit_planner") + "/config/dyno_map11.yaml";

        nh_.param<std::string>("map_file",  map_file_path_,  default_map);
        nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, default_yaml);

        // Latched publishers for final trajectories
        robot_path_pub_  = nh_.advertise<geometry_msgs::PoseArray>("robot_path",  1, true);
        target_path_pub_ = nh_.advertise<geometry_msgs::PoseArray>("target_path", 1, true);
        // obstacle_traj_pub_map_ will be filled after we know obstacle IDs

        if (!loadProblemFromFile()) {
            ROS_FATAL("PlannerNode: failed to load problem from file.");
            finished_ = true;
            return;
        }

        // allocate action_ptr once
        action_ptr_ = new int[2];

        // Seed robot_path_ with initial pose
        geometry_msgs::Pose2D start_pose;
        start_pose.x = robotposeX_;
        start_pose.y = robotposeY_;
        start_pose.theta = 0.0;
        robot_path_.push_back(start_pose);

        ROS_INFO("PlannerNode initialized from file. Ready to run planner.");
    }

    ~PlannerNode() {
        if (target_traj_) delete[] target_traj_;
        if (map_) delete[] map_;
        if (action_ptr_) delete[] action_ptr_;
    }

    bool isFinished() const { return finished_; }

    // One iteration of the planner loop (mirrors runtest.cpp while-body)
    void spinOnce()
    {
        if (finished_) return;

        auto start = std::chrono::high_resolution_clock::now();

        // Static goal (final target waypoint)
        int targetposeX = goalX_;
        int targetposeY = goalY_;

        // Call planner with static goal and dynamic obstacles
        planner(map_,
                compactObs_,
                collision_thresh_,
                x_size_,
                y_size_,
                robotposeX_,
                robotposeY_,
                target_steps_,
                target_traj_,
                targetposeX,
                targetposeY,
                curr_time_,
                action_ptr_);

        int newrobotposeX = action_ptr_[0];
        int newrobotposeY = action_ptr_[1];

        ROS_INFO_STREAM("Planner selected next waypoint: ("
                        << newrobotposeX << "," << newrobotposeY << ")");

        // Validity checks (same as runtest.cpp)
        if (newrobotposeX < 1 || newrobotposeX > x_size_ ||
            newrobotposeY < 1 || newrobotposeY > y_size_)
        {
            ROS_ERROR("ERROR: out-of-map robot position commanded");
            finished_ = true;
            return;
        }

        if (map_[(newrobotposeY-1)*x_size_ + newrobotposeX-1] >= collision_thresh_) {
            ROS_ERROR("ERROR: planned action leads to collision");
            finished_ = true;
            return;
        }

        if (std::abs(robotposeX_ - newrobotposeX) > 1 ||
            std::abs(robotposeY_ - newrobotposeY) > 1)
        {
            ROS_ERROR("ERROR: invalid action commanded. robot must move on 8-connected grid.");
            finished_ = true;
            return;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        int movetime = std::max(1, (int)std::ceil(duration));

        if (newrobotposeX == robotposeX_ && newrobotposeY == robotposeY_) {
            numofmoves_ -= 1;
        }

        if (curr_time_ + movetime >= target_steps_) {
            finished_ = true;
            ROS_INFO("Reached end of time horizon (target_steps_)");
            printResult();
            return;
        }

        // Dynamic obstacle collision check at arrival time
        int arrival_time = curr_time_ + movetime;
        if (compactObs_.isOccupied(newrobotposeX, newrobotposeY, arrival_time)) {
            ROS_ERROR_STREAM("ERROR: Robot will collide with dynamic obstacle at t="
                             << arrival_time << " pos=("
                             << newrobotposeX << "," << newrobotposeY << ")");
            finished_ = true;
            return;
        }

        curr_time_  = arrival_time;
        numofmoves_ = numofmoves_ + 1;
        pathcost_   = pathcost_ + movetime * map_[(robotposeY_-1)*x_size_ + robotposeX_-1];

        int prevRobotX = robotposeX_;
        int prevRobotY = robotposeY_;
        robotposeX_    = newrobotposeX;
        robotposeY_    = newrobotposeY;

        // Record robot path waypoint (with heading)
        geometry_msgs::Pose2D robot_wp;
        robot_wp.x = robotposeX_;
        robot_wp.y = robotposeY_;
        robot_wp.theta = computeRobotHeading(prevRobotX, prevRobotY,
                                             robotposeX_, robotposeY_);
        robot_path_.push_back(robot_wp);

        // Goal check w.r.t static goal
        float thresh = 0.5f;
        if (std::abs(robotposeX_ - goalX_) <= thresh &&
            std::abs(robotposeY_ - goalY_) <= thresh)
        {
            caught_   = true;
            finished_ = true;
            printResult();
        }
    }

    // Publish all trajectories once, as latched PoseArray topics
    void publishTrajectoriesOnce()
    {
        // 1) Robot path
        geometry_msgs::PoseArray robot_arr;
        robot_arr.header.stamp = ros::Time::now();
        robot_arr.header.frame_id = "map"; // adjust if needed

        for (const auto& p2d : robot_path_) {
            geometry_msgs::Pose p;
            p.position.x = p2d.x;
            p.position.y = p2d.y;
            p.position.z = 0.0;

            // Simple yaw-only quaternion from theta
            double yaw  = p2d.theta;
            double half = 0.5 * yaw;
            double cz   = std::cos(half);
            double sz   = std::sin(half);
            p.orientation.x = 0.0;
            p.orientation.y = 0.0;
            p.orientation.z = sz;
            p.orientation.w = cz;

            robot_arr.poses.push_back(p);
        }

        robot_path_pub_.publish(robot_arr);
        ROS_INFO_STREAM("Published robot_path with " << robot_arr.poses.size() << " waypoints");

        // 2) Target path (static list from map file)
        geometry_msgs::PoseArray target_arr;
        target_arr.header = robot_arr.header;
        for (const auto& p2d : target_path_) {
            geometry_msgs::Pose p;
            p.position.x = p2d.x;
            p.position.y = p2d.y;
            p.position.z = 0.0;
            p.orientation.x = 0.0;
            p.orientation.y = 0.0;
            p.orientation.z = 0.0;
            p.orientation.w = 1.0;
            target_arr.poses.push_back(p);
        }

        target_path_pub_.publish(target_arr);
        ROS_INFO_STREAM("Published target_path with " << target_arr.poses.size() << " waypoints");

        // 3) Dynamic obstacle trajectories
        for (size_t i = 0; i < obstacles_.size(); ++i) {
            const auto& ob   = obstacles_[i];
            const auto& traj = obstacle_trajs_[i];

            geometry_msgs::PoseArray arr;
            arr.header = robot_arr.header;

            // One center pose per time-step
            for (size_t t = 0; t < traj.size(); ++t) {
                if (traj[t].empty()) continue;
                int ox = traj[t][0].first;
                int oy = traj[t][0].second;

                geometry_msgs::Pose p;
                p.position.x = ox;
                p.position.y = oy;
                p.position.z = 0.0;
                p.orientation.x = 0.0;
                p.orientation.y = 0.0;
                p.orientation.z = 0.0;
                p.orientation.w = 1.0;

                arr.poses.push_back(p);
            }

            auto it = obstacle_traj_pub_map_.find(ob.id);
            if (it != obstacle_traj_pub_map_.end()) {
                it->second.publish(arr);
                ROS_INFO_STREAM("Published trajectory for obstacle " << ob.id
                                << " with " << arr.poses.size() << " waypoints");
            }
        }
    }

private:
    ros::NodeHandle nh_;

    // Trajectory publishers (latched)
    ros::Publisher  robot_path_pub_;
    ros::Publisher  target_path_pub_;
    std::map<std::string, ros::Publisher> obstacle_traj_pub_map_;

    // dynamic obstacles
    std::vector<DynamicObstacle> obstacles_;
    std::vector<ObstacleTraj>    obstacle_trajs_;

    // problem data
    int* map_;
    int* target_traj_;
    CompactDynamicObstacles compactObs_;

    int x_size_, y_size_;
    int collision_thresh_;
    int robotposeX_, robotposeY_;
    int curr_time_;
    int target_steps_;
    int goalX_, goalY_;
    int* action_ptr_;

    int numofmoves_;
    bool caught_;
    int pathcost_;
    bool finished_;

    std::string map_file_path_;
    std::string dyno_yaml_path_;

    // Paths stored as Pose2D for convenience
    std::vector<geometry_msgs::Pose2D> robot_path_;
    std::vector<geometry_msgs::Pose2D> target_path_;

    double computeRobotHeading(int oldX, int oldY, int newX, int newY)
    {
        int dx = newX - oldX;
        int dy = newY - oldY;
        if (dx == 0 && dy == 0) {
            return 0.0;
        }
        return std::atan2(dy, dx);   // radians
    }

    // Convert from obstacle data structures to CompactDynamicObstacles
    CompactDynamicObstacles convertToCompactObstacles(
        const std::vector<DynamicObstacle>& obstacles,
        const std::vector<ObstacleTraj>& obstacle_trajs,
        int max_t)
    {
        CompactDynamicObstacles compact;
        compact.max_time = max_t;
        compact.timesteps.resize(max_t + 1);

        for (size_t obs_idx = 0; obs_idx < obstacles.size(); ++obs_idx) {
            const auto& ob   = obstacles[obs_idx];
            const auto& traj = obstacle_trajs[obs_idx];

            for (int t = 0; t <= max_t; ++t) {
                if (t >= (int)traj.size() || traj[t].empty()) continue;

                ObstacleState state;
                state.x = traj[t][0].first;
                state.y = traj[t][0].second;
                state.footprint.kind   = ob.footprint.kind;
                state.footprint.radius = ob.footprint.radius;
                state.footprint.width  = ob.footprint.width;
                state.footprint.height = ob.footprint.height;

                compact.timesteps[t].push_back(state);
            }
        }

        return compact;
    }

    bool loadProblemFromFile()
    {
        ROS_INFO_STREAM("Reading problem definition from: " << map_file_path_);

        std::ifstream myfile(map_file_path_);
        if (!myfile.is_open()) {
            ROS_ERROR_STREAM("Failed to open the file: " << map_file_path_);
            return false;
        }

        char letter;
        std::string line;

        // N x y
        myfile >> letter;
        if (letter != 'N') {
            ROS_ERROR("Error parsing file: expected 'N'");
            return false;
        }
        myfile >> x_size_ >> letter >> y_size_;
        ROS_INFO_STREAM("map size: " << x_size_ << "x" << y_size_);

        // C collision_thresh
        myfile >> letter;
        if (letter != 'C') {
            ROS_ERROR("Error parsing file: expected 'C'");
            return false;
        }
        myfile >> collision_thresh_;
        ROS_INFO_STREAM("collision threshold: " << collision_thresh_);

        // R robotposeX robotposeY
        myfile >> letter;
        if (letter != 'R') {
            ROS_ERROR("Error parsing file: expected 'R'");
            return false;
        }
        myfile >> robotposeX_ >> letter >> robotposeY_;
        ROS_INFO_STREAM("robot pose: " << robotposeX_ << "x" << robotposeY_);

        // read trajectory (T ... M)
        std::vector<std::vector<int>> traj;
        std::getline(myfile, line); // consume end of line after R line

        do {
            std::getline(myfile, line);
        } while (line != "T" && myfile.good());

        while (std::getline(myfile, line) && line != "M") {
            std::stringstream ss(line);
            int num1, num2;
            ss >> num1 >> letter >> num2;
            traj.push_back({num1, num2});
        }

        target_steps_ = traj.size();
        target_traj_  = new int[2 * target_steps_];
        for (size_t i = 0; i < target_steps_; ++i) {
            target_traj_[i]                 = traj[i][0];
            target_traj_[i + target_steps_] = traj[i][1];
        }
        ROS_INFO_STREAM("target_steps: " << target_steps_);

        // Static goal from final target position (like runtest.cpp)
        goalX_ = target_traj_[target_steps_ - 1];
        goalY_ = target_traj_[2 * target_steps_ - 1];
        ROS_INFO_STREAM("Static goal: (" << goalX_ << "," << goalY_ << ")");

        // Build target_path_ as static list
        target_path_.clear();
        for (int k = 0; k < target_steps_; ++k) {
            geometry_msgs::Pose2D p;
            p.x = target_traj_[k];
            p.y = target_traj_[k + target_steps_];
            p.theta = 0.0;
            target_path_.push_back(p);
        }

        // load dynamic obstacles + compact representation
        auto obstacles = loadDynamicObstacles(dyno_yaml_path_);
        ROS_INFO_STREAM("Loaded " << obstacles.size() << " dynamic obstacles from " << dyno_yaml_path_);
        obstacles_ = obstacles;

        obstacle_trajs_.clear();
        for (const auto& ob : obstacles_) {
            obstacle_trajs_.push_back(simulateObstacle(ob, target_steps_));
        }

        compactObs_ = convertToCompactObstacles(obstacles_, obstacle_trajs_, target_steps_);

        // Create latched publishers for dynamic obstacle trajectories
        obstacle_traj_pub_map_.clear();
        for (const auto& ob : obstacles_) {
            std::string topic = "dynamic_obstacles/" + ob.id + "/trajectory";
            obstacle_traj_pub_map_[ob.id] =
                nh_.advertise<geometry_msgs::PoseArray>(topic, 1, true);
            ROS_INFO_STREAM("Advertising dynamic obstacle trajectory topic: " << topic);
        }

        // read map (M section just finished)
        map_ = new int[x_size_ * y_size_];
        for (size_t i = 0; i < (size_t)x_size_; i++) {
            std::getline(myfile, line);
            std::stringstream ss(line);
            for (size_t j = 0; j < (size_t)y_size_; j++) {
                double value;
                ss >> value;
                map_[j * x_size_ + i] = (int)value;
                if (j != (size_t)y_size_ - 1) ss.ignore();
            }
        }

        myfile.close();
        curr_time_ = 0;
        return true;
    }

    void printResult()
    {
        ROS_INFO_STREAM("\nRESULT");
        ROS_INFO_STREAM("goal reached = " << (caught_ ? "true" : "false"));
        ROS_INFO_STREAM("time taken (s) = " << curr_time_);
        ROS_INFO_STREAM("moves made = " << numofmoves_);
        ROS_INFO_STREAM("path cost = " << pathcost_);
    }
};

// ============================================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pursuit_planner_runtest_node");
    ros::NodeHandle nh("~");

    PlannerNode node(nh);
    ros::Rate rate(1);   // IMPORTANT: slow this down

    while (ros::ok() && !node.isFinished()) {
        ros::spinOnce();
        node.spinOnce();   // one planner step
        rate.sleep();      // gives memory time to release
    }
    if (ros::ok()) {
        node.publishTrajectoriesOnce();
        ROS_INFO("Trajectories published. Node will keep spinning so latched topics remain available.");
        ros::spin();
    }

    return 0;
}
