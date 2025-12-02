#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/OccupancyGrid.h>   // not strictly needed anymore, can remove
#include <geometry_msgs/Pose2D.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>

#include <yaml-cpp/yaml.h>

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <fstream>

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

struct Footprint {
    std::string kind;  // "point", "circle", "box"
    double radius = 0.0;
    int width  = 0;
    int height = 0;
};

struct DynamicObstacle {
    std::string id;
    Footprint footprint;
    std::vector<Waypoint> waypoints;
};

static std::vector<DynamicObstacle> loadDynamicObstacles(const std::string& yaml_path)
{
    std::vector<DynamicObstacle> obstacles;

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

// 3D occupancy grid: occ[t][y][x], with 1-based x,y indices
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
          numofmoves_(0),
          caught_(false),
          pathcost_(0),
          finished_(false)
    {
        // Get map file + dyno yaml from params
        nh_.param<std::string>("map_file", map_file_path_, std::string(""));
        nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, std::string(""));

        //Publisher for robot pose and target 
        robot_pub_ = nh_.advertise<geometry_msgs::Pose2D>("robot_pose", 1);

        target_pub_ = nh_.advertise<geometry_msgs::Pose2D>("target_pose", 1);

        if (!loadProblemFromFile()) {
            ROS_FATAL("PlannerNode: failed to load problem from file.");
            finished_ = true;
            return;
        }

        // allocate action_ptr once
        action_ptr_ = new int[2];

        // open output trajectory file (same as runtest.cpp)
        std::string outputDir = OUTPUT_DIR;
        std::string outputFilePath = outputDir + "/robot_trajectory.txt";
        // output_file_.open(outputFilePath);
        // if (!output_file_.is_open()) {
        //     ROS_ERROR_STREAM("Failed to open output file: " << outputFilePath);
        //     finished_ = true;
        //     return;
        // }

        // write initial state
        // output_file_ << curr_time_ << "," << robotposeX_ << "," << robotposeY_ << std::endl;

        ROS_INFO("PlannerNode initialized from file. Ready to run control loop.");
    }

    ~PlannerNode() {
        if (target_traj_) delete[] target_traj_;
        if (map_) delete[] map_;
        if (action_ptr_) delete[] action_ptr_;
        // if (output_file_.is_open()) output_file_.close();
    }

    bool isFinished() const { return finished_; }

    double computeRobotHeading(int oldX, int oldY, int newX, int newY)
    {
        int dx = newX - oldX;
        int dy = newY - oldY;
        return std::atan2(dy, dx);   // radians
    }


    void spinOnce()
    {
        if (finished_) return;

        // === This is one iteration of the original while(true) body ===

        auto start = std::chrono::high_resolution_clock::now();

        int targetposeX = target_traj_[curr_time_];
        int targetposeY = target_traj_[curr_time_ + target_steps_];

        planner(map_,
                occ3D_,
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

        // --- same checks as runtest.cpp ---
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
            ROS_INFO("Reached end of target_steps_");
            printResult();
            return;
        }

        curr_time_ = curr_time_ + movetime;
        numofmoves_ = numofmoves_ + 1;
        pathcost_ = pathcost_ + movetime * map_[(robotposeY_-1)*x_size_ + robotposeX_-1];

        int prevRobotX = robotposeX_;
        int prevRobotY = robotposeY_;
        robotposeX_ = newrobotposeX;
        robotposeY_ = newrobotposeY;

        // Publish robot pose
        geometry_msgs::Pose2D robot_msg;
        robot_msg.x = newrobotposeX;
        robot_msg.y = newrobotposeY;
        robot_msg.theta = computeRobotHeading(prevRobotX, prevRobotY, newrobotposeX, newrobotposeY);
        robot_pub_.publish(robot_msg);

        // Publish target pose
        geometry_msgs::Pose2D target_msg;
        target_msg.x = targetposeX;
        target_msg.y = targetposeY;

        // If you know next target waypoint:
        if (curr_time_ + 1 < target_steps_) {
            int nextX = target_traj_[curr_time_ + 1];
            int nextY = target_traj_[curr_time_ + 1 + target_steps_];
            target_msg.theta = std::atan2(nextY - targetposeY, nextX - targetposeX);
        } else {
            target_msg.theta = 0.0;
        }

        target_pub_.publish(target_msg);

        // --- Publish dynamic obstacle poses at current curr_time_ ---
        if (!obstacles_.empty() &&
            !obstacle_trajs_.empty() &&
            curr_time_ >= 0 &&
            curr_time_ < target_steps_)
        {
            for (size_t i = 0; i < obstacles_.size(); ++i) {
                const auto& ob   = obstacles_[i];
                const auto& traj = obstacle_trajs_[i];

                if ((size_t)curr_time_ >= traj.size()) continue;
                const auto& cells = traj[curr_time_];
                if (cells.empty()) continue;

                int ox = cells[0].first;
                int oy = cells[0].second;

                geometry_msgs::Pose2D msg;
                msg.x = ox;
                msg.y = oy;

                // simple heading estimate based on next time step
                if ((size_t)(curr_time_ + 1) < traj.size() &&
                    !traj[curr_time_ + 1].empty())
                {
                    int nx = traj[curr_time_ + 1][0].first;
                    int ny = traj[curr_time_ + 1][0].second;
                    msg.theta = std::atan2(ny - oy, nx - ox);
                } else {
                    msg.theta = 0.0;
                }

                auto it = obstacle_pub_map_.find(ob.id);
                if (it != obstacle_pub_map_.end()) {
                    it->second.publish(msg);
                }
            }
        }

        // log to file
        // output_file_ << curr_time_ << "," << robotposeX_ << "," << robotposeY_ << std::endl;

        // check if target is caught
        float thresh = 0.5f;
        targetposeX = target_traj_[curr_time_];
        targetposeY = target_traj_[curr_time_ + target_steps_];
        if (std::abs(robotposeX_ - targetposeX) <= thresh &&
            std::abs(robotposeY_ - targetposeY) <= thresh)
        {
            caught_ = true;
            finished_ = true;
            printResult();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher  robot_pub_;
    ros::Publisher  target_pub_;
    std::map<std::string, ros::Publisher> obstacle_pub_map_;

    // dynamic obstacles
    std::vector<DynamicObstacle> obstacles_;
    std::vector<ObstacleTraj>    obstacle_trajs_;

    // problem data
    int* map_;
    int* target_traj_;
    std::vector<std::vector<std::vector<bool>>> occ3D_;

    int x_size_, y_size_;
    int collision_thresh_;
    int robotposeX_, robotposeY_;
    int curr_time_;
    int target_steps_;
    int* action_ptr_;

    int numofmoves_;
    bool caught_;
    int pathcost_;
    bool finished_;

    std::string map_file_path_;
    std::string dyno_yaml_path_;

    // std::ofstream output_file_;

    bool loadProblemFromFile()
    {
        std::string default_map =
            ros::package::getPath("pursuit_planner") + "/maps/map4.txt";
        std::string default_yaml =
            ros::package::getPath("pursuit_planner") + "/config/dyno_map4.yaml";

            // ---- Load parameters from the private namespace "~" ----
        nh_.param<std::string>("map_file",  map_file_path_,  default_map);
        nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, default_yaml);

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

        // read trajectory
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
        target_traj_ = new int[2 * target_steps_];
        for (size_t i = 0; i < target_steps_; ++i) {
            target_traj_[i]             = traj[i][0];
            target_traj_[i + target_steps_] = traj[i][1];
        }
        ROS_INFO_STREAM("target_steps: " << target_steps_);

        // load dynamic obstacles + build occ3D
        auto obstacles = loadDynamicObstacles(dyno_yaml_path_);
        ROS_INFO_STREAM("Loaded " << obstacles.size() << " dynamic obstacles from " << dyno_yaml_path_);
        obstacles_ = obstacles;
        occ3D_ = buildDynamicOccupancyGrid(obstacles_, x_size_, y_size_, target_steps_);

        // simulate and store obstacle trajectories
        obstacle_trajs_.clear();
        for (const auto& ob : obstacles_) {
            obstacle_trajs_.push_back(simulateObstacle(ob, target_steps_));
        }
        obstacle_pub_map_.clear();
        //Create publishers for dynamic obstacles
        for (const auto& ob : obstacles_) {
            std::string topic = "/dynamic_obstacles/" + ob.id + "/pose";
            obstacle_pub_map_[ob.id] =
                nh_.advertise<geometry_msgs::Pose2D>(topic, 1);
            ROS_INFO_STREAM("Advertising dynamic obstacle topic: " << topic);
        }

        // read map (M section just finished)
        map_ = new int[x_size_ * y_size_];
        for (size_t i = 0; i < x_size_; i++) {
            std::getline(myfile, line);
            std::stringstream ss(line);
            for (size_t j = 0; j < y_size_; j++) {
                double value;
                ss >> value;
                map_[j * x_size_ + i] = (int)value;
                if (j != y_size_ - 1) ss.ignore();
            }
        }

        myfile.close();
        curr_time_ = 0;
        return true;
    }

    void printResult()
    {
        ROS_INFO_STREAM("\nRESULT");
        ROS_INFO_STREAM("target caught = " << (caught_ ? "true" : "false"));
        ROS_INFO_STREAM("time taken (s) = " << curr_time_);
        ROS_INFO_STREAM("moves made = " << numofmoves_);
        ROS_INFO_STREAM("path cost = " << pathcost_);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pursuit_planner_runtest_node");
    ros::NodeHandle nh("~");

    PlannerNode node(nh);

    ros::Rate rate(100.0);  // match the original "1 Hz-ish" style
    while (ros::ok() && !node.isFinished()) {
        ros::spinOnce();
        node.spinOnce();
        rate.sleep();
    }

    return 0;
}
