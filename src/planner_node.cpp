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

// ============================================================================
// PlannerNode
// ============================================================================

// class PlannerNode {
// public:
//     PlannerNode(ros::NodeHandle& nh)
//         : nh_(nh),
//           got_robot_pose_(false),
//           got_target_traj_(false),
//           got_target_pose_(false),
//           occ3D_ready_(false),
//           curr_time_(0),
//           target_steps_(0),
//           caught_(false),
//           x_size_from_file_(0),
//           y_size_from_file_(0),
//           problem_loaded_(false)
//     {
//         // Parameters
//         std::string default_yaml =
//             ros::package::getPath("pursuit_planner") + "/config/dyno_map3.yaml";
//         nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, default_yaml);

//         std::string default_map =
//             ros::package::getPath("pursuit_planner") + "/maps/map3.txt";
//         nh_.param<std::string>("map_file", map_file_path_, default_map);

//         nh_.param<double>("catch_threshold", catch_thresh_, 0.5);

//         // Load problem (map + robot + trajectory) once from file
//         if (!loadProblemFromFile(map_file_path_)) {
//             ROS_ERROR("Failed to load problem from map_file; planner_node will wait for topics.");
//         }

//         // Subscriptions (no map topic needed now)
//         robot_sub_       = nh_.subscribe("robot_pose",  1, &PlannerNode::robotCallback, this);
//         target_traj_sub_ = nh_.subscribe("target_traj", 1, &PlannerNode::targetTrajCallback, this);
//         target_pose_sub_ = nh_.subscribe("target_pose", 1, &PlannerNode::targetPoseCallback, this);

//         // Publishers
//         cmd_pub_    = nh_.advertise<geometry_msgs::Pose2D>("next_waypoint", 1);
//         status_pub_ = nh_.advertise<std_msgs::String>("planner_status", 1);
//         caught_pub_ = nh_.advertise<std_msgs::Bool>("target_caught", 1);
//         ROS_INFO_STREAM("PlannerNode initialized.");
//     }

//     void spinOnce()
//     {
//         if (!(problem_loaded_ && got_robot_pose_ && got_target_traj_)) {
//             ROS_WARN_THROTTLE(1.0,
//                 "Waiting for all inputs: problem_loaded=%d, got_robot_pose=%d, got_target_traj=%d",
//                 problem_loaded_, got_robot_pose_, got_target_traj_);
//             return;  // wait until everything is ready
//         }

//         const int x_size = x_size_from_file_;
//         const int y_size = y_size_from_file_;

//         if (x_size <= 0 || y_size <= 0) {
//             ROS_WARN_THROTTLE(1.0, "Map size is zero; skipping planning step.");
//             return;
//         }

//         int* map = map_vec_.data();

//         // For ASCII maps (0 free, 1 obstacle) we use threshold 1
//         int collision_thresh = 1;
//         ROS_INFO_STREAM("Planning step at time " << curr_time_
//                         << " with map size (" << x_size << "," << y_size << ")");   
//         // --------------------------------------------------------------------
//         // Ensure dynamic obstacles & occ3D are ready once we know target_steps_
//         // --------------------------------------------------------------------
//         if (!occ3D_ready_) {
//             if (target_steps_ <= 0) {
//                 ROS_WARN_THROTTLE(1.0,
//                     "Target trajectory has zero length; cannot build occ3D yet.");
//                 return;
//             }

//             try {
//                 obstacles_ = loadDynamicObstacles(dyno_yaml_path_);
//                 ROS_INFO_STREAM("Loaded " << obstacles_.size()
//                                 << " dynamic obstacles from " << dyno_yaml_path_);

//                 // Precompute trajectories for publishing
//                 obstacle_trajs_.clear();
//                 for (const auto& ob : obstacles_) {
//                     obstacle_trajs_.push_back(simulateObstacle(ob, target_steps_));
//                 }

//                 // Build 3D occupancy grid for planner
//                 occ3D_ = buildDynamicOccupancyGrid(obstacles_, x_size, y_size, target_steps_);

//                 // Create publishers for each obstacle
//                 obstacle_pubs_.clear();
//                 for (const auto& ob : obstacles_) {
//                     const std::string topic =
//                         "/dynamic_obstacles/" + ob.id + "/pose";
//                     obstacle_pubs_[ob.id] =
//                         nh_.advertise<geometry_msgs::Pose2D>(topic, 1);
//                     ROS_INFO_STREAM("Advertising dynamic obstacle topic: " << topic);
//                 }

//                 occ3D_ready_ = true;
//             }
//             catch (const std::exception& e) {
//                 ROS_ERROR_STREAM("Failed to load dynamic obstacles from "
//                                  << dyno_yaml_path_ << ": " << e.what());
//                 // Fallback: empty occ3D (no dynamic obstacles)
//                 occ3D_.assign(
//                     target_steps_ + 1,
//                     std::vector<std::vector<bool>>(
//                         y_size + 1, std::vector<bool>(x_size + 1, false)));
//                 occ3D_ready_ = true;
//             }
//             ROS_INFO_STREAM("Dynamic occupancy grid (occ3D) ready.");
//         }

//         if (!occ3D_ready_) {
//             return;
//         }

//         // --------------------------------------------------------------------
//         // Get robot & target poses in grid coordinates (assume already in cells)
//         // --------------------------------------------------------------------
//         int robotposeX = static_cast<int>(std::round(robot_pose_.x));
//         int robotposeY = static_cast<int>(std::round(robot_pose_.y));

//         if (robotposeX < 1 || robotposeX > x_size ||
//             robotposeY < 1 || robotposeY > y_size)
//         {
//             ROS_WARN_THROTTLE(1.0,
//                 "Robot pose (%d,%d) out of map bounds; skipping.",
//                 robotposeX, robotposeY);
//             return;
//         }

//         // Target pose from trajectory at curr_time
//         if (curr_time_ < 0 || curr_time_ >= target_steps_) {
//             ROS_WARN_THROTTLE(1.0,
//                 "curr_time=%d out of target trajectory range [0,%d).",
//                 curr_time_, target_steps_);
//             return;
//         }
//         int targetposeX = target_traj_buf_[curr_time_];
//         int targetposeY = target_traj_buf_[curr_time_ + target_steps_];

//         // --------------------------------------------------------------------
//         // Call planner (same signature as in planner.cpp)
//         // --------------------------------------------------------------------
//         int action[2] = { robotposeX, robotposeY };
//         // ROS_INFO_STREAM("Calling planner for time step " << curr_time_
//         //                 << " with robot at (" << robotposeX << "," << robotposeY
//         //                 << ") and target at (" << targetposeX << "," << targetposeY << ")");
//         planner(map,
//                 occ3D_,
//                 collision_thresh,
//                 x_size,
//                 y_size,
//                 robotposeX,
//                 robotposeY,
//                 target_steps_,
//                 target_traj_buf_.data(),
//                 targetposeX,
//                 targetposeY,
//                 curr_time_,
//                 action);

//         int newrobotposeX = action[0];
//         int newrobotposeY = action[1];
//         ROS_INFO_STREAM("Planner selected next waypoint: ("
//                         << newrobotposeX << "," << newrobotposeY << ")");
//         // --------------------------------------------------------------------
//         // Basic sanity checks (matching runtest.cpp behavior)
//         // --------------------------------------------------------------------
//         if (newrobotposeX < 1 || newrobotposeX > x_size ||
//             newrobotposeY < 1 || newrobotposeY > y_size)
//         {
//             ROS_ERROR("Planner commanded out-of-map position (%d,%d).",
//                       newrobotposeX, newrobotposeY);
//             return;
//         }

//         int map_idx = (newrobotposeY - 1) * x_size + (newrobotposeX - 1);
//         if (map[map_idx] >= collision_thresh) {
//             ROS_ERROR("Planner action leads to collision at (%d,%d).",
//                       newrobotposeX, newrobotposeY);
//             return;
//         }

//         if (std::abs(robotposeX - newrobotposeX) > 1 ||
//             std::abs(robotposeY - newrobotposeY) > 1)
//         {
//             ROS_WARN("Planner commanded non-8-connected move: (%d,%d)->(%d,%d).",
//                      robotposeX, robotposeY, newrobotposeX, newrobotposeY);
//         }

//         // --------------------------------------------------------------------
//         // "Caught" condition
//         // --------------------------------------------------------------------
//         int tgtX_now = target_traj_buf_[curr_time_];
//         int tgtY_now = target_traj_buf_[curr_time_ + target_steps_];

//         if (std::abs(newrobotposeX - tgtX_now) <= catch_thresh_ &&
//             std::abs(newrobotposeY - tgtY_now) <= catch_thresh_)
//         {
//             caught_ = true;
//         }

//         // --------------------------------------------------------------------
//         // Publish robot command
//         // --------------------------------------------------------------------
//         geometry_msgs::Pose2D cmd;
//         cmd.x     = newrobotposeX;
//         cmd.y     = newrobotposeY;
//         cmd.theta = 0.0;
//         cmd_pub_.publish(cmd);

//         // --------------------------------------------------------------------
//         // Publish planner status
//         // --------------------------------------------------------------------
//         std_msgs::String status_msg;
//         std::stringstream ss;
//         ss << "t=" << curr_time_
//            << " robot=(" << robotposeX << "," << robotposeY << ")"
//            << " target=(" << tgtX_now << "," << tgtY_now << ")"
//            << " cmd=(" << newrobotposeX << "," << newrobotposeY << ")"
//            << " caught=" << (caught_ ? "true" : "false");
//         status_msg.data = ss.str();
//         status_pub_.publish(status_msg);

//         // --------------------------------------------------------------------
//         // Publish target_caught boolean
//         // --------------------------------------------------------------------
//         std_msgs::Bool caught_msg;
//         caught_msg.data = caught_;
//         caught_pub_.publish(caught_msg);

//         // --------------------------------------------------------------------
//         // Publish dynamic obstacle poses for simulation
//         // --------------------------------------------------------------------
//         if (!obstacles_.empty() &&
//             !obstacle_trajs_.empty() &&
//             curr_time_ >= 0 &&
//             curr_time_ < target_steps_)
//         {
//             for (size_t i = 0; i < obstacles_.size(); ++i) {
//                 const auto& ob   = obstacles_[i];
//                 const auto& traj = obstacle_trajs_[i];

//                 if ((int)traj.size() <= curr_time_) continue;
//                 const auto& cells = traj[curr_time_];
//                 if (cells.empty()) continue;

//                 geometry_msgs::Pose2D p;
//                 p.x     = cells[0].first;
//                 p.y     = cells[0].second;
//                 p.theta = 0.0;

//                 auto it = obstacle_pubs_.find(ob.id);
//                 if (it != obstacle_pubs_.end()) {
//                     it->second.publish(p);
//                 }
//             }
//         }

//         if (curr_time_ + 1 < target_steps_) {
//             ++curr_time_;
//         }
//     }

// private:
//     ros::NodeHandle nh_;

//     // Subscribers
//     ros::Subscriber robot_sub_;
//     ros::Subscriber target_traj_sub_;
//     ros::Subscriber target_pose_sub_;

//     // Publishers
//     ros::Publisher cmd_pub_;
//     ros::Publisher status_pub_;
//     ros::Publisher caught_pub_;
//     std::map<std::string, ros::Publisher> obstacle_pubs_;

//     // Cached messages
//     geometry_msgs::Pose2D   robot_pose_;
//     geometry_msgs::Pose2D   target_pose_;
//     std_msgs::Int32MultiArray target_traj_;

//     // Flags
//     bool got_robot_pose_;
//     bool got_target_traj_;
//     bool got_target_pose_;
//     bool occ3D_ready_;

//     // Time index
//     int curr_time_;
//     int target_steps_;
//     bool caught_;

//     // Parameters
//     std::string dyno_yaml_path_;
//     std::string map_file_path_;
//     double catch_thresh_;

//     // Map from ASCII file
//     std::vector<int> map_vec_;
//     int x_size_from_file_;
//     int y_size_from_file_;
//     bool problem_loaded_;

//     // Dynamic obstacles
//     std::vector<DynamicObstacle> obstacles_;
//     std::vector<ObstacleTraj> obstacle_trajs_;
//     std::vector<std::vector<std::vector<bool>>> occ3D_;

//     std::vector<int> target_traj_buf_;
//     // ------------------------------------------------------------------------
//     // Helpers
//     // ------------------------------------------------------------------------

//     bool loadProblemFromFile(const std::string& path)
//     {
//         std::ifstream myfile(path);
//         if (!myfile.is_open()) {
//             ROS_ERROR_STREAM("Failed to open problem/map file: " << path);
//             problem_loaded_ = false;
//             return false;
//         }

//         char letter;
//         std::string line;
//         int x_size, y_size;

//         // --- N x_size x y_size ---
//         myfile >> letter;
//         if (letter != 'N') {
//             ROS_ERROR("Error parsing map file: expected 'N' at start");
//             problem_loaded_ = false;
//             return false;
//         }

//         myfile >> x_size >> letter >> y_size;
//         ROS_INFO_STREAM("Map size from file: " << x_size << "x" << y_size);

//         // --- C collision_thresh ---
//         int collision_thresh_;
//         myfile >> letter;
//         if (letter != 'C') {
//             ROS_ERROR("Error parsing map file: expected 'C' (collision threshold)");
//             problem_loaded_ = false;
//             return false;
//         }
//         myfile >> collision_thresh_;
//         ROS_INFO_STREAM("Collision threshold from file: " << collision_thresh_);

//         // --- R robotposeX x robotposeY ---
//         int robotposeX, robotposeY;
//         myfile >> letter;
//         if (letter != 'R') {
//             ROS_ERROR("Error parsing map file: expected 'R' (robot pose)");
//             problem_loaded_ = false;
//             return false;
//         }
//         myfile >> robotposeX >> letter >> robotposeY;
//         ROS_INFO_STREAM("Initial robot pose from file: "
//                         << robotposeX << "x" << robotposeY);

//         // Consume end of line before searching for 'T'
//         std::getline(myfile, line);

//         // --- seek 'T' (trajectory section) ---
//         do {
//             if (!std::getline(myfile, line)) {
//                 ROS_ERROR("Error parsing map file: could not find 'T' line");
//                 problem_loaded_ = false;
//                 return false;
//             }
//         } while (line != "T");

//         // --- read trajectory lines until 'M' ---
//         std::vector<std::vector<int>> traj;
//         while (std::getline(myfile, line) && line != "M") {
//             if (line.empty()) continue;
//             std::stringstream ss(line);
//             int num1, num2;
//             char delim;
//             ss >> num1 >> delim >> num2;  // e.g. "10x20"
//             traj.push_back({num1, num2});
//         }

//         target_steps_ = static_cast<int>(traj.size());
//         if (target_steps_ <= 0) {
//             ROS_ERROR("No target trajectory points found in map file.");
//             problem_loaded_ = false;
//             return false;
//         }

//         target_traj_buf_.assign(2 * target_steps_, 0);
//         for (int i = 0; i < target_steps_; ++i) {
//             target_traj_buf_[i]               = traj[i][0]; // x(t)
//             target_traj_buf_[i + target_steps_] = traj[i][1]; // y(t)
//         }
//         ROS_INFO_STREAM("Loaded target trajectory with "
//                         << target_steps_ << " steps from file.");

//         // --- read map grid values ---
//         map_vec_.assign(x_size * y_size, 0);
//         for (int i = 0; i < x_size; ++i) {
//             if (!std::getline(myfile, line)) {
//                 ROS_ERROR("Unexpected end of file while reading map grid.");
//                 problem_loaded_ = false;
//                 return false;
//             }
//             if (line.empty()) { 
//                 // handle possible blank lines
//                 --i;
//                 continue;
//             }

//             std::stringstream ss(line);
//             for (int j = 0; j < y_size; ++j) {
//                 double value;
//                 char delim;
//                 ss >> value;
//                 if (j != y_size - 1) {
//                     ss >> delim;  // skip separator if present
//                 }
//                 map_vec_[j * x_size + i] = static_cast<int>(value);
//             }
//         }

//         myfile.close();

//         // Store sizes
//         x_size_from_file_ = x_size;
//         y_size_from_file_ = y_size;

//         // Initialize internal state flags
//         robot_pose_.x = robotposeX;
//         robot_pose_.y = robotposeY;
//         robot_pose_.theta = 0.0;
//         got_robot_pose_  = true;

//         got_target_traj_ = true;
//         curr_time_       = 0;
//         caught_          = false;
//         occ3D_ready_     = false;

//         problem_loaded_ = true;

//         ROS_INFO("Successfully loaded problem (map, robot pose, trajectory) from file.");
//         return true;
//     }


//     // ------------------------------------------------------------------------
//     // Callbacks
//     // ------------------------------------------------------------------------
//     void robotCallback(const geometry_msgs::Pose2D::ConstPtr& msg)
//     {
//         robot_pose_ = *msg;
//         got_robot_pose_ = true;
//     }

//     void targetTrajCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
//     {
//         target_traj_      = *msg;
//         target_traj_buf_  = target_traj_.data;
//         target_steps_     = target_traj_buf_.size() / 2;
//         occ3D_ready_      = false;
//         got_target_traj_  = true;
//         curr_time_        = 0;
//         caught_           = false;
//     }

//     void targetPoseCallback(const geometry_msgs::Pose2D::ConstPtr& msg)
//     {
//         target_pose_ = *msg;
//         got_target_pose_ = true;
//     }
// };

// // ============================================================================

// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "pursuit_planner_node");
//     ros::NodeHandle nh("~");

//     PlannerNode node(nh);

//     ros::Rate rate(1.0); 
//     while (ros::ok()) {
//         ros::spinOnce();
//         node.spinOnce();
//         rate.sleep();
//     }

//     return 0;
// }

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
        // Get map file + dyno yaml from params (instead of argv)
        nh_.param<std::string>("map_file", map_file_path_, std::string(""));
        nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, std::string(""));

        // Optional: publisher for robot pose (so a sim can listen)
        cmd_pub_ = nh_.advertise<geometry_msgs::Pose2D>("planner_cmd", 1);

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

        robotposeX_ = newrobotposeX;
        robotposeY_ = newrobotposeY;

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

        // Optional: publish current robot pose to simulation
        geometry_msgs::Pose2D cmd;
        cmd.x = robotposeX_;
        cmd.y = robotposeY_;
        cmd.theta = 0.0;
        cmd_pub_.publish(cmd);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher  cmd_pub_;

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
        std::string mapFilePath =
            ros::package::getPath("pursuit_planner") + "/maps/map3.txt";
        nh_.param<std::string>("map_file", map_file_path_, mapFilePath);

        // std::string mapDirPath = MAPS_DIR;
        // std::string mapFilePath = mapDirPath + "/" + map_file_;
        ROS_INFO_STREAM("Reading problem definition from: " << mapFilePath);

        std::ifstream myfile(mapFilePath);
        if (!myfile.is_open()) {
            ROS_ERROR_STREAM("Failed to open the file: " << mapFilePath);
            return false;
        }        

        std::string default_yaml =
            ros::package::getPath("pursuit_planner") + "/config/dyno_map3.yaml";
        nh_.param<std::string>("dyno_yaml", dyno_yaml_path_, default_yaml);

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

        occ3D_ = buildDynamicOccupancyGrid(obstacles, x_size_, y_size_, target_steps_);

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
