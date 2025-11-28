#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose2D.h>
#include <std_msgs/Int32MultiArray.h>

#include "pursuit_planner/planner.h"

class PlannerNode {
public:
    PlannerNode(ros::NodeHandle& nh)
        : nh_(nh),
          got_map_(false),
          got_robot_pose_(false),
          got_target_traj_(false),
          got_target_pose_(false),
          curr_time_(0)
    {
        map_sub_         = nh_.subscribe("map",         1, &PlannerNode::mapCallback, this);
        robot_sub_       = nh_.subscribe("robot_pose",  1, &PlannerNode::robotCallback, this);
        target_traj_sub_ = nh_.subscribe("target_traj", 1, &PlannerNode::targetTrajCallback, this);
        target_pose_sub_ = nh_.subscribe("target_pose", 1, &PlannerNode::targetPoseCallback, this);

        cmd_pub_ = nh_.advertise<geometry_msgs::Pose2D>("next_waypoint", 1);
    }

    void spinOnce()
    {
        if (!(got_map_ && got_robot_pose_ && got_target_traj_ && got_target_pose_)) {
            return; // wait until everything is ready
        }

        // 1) Convert OccupancyGrid to int* map
        int x_size = map_msg_.info.width;
        int y_size = map_msg_.info.height;

        static std::vector<int> map_vec;
        map_vec.resize(x_size * y_size);

        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                int idx = y * x_size + x;
                int val = map_msg_.data[idx];
                if (val < 0) {
                    map_vec[idx] = 100;  // unknown as obstacle-ish
                } else if (val > 0) {
                    map_vec[idx] = 100;  // occupied
                } else {
                    map_vec[idx] = 1;    // free cell cost
                }
            }
        }

        int* map = map_vec.data();
        int collision_thresh = 100;  // anything >= is obstacle

        // 2) Robot and target pose in grid cell coordinates
        int robotposeX = static_cast<int>(robot_pose_.x);
        int robotposeY = static_cast<int>(robot_pose_.y);

        int targetposeX = static_cast<int>(target_pose_.x);
        int targetposeY = static_cast<int>(target_pose_.y);

        // 3) Target trajectory: Int32MultiArray with [x(0..T-1), y(0..T-1)]
        int target_steps = target_traj_.data.size() / 2;
        static std::vector<int> target_traj_buf;
        target_traj_buf = target_traj_.data; // contiguous

        int action[2] = {robotposeX, robotposeY};

        // 4) Call your planner()
        planner(map,
                collision_thresh,
                x_size,
                y_size,
                robotposeX,
                robotposeY,
                target_steps,
                target_traj_buf.data(),
                targetposeX,
                targetposeY,
                curr_time_,
                action);

        // 5) Publish the result as next waypoint in grid
        geometry_msgs::Pose2D cmd;
        cmd.x = action[0];
        cmd.y = action[1];
        cmd.theta = 0.0;

        cmd_pub_.publish(cmd);

        curr_time_++;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_, robot_sub_, target_traj_sub_, target_pose_sub_;
    ros::Publisher cmd_pub_;

    nav_msgs::OccupancyGrid map_msg_;
    geometry_msgs::Pose2D robot_pose_, target_pose_;
    std_msgs::Int32MultiArray target_traj_;

    bool got_map_, got_robot_pose_, got_target_traj_, got_target_pose_;
    int curr_time_;

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        map_msg_ = *msg;
        got_map_ = true;
    }

    void robotCallback(const geometry_msgs::Pose2D::ConstPtr& msg) {
        robot_pose_ = *msg;
        got_robot_pose_ = true;
    }

    void targetTrajCallback(const std_msgs::Int32MultiArray::ConstPtr& msg) {
        target_traj_ = *msg;
        got_target_traj_ = true;
    }

    void targetPoseCallback(const geometry_msgs::Pose2D::ConstPtr& msg) {
        target_pose_ = *msg;
        got_target_pose_ = true;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pursuit_planner_node");
    ros::NodeHandle nh;

    PlannerNode node(nh);

    ros::Rate rate(10.0); // 10 Hz
    while (ros::ok()) {
        ros::spinOnce();
        node.spinOnce();
        rate.sleep();
    }

    return 0;
}
