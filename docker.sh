#!/bin/bash

# Helper script to work with the ROS Docker container

case "$1" in
  build)
    echo "🔨 Building Docker image..."
    docker-compose build
    ;;
    
  shell)
    echo "🐚 Starting interactive shell in container..."
    docker-compose run --rm pursuit_planner bash
    ;;
    
  compile)
    echo "⚙️  Compiling pursuit_planner..."
    docker-compose run --rm pursuit_planner bash -c "
      source /opt/ros/noetic/setup.bash && 
      cd /catkin_ws && 
      catkin_make
    "
    ;;
    
  run)
    echo "🚀 Running pursuit_planner node..."
    if [ -z "$2" ]; then
      docker-compose run --rm pursuit_planner bash -c "
        source /opt/ros/noetic/setup.bash && 
        source /catkin_ws/devel/setup.bash && 
        roslaunch pursuit_planner pursuit_planner.launch
      "
    else
      docker-compose run --rm pursuit_planner bash -c "
        source /opt/ros/noetic/setup.bash && 
        source /catkin_ws/devel/setup.bash && 
        roslaunch pursuit_planner pursuit_planner.launch $2
      "
    fi
    ;;
    
  clean)
    echo "🧹 Cleaning build artifacts..."
    docker-compose run --rm pursuit_planner bash -c "
      rm -rf /catkin_ws/build/* /catkin_ws/devel/*
    "
    ;;
    
  stop)
    echo "⏹️  Stopping all containers..."
    docker-compose down
    ;;
    
  *)
    echo "Usage: ./docker.sh {build|shell|compile|run|clean|stop}"
    echo ""
    echo "Commands:"
    echo "  build    - Build the Docker image"
    echo "  shell    - Start an interactive bash shell in the container"
    echo "  compile  - Compile the pursuit_planner package"
    echo "  run      - Run the pursuit_planner launch file"
    echo "  clean    - Remove build artifacts"
    echo "  stop     - Stop and remove containers"
    echo ""
    echo "Examples:"
    echo "  ./docker.sh build"
    echo "  ./docker.sh shell"
    echo "  ./docker.sh compile"
    echo "  ./docker.sh run"
    echo "  ./docker.sh run 'map_file:=maps/map4.txt dyno_yaml:=config/dyno_map4.yaml'"
    exit 1
    ;;
esac
