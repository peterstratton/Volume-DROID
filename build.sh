#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t ros_image:latest "
# DOCKER_OPTIONS+="-f $SCRIPT_DIR/container.Dockerfile "
DOCKER_OPTIONS+="--build-arg USER_ID=$(whoami)"

DOCKER_CMD="docker build $DOCKER_OPTIONS - < $SCRIPT_DIR/container.Dockerfile"
echo $DOCKER_CMD
exec $DOCKER_CMD