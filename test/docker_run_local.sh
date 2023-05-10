#!/bin/bash

export JOB_NAME="ml-lbp"
export IMAGE="ml/lbp"
export TAG="latest"
export PYTHON_ENV="development"
export API_PORT=8080
export WORKERS=2
export TIMEOUT=300
export LOG_FOLDER=/var/log/ml-lbp

echo ${IMAGE}:${TAG}

#Create log folder if not exists

if [ ! -d ${LOG_FOLDER} ]; then
     mkdir ${LOG_FOLDER}
fi

# Add your authentication command for the docker image registry here

# force pull and update the image, use this in remote host only
# docker pull ${IMAGE}:${TAG}

# stop running container with same job name, if any
if [ "$(docker ps -a | grep $JOB_NAME)" ]; then
  docker stop ${JOB_NAME} && docker rm ${JOB_NAME}
fi

# start docker container WITHOUT gpu and log volume
docker run -d \
  --rm \
  -p ${API_PORT}:80 \
  -e "WORKERS=${WORKERS}" \
  -e "TIMEOUT=${TIMEOUT}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG}