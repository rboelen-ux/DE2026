# A Cheat Sheet with most Docker Commands

https://docs.docker.com/get-started/docker_cheatsheet.pdf

# Dockerize the prediction-ui component - create a Docker image, create a container, run it and check its logs

sudo docker build -t indikakumara/prediction-ui:0.0.1 .

sudo docker run -p  5000:5000 -d --name=prediction-ui indikakumara/prediction-ui:0.0.1

sudo docker start prediction-ui

sudo docker logs prediction-ui

# Log in to a container

sudo docker exec -it prediction-ui /bin/bash

# Stop and remove all Docker containers

sudo docker stop $(sudo docker ps -a -q)

sudo docker rm $(sudo docker ps -a -q)

sudo docker ps -as


# Remove all Docker networks and volumes

sudo docker network prune

sudo docker volume prune

sudo docker volume rm $(sudo docker volume ls -qf dangling=true)

# Remove all images

sudo docker rmi -f $(sudo docker images -aq)

# Delete everything

sudo docker system prune -a --volumes
