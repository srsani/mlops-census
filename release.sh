branch_name=$1

echo $branch_name

echo build and push Docker image to ECR
sh docker_ecr.sh latest

curl -L https://raw.githubusercontent.com/docker/compose-cli/main/scripts/install/install_linux.sh | sh

sudo docker context create ecs srs-fastapi
docker context use srs-fastapi
docker compose up