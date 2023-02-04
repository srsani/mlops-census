branch_name=$1

echo $branch_name

echo build and push Docker image to ECR
sh docker_ecr.sh latest

echo Installing compose-cli
curl -L https://raw.githubusercontent.com/docker/compose-cli/main/scripts/install/install_linux.sh | sh

# docker context create ecs ecs-context --from-env
# docker --context ecs-context compose up
