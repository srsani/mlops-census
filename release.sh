branch_name=$1

echo $branch_name
pip3 install -r requirements.txt
sh docker_ecr.sh latest

# docker context create ecs fastapi
# docker context use fastapi
# docker compose up