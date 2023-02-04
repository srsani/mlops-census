branch_name=$1

echo $branch_name
pip3 install -r requirements.txt
sh docker_ecr.sh latest

pwd

ls -al 

# docker context create ecs srs-fastapi
# docker context use srs-fastapi
# docker compose up