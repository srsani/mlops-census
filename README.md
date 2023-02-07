# MlOps FastAPI

This repo contains the code for training a Random Forest model on the `census.csv` data set using `sci-kit-learn` and `hydra-core`. The data and artefacts for this project are stored in `S3` by using `DVC`.

The model is then served using FastAPI and AWS `ECR`, and `ECS`.

Some high-level `pytest` are also implemented to test the API performance.

Furthermore, I used Docker compose to deploy the `ECS` infrastructure to serve the docker image that was built and pushed to `ECR`.

Lastly, I have implemented a CI/CD pipeline using `GitHub Actions`.


This repo contain the code for train a RF
## Environment Set up

All the code have tested using `python3.8` and `requirements.txt`

## Local dev/test:

### Model training:

Run the training pipeline:

- `cd  src`
- `python main_train.py`
### pytests

`python -m pytest src/tests -rP -s -v`
### API
Start the FastAPI server with :
-  `uvicorn src.main_api:app --reload`

API doc is under `http://127.0.0.1:8000/docs`

## ECR

Run the following command to push the latest docker image to ECR

- sh docker_ecr.sh latest

## ECS

```bash
docker context create ecs srs-fastapi
docker context use srs-fastapi
docker compose up
```

You should see the following image, and it takes about 5 min for the API to be acceptable via the load balancer's DNS address.

![Deploy the end point](/screenshots/1.png)

- Clean and delete all the resources
    - `docker compose down`

The following image depicts the deletion process.

![Deploy the end point](/screenshots/2.png)

## CI/CD

On push to the main branch of this repo, GitHub Action will do the followings:

1. Run Flake8 to check for syntax errors or undefined variables names
2. Run Frun pytest
3. Build and push the latest docker image to ECR
4. update task in ECS


### Continuous Development

You cannot connect a GitHub repo to AWS, similar to Heroku. Instead, you must write the code that pushes the changes to AWS's appropriate service. In this work, I the docker_ecr.sh code uses AWS CLI to connect to AWS Elastic Container Registry (ECR), and then docker builds and push the new Docker image to that repository and tag this image as the latest.

![ECR](/screenshots/continuous_deployment1.png)

After updating the Docker image, I installed compose-cli and used docker context to connect to Elastic Container Service (ECS) and either deploy the 14 required resources, shown in ECS section of this README or update the current in-service cluster.

![ECS](/screenshots/continuous_deployment2.png)

### Load balance

The following images shows that the resources have launched in AWS to deploy the API

![Load balancer](/screenshots/3.png)

## API  call

The following shows that the API is working

![working api image](/screenshots/working_api.png)