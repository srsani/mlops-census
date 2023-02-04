# MlOps FastAPI

This repo contains the code for training a Random Forest model on the `census.csv` data set using `sci-kit-learn` and `hydra-core`. The data and artefacts for this project are stored in `S3` by using `DVC`.

The model is then served using FastAPI and AWS `ECR`, and `ECS`.

Some high-level `pytest` are also implemented to test the API performance.

Furthermore, I used Docker compose to deploy the `ECS` infrastructure to serve the docker image that was built and pushed to `ECR`.

Lastly, I have implemented a CI/CD pipeline using `GitHub Actions`.


This repo contain the code for train a RF
## Environment Set up

All the code have tested using python3.8 and requirements.txt

## Local dev/test:

Start the FastAPI server with :
-  `uvicorn src.main_api:app --reload`

API doc is under `http://127.0.0.1:8000/docs`
## ECS

- # docker context create ecs srs-fastapi
# docker context use srs-fastapi
# docker compose up


- Clean and delete all the resources `docker compose down`

* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.
