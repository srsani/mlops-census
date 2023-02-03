FROM python:3.8

COPY requirements.txt .
RUN pip3 install -r requirements.txt 
ENV PYTHONUNBUFFERED=TRUE
COPY src /src

EXPOSE 8000
CMD uvicorn src.main_api:app --reload --host 0.0.0.0
