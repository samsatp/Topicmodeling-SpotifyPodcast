#FROM python:3.8-slim
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

RUN apt-get -y update && apt-get -y upgrade

COPY . /web
WORKDIR /web

RUN pip install --upgrade pip
RUN pip install -r requirements-dev.txt

EXPOSE 80

CMD [ "uvicorn", "backend.main:app" , "--host", "0.0.0.0", "--port", "80"]