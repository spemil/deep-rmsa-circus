# syntax=docker/dockerfile:1
#FROM python:3.8-slim-buster
FROM python:3.9

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 5000
CMD ["python", "Deep_RMSA_A3C_configtable.py"]
