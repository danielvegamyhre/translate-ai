FROM python:3.10

WORKDIR app

# training and perf analysis scripts
COPY *.sh .

# codebase
COPY translate_ai translate_ai

# dataset
RUN mkdir -p data/
COPY data/english-spanish.csv data

# dependencies
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt