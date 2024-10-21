FROM python:3.10

WORKDIR app

# training and perf analysis scripts
COPY dist-train.sh dist-train.sh
COPY dist-perf-analysis.sh dist-perf-analysis.sh

# codebase
COPY translate_ai translate_ai

# dataset
RUN mkdir -p data/
COPY data/english-spanish.csv data

# dependencies
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt