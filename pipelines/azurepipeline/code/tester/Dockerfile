FROM python:3.8-slim-buster
COPY requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

COPY test.py /scripts/test.py
COPY pipelineWrapper.py /scripts/pipelineWrapper.py

# python train.py -o model
# will be overwritten by kf pipeline
ENTRYPOINT [ "python", \
            "/scripts/test.py" ]
