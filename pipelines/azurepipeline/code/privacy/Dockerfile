FROM tensorflow/tensorflow:2.0.0a0-gpu-py3

# pip install
COPY requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

COPY train.py /scripts/train.py
COPY dp_covariance.py /scripts/dp_covariance.py
COPY dp_lin_reg.py /scripts/dp_lin_reg.py
COPY pipelineWrapper.py /scripts/pipelineWrapper.py

# python train.py -o model
# will be overwritten by kf pipeline
ENTRYPOINT [ "python", \
            "/scripts/train.py", \
            "-o", "model" ]
