FROM nvidia/cuda:11.0.3-base-ubuntu20.04

COPY requirements.txt /

RUN apt update
RUN apt install -y python3-pip
RUN apt install -y git

RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3-opencv

RUN pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r /requirements.txt

WORKDIR /rl_scratch

COPY . /rl_scratch


CMD ["python3", "test_evaluation.py"]