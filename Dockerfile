FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

