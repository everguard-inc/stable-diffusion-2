FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY eg_data_tools eg_data_tools/
RUN pip install -e eg_data_tools

RUN pip uninstall opencv-python-headless opencv-contrib-python opencv-python -y
RUN pip install opencv-python==4.5.5.64

ENV APP_PATH="/app"
RUN mkdir -p ${APP_PATH}
WORKDIR ${APP_PATH}

ENV PYTHONPATH "${PYTHONPATH}:/app/"