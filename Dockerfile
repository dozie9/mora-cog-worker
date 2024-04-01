ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dependencies

# Install necessary packages and Python 3.10
RUN apt-get update && apt-get upgrade -y

# Create a virtual environment
# RUN python3 -m venv /opt/venv

COPY requirements.txt .

# Install runpod within the virtual environment
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ADD src/handler.py /rp_handler.py
ADD src/mora.py /mora.py

CMD ["python", "-u", "/rp_handler.py"]
