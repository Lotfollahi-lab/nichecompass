FROM nvidia/cuda:11.7.1-cudnn8-runtime-rockylinux8

RUN dnf install python39 -y
RUN python3.6 -m ensurepip
RUN pip install poetry

RUN mkdir -p /app

COPY . /app

WORKDIR /app

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "-m", "nichecompass"]
