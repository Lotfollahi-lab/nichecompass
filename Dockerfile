FROM nvidia/cuda:11.7.1-cudnn8-runtime-rockylinux8

RUN dnf install python39 python39-devel python39-setuptools -y
RUN easy_install-3.9 pip
RUN pip install poetry==1.4.2

RUN mkdir -p /app

COPY . /app

WORKDIR /app

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "-m", "nichecompass"]
