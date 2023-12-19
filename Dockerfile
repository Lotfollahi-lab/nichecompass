FROM nvidia/cuda:11.7.1-cudnn8-runtime-rockylinux8

RUN dnf install python39 python39-devel python39-setuptools gcc gcc-c++ zlib-devel git -y

RUN python3.9 -m venv /poetry
RUN /poetry/bin/pip install -U pip setuptools
RUN /poetry/bin/pip install poetry==1.7.1

COPY . /nichecompass
RUN cd nichecompass && /poetry/bin/poetry config virtualenvs.create false && /poetry/bin/poetry install

ENTRYPOINT ["nichecompass"]
