FROM nvidia/cuda:11.7.1-cudnn8-runtime-rockylinux8

RUN dnf install python39 python39-devel python39-setuptools gcc gcc-c++ zlib-devel git -y
RUN easy_install-3.9 pip
RUN pip install poetry==1.7.1

COPY . /nichecompass
RUN cd nichecompass && poetry config virtualenvs.create false && poetry install

ENTRYPOINT ["nichecompass"]
