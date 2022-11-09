FROM --platform=linux/amd64 ubuntu:22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Set mode to non-interactive (only during build)
ARG DEBIAN_FRONTEND=noninteractive

# Install common useful linux tools
RUN apt-get update && apt-get install -y wget \
                                         vim \
                                         git

# Download & install VSCode code-server for interactive development
ARG CODE_SERVER_VERSION=4.2.0
WORKDIR /opt
RUN wget https://github.com/cdr/code-server/releases/download/v$CODE_SERVER_VERSION/code-server-$CODE_SERVER_VERSION-linux-amd64.tar.gz 
RUN tar -xzvf code-server-$CODE_SERVER_VERSION-linux-amd64.tar.gz
RUN rm code-server-$CODE_SERVER_VERSION-linux-amd64.tar.gz
RUN mv /opt/code-server-$CODE_SERVER_VERSION-linux-amd64 /opt/code-server
ENV PATH="/opt/code-server/bin:${PATH}"

# Create VSCode code-server config file (with random password) and expose vscode code-server port
RUN mkdir -p ~/.config/code-server/
RUN printf "bind-addr: 0.0.0.0:8337\nauth: password\npassword: $(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)\ncert: false" \
    >> ~/.config/code-server/config.yaml 
EXPOSE 8337

# Install vscode code-server extensions
RUN /opt/code-server/bin/code-server --install-extension ms-python.python

# Create some basic directories for binding HMGU ICB storages
RUN mkdir -p /lustre/groups/ /lustre/scratch/

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Create conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Copy environment file (for Charliecloud)
COPY environment /environment

# Remove retrieved packages from local cache and remove orphaned dependency packages 
RUN apt-get clean -y && apt-get autoremove -y