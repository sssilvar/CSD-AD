# Docker container for Pycurvelab
FROM ubuntu:18.04
MAINTAINER Santiago Smith <sssilvar@unal.edu.co>

# Set proxy (if necessary) --build-arg
ARG proxy
ENV http_proxy=$proxy

ARG scales
ENV SCALES=$scales

ARG angles
ENV ANGLES=$angles

# Prevent dpkg errors
ENV TERM=xterm-256color
ENV DEBIAN_FRONTEND=noninteractive

# Create data folder and set an ENV var
RUN mkdir /root/data /py
ENV DATA_FOLDER=/root/data

# Creat app folder
ADD ./software /app
RUN chmod -R 777 /app

# Add the scripts to the Docker container
ADD scripts /root/scripts
RUN chmod -R 777 /root/scripts

# Install some stuff
RUN apt-get -y update && apt-get -y install build-essential python2.7 python-pip python-numpy python-tk fftw* swig
RUN cd /app/CurveLab-2.1.3 && make
RUN export FFTW="/usr/local/lib" && export FDCT="/app/CurveLab-2.1.3" &&\
    cd /app/PyCurvelab/ && python setup.py install && python2.7 setup.py install

# Install dependencies
RUN pip2 install --no-cache-dir -r /root/scripts/requirements.txt

# Define an entrypoint
ENTRYPOINT ["/root/scripts/entrypoint.sh"]