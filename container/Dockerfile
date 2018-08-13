# Docker container for Pycurvelab
FROM python:2
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

# Create data folder and set an ENV var
RUN mkdir /root/data /py
ENV DATA_FOLDER=/root/data

# Creat app folder
ADD ./software /app
RUN chmod -R 777 /app

# Add the scripts to the Docker container
ADD scripts /root/scripts
RUN chmod -R 777 /root/scripts

# Install dependencies
RUN pip install --no-cache-dir -r /root/scripts/requirements.txt

# Install some stuff
RUN apt-get update && apt-get -y install fftw* swig
RUN cd /app/CurveLab-2.1.3 && make
RUN export FFTW="/usr/local/lib" && export FDCT="/app/CurveLab-2.1.3" &&\
    cd /app/PyCurvelab/ && python setup.py install && python2.7 setup.py install

# Define an entrypoint
ENTRYPOINT ["/root/scripts/entrypoint.sh"]