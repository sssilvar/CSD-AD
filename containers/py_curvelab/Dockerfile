FROM python:2
MAINTAINER Santiago Smith <sssilvar@unal.edu.co>

# Install FFTW
RUN apt-get -y update && apt-get -y install fftw* swig bash

# Install CurveLab and pyCurvelab
COPY app /app
RUN tar -C /app -xvzf /app/CurveLab-2.1.3.tar.gz && \
    cd /app/CurveLab-2.1.3/fdct3d/src && make
#    cd /app/CurveLab-2.1.3/ && make && \
ENV FFTW=/usr/local/lib
ENV FFTW_DIR=/usr/local/lib
ENV FDCT=/app/CurveLab-2.1.3

RUN pip2 install numpy pandas sklearn nibabel scipy
RUN tar -C /app -xvzf /app/PyCurvelab-master.tar.gz && \
    cd /app/PyCurvelab-master && python2 setup.py build install

# Create work folders
RUN mkdir /input /output /scripts
ADD scripts /
RUN chmod +x /entrypoint.sh

# Create an entrypoint
ENTRYPOINT ["/entrypoint.sh"]
