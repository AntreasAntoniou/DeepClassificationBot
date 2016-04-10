FROM kaixhin/theano:latest
# You need to build with:
# $ docker build -t tag -f dockerfiles/ci/Dockerfile .
# (in order to copy requirements.txt)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y libffi-dev libssl-dev libhdf5-dev language-pack-en-base python-pandas python-opencv
# disable camera driver: http://stackoverflow.com/a/26028597/20226
RUN ln /dev/null /dev/raw1394
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install virtualenv
