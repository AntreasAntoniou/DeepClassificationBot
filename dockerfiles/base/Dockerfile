FROM kaixhin/theano:latest
# You need to build with:
# $ docker build -t {tag} -f dockerfiles/base/Dockerfile .
# (in order to copy requirements.txt)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y libffi-dev libssl-dev libhdf5-dev language-pack-en-base python-pandas python-opencv python-numpy

COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
