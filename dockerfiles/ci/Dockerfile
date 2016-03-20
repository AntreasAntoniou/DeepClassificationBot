FROM kaixhin/theano:latest
# You need to build with:
# $ docker build -t tag -f dockerfiles/ci/Dockerfile .
# (in order to copy requirements.txt)

RUN apt-get update && apt-get install -y python-pandas
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install virtualenv
