FROM classificationbot/base:latest
# You need to build with:
# $ docker build -t {tag} -f dockerfiles/ci/Dockerfile .
# (in order to copy requirements-dev.txt)

COPY ./requirements.txt /tmp/
COPY ./requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt
RUN pip install virtualenv
