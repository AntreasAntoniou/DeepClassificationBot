FROM classificationbot/base:latest

RUN mkdir -p /opt/bot/default_workspace
COPY default_workspace/data /opt/bot/default_workspace/data
COPY default_workspace/pre_trained_weights /opt/bot/default_workspace/pre_trained_weights

ENV PYTHONPATH $PYTHONPATH:/opt/bot
COPY gceutil.py data.py deploy.py model.py /opt/bot/
COPY deepanimebot /opt/bot/deepanimebot
