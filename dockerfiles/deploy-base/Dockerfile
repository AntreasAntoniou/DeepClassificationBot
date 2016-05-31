FROM classificationbot/base:latest

RUN mkdir -p /opt/bot
COPY data /opt/bot/data
COPY pre_trained_weights /opt/bot/pre_trained_weights

ENV PYTHONPATH $PYTHONPATH:/opt/bot
COPY gceutil.py data.py deploy.py model.py /opt/bot/
COPY deepanimebot /opt/bot/deepanimebot
