# docker build -t classificationbot/bot:latest -f dockerfiles/bot/Dockerfile .
FROM classificationbot/base:latest

RUN pip install supervisor
RUN mkdir -p /opt/bot
COPY bot.py gceutil.py data.py deploy.py model.py /opt/bot/
COPY data /opt/bot/data
COPY pre_trained_weights /opt/bot/pre_trained_weights
COPY etc/supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["/usr/local/bin/supervisord"]
