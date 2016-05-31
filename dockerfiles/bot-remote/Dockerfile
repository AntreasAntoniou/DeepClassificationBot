FROM classificationbot/deploy-base:latest

COPY ./requirements-bot.txt /tmp/
RUN pip install -r /tmp/requirements-bot.txt

WORKDIR /opt/bot
ENTRYPOINT ["/usr/bin/python", "deepanimebot/bot.py"]
