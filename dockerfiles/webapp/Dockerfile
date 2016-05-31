FROM classificationbot/deploy-base:latest

COPY ./requirements-webapp.txt /tmp/
RUN pip install -r /tmp/requirements-webapp.txt

WORKDIR /opt/bot
ENTRYPOINT ["/usr/local/bin/gunicorn", "-b", ":80", "deepanimebot.wsgi:app"]
