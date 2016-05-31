# Deprecated: Standalone bot deployment

This is an image for deploying the bot behind supervisord.

## Creating and deleting your Google Compute instance

`tasks.py` provides a handy shortcut for creating an Google Compute instance
with the Docker image specified in `etc/standalone-bot-containers.yaml`.
Twitter credentials are pulled from `bot.ini` and stored as instance metadata.

```
$ python tasks.py create_standalone_instance
$ python tasks.py delete_standalone_instance
```

#### When something goes wrong

Or when you want to see if it's working for yourself:

```
## SSH into your instance
$ gcloud compute ssh --zone us-central1-a bot-standalone

## Wait until our container comes up:
you@bot:~$ sudo watch docker ps

## If it appears to be stuck, check kubelet's log:
you@bot:~$ sudo less /var/log/kubelet.log

## Once it's up, you can drop into its shell:
you@bot:~$ sudo docker exec -it $(sudo docker ps --filter=ancestor=classificationbot/bot -q) bash

## And run supervisorctl to check the bot.py process
# supervisorctl

## You can run it manually too:
# cd /opt/bot/
# python deepanimebot/bot.py --mock --debug
```
