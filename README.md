Classification Bot
------------------

[![Join the chat at https://gitter.im/AntreasAntoniou/DeepClassificationBot](https://badges.gitter.im/AntreasAntoniou/DeepClassificationBot.svg)](https://gitter.im/AntreasAntoniou/DeepClassificationBot?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
Welcome to the Classification Bot codebase. Classification Bot is an attempt of simplifying the collection, extraction and preprocessing of data as well as providing an end to end pipeline for using them to train large deep neural networks.

The system is composed of scrapers, data extractors, preprocessors, deep neural network models using [Keras](https://github.com/fchollet/keras) provided by [Francois Chollet](https://github.com/fchollet) and an easy to use deployment module.

## Installation
Make sure you have a GPU as the training is very compute intensive

1. (OSX) Install gcc: `brew install gcc`
2. Install CUDA_toolkit 7.5
3. Install cuDNN 4
4. Install Theano, using `sudo pip install git+git://github.com/Theano/Theano.git`
5. Install OpenCV
6. Install hdf5 library (libhdf5-dev)
7. Make sure you have Python 2.7.6 and virtualenv installed on your system
8. Install Python dependencies

```
$ virtualenv --python=python2 --system-site-packages env
$ . env/bin/activate
$ pip install -r requirements.txt
```

## Training and deploying

### To download images

Use `google_image_scraper.py` to download images. It takes a .csv file of the categories you want, and downloads a number of images per line.

The first line of the .csv file will be ignored.

The number of images per category is configurable. We suggest a number between 200-1000:

```
$ google_image_scraper.py -n 200 yourfilehere.csv
```

#### Easy Mode:
(For users that have a list of categories available at hand):

1. Create a .csv file with one category per line of what you want the scraper to search for.
2. Now let's download some images! Run `python google_image_scraper.py yourfilehere.csv`

#### Hacker Mode:
 (For users that know an online repo that has their categories and want to fetch them, or if their categories are too many and you MUST automate the procedure, or if you much rather code stuff rather than copy and paste)

1. Write a script that can fetch your categories using Wikipedia or any other resource you would like. For an example look at `examples/anime_names.py` to see what we used to get our categories.
2. Have your script create a .csv file with the categories you require.
3. Then run `python google_image_scraper.py yourfilehere.csv`

### To extract and preprocess data ready for training

1. Once you have your data ready, run `python train.py extract_data` to get all of your data ready and saved in HDF5 files.

### To train your network

1. Once all of the above have been met then you are ready to train your network, by running `python train.py --run` to load data from HDF5 files or `python train.py --run --extract_data` to extract data and train in one procedure.
2. If you want to continue training a model, you can. After each epoch the weights are saved. If you want to continue training simply run `python train.py --run --continue`


### Deploying a model

1. Once your training has finished and a good model has been trained then you can deploy your model.
2. To deploy a model on a single URL image use `python deploy.py --URL [URL_LINK]`
3. To deploy a model on a folder full of images use `python deploy --image-folder path/to/folder`
4. To deploy a model on a single file use `python deploy --image-path path/to/file`

Once deployed the model should return the top 5 predictions on each image in a nice string formatted view: e.g.

```
Image Name: Tengen.Toppa.Gurren-Lagann.full.174481.jpg
Categories:
0. Gurren Lagann: 0.999914288521
1. Kill La Kill: 7.29278544895e-05
2. Naruto: 4.92283288622e-06
3. Redline: 2.71744352176e-06
4. Cowboy Bebop: 1.41406655985e-06
_________________________________________________
```

### Things for you to try

1. Create your own classifiers
2. Try different model architectures (Hint: go to google scholar or arxiv and search for GoogLeNet, VGG-Net, AlexNet, ResNet and follow the waves :) )

## Twitter bot

`deepanimebot/bot.py` is a Twitter bot that provides an interface for querying the classifier.

### Running the bot locally

#### Prerequisites

* A classifier
* [A Twitter app](https://apps.twitter.com/) registered under the bot account
* Consumer key and secret for that app
* [Your access token and secret for that app](https://dev.twitter.com/oauth/overview/application-owner-access-tokens)

Copy `bot.ini.example` to `bot.ini` and overwrite with your key/secret and token/secret.

#### Run it

```
$ python deepanimebot/bot.py -c bot.ini --debug
```

`python deepanimebot/bot.py --help` will list all available command line options.

### Deploying to Google Compute Engine

This repo comes with the necessary support files for deploying the Twitter bot
to a dedicated GCE container-optimized instance.

#### Prerequisites

* A classifier
* `bot.ini` with Twitter credentials (see above)
* [Docker](https://www.docker.com/) tools and an account on a docker registry
* [Google Cloud SDK](https://cloud.google.com/sdk/#Quick_Start)
* [A Google Cloud Platform project](https://cloud.google.com/compute/docs/linux-quickstart#set_up_a_google_cloud_platform_project)

#### Build and register your own docker image

`classificationbot/base:latest` comes with all the dependencies installed.
If you've modified the code and added a new dependency,
make a new Docker image based on the dockerfiles in this repo.

`dockerfiles/bot/Dockerfile` will contain the bot and the classifier when built.

This repo's associated images are built with these commands:

```
$ docker build -t classificationbot/base:latest -f dockerfiles/base/Dockerfile .
$ docker push classificationbot/base:latest

$ docker build -t classificationbot/ci:latest -f dockerfiles/ci/Dockerfile .
$ docker push classificationbot/ci:latest

$ docker build -t classificationbot/bot-standalone:latest -f dockerfiles/bot-standalone/Dockerfile .
$ docker push classificationbot/bot-standalone:latest
```

When you've registered your own image, update the `image` value in `etc/standalone-bot-containers.yaml`.

#### Creating and deleting your instance

`tasks.py` provides a handy shortcut for creating an instance
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
# python bot.py --mock --debug
```

## Special Thanks
Special thanks to Francois Chollet (fchollet) for building the superb [Keras](https://github.com/fchollet/keras) deep learning library.
We couldn't have brought a project ready to be used by non-machine learning people if it wasn't for the ease of use of Keras.

Special thanks to https://github.com/shuvronewscred/ for building the image scraper we adapted for our project.
Original source code can be found at https://github.com/shuvronewscred/google-search-image-downloader
