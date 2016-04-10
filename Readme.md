**Classification Bot**
----------------------
Welcome to the Classification Bot codebase. Classification Bot is an attempt of simplifying the collection, extraction and preprocessing of data as well as providing an end to end pipeline for using them to train large deep neural networks.

The system is composed of scrapers, data_extractors, preprocessors, deep neural network models using [Keras](https://github.com/fchollet/keras) provided by [Francois Chollet](https://github.com/fchollet)  and an easy to use deployment module.

**To Install**
Make sure you have a GPU as the training is very compute intensive

1. Install CUDA_toolkit 7.5
2. Install cuDNN 4
3. Install theano, using `sudo pip install git+git://github.com/Theano/Theano.git`
4. Make sure you have Python 2.7.6 and virtualenv installed on your system
5. Install Python dependencies

```
$ virtualenv --python=python2 env
$ . env/bin/activate
$ pip install -r requirements.txt
```

**To download images**

Easy Mode:
  1. Edit name_extractor.py and add a get_* method. For your convenience we provide a get_cars() method to showcase a use-case.
  2. Run `python name_extractor.py cars > cars.csv` to create a csv with the names you want to download images of.
  3. Now let's download some images! Run `python google_image_scraper.py cars.csv`

  **Note**: Make sure to replace cars with your own item names otherwise you will be using cars, unless you modified the get_cars method to get something other than cars, in which case you should really rename your method for proper code readability.

Hacker Mode (Not really useful unless you know what you are doing):
1. Edit google_image_scraper.py using your favourite editor. In the main loop add classes in the "list" or use your own custom category extractor to find classes online, such as species of dogs.
types of fruits, names of gundams etc. Set the number of images to the number of images per class, we suggest a number between 200-1000.
2. Then run `python google_image_scraper.py` to get all of the required images.
3. Wait until images have been downloaded. You should be able to see them in folders under downloaded_images, in classes.

**To extract and preprocess data ready for training**

1. Once you have your data ready, run `python train.py extract_data` to get all of your data ready and saved in HDF5 files.

**To train your network**

1. Once all of the above have been met then you are ready to train your network, by running `python train.py --run` to load data from HDF5 files or `python train.py --run --extract_data` to extract data and train in one procedure.
2. If you want to continue training a model, you can. After each epoch the weights are saved. If you want to continue training simply run `python train.py --run --continue`


**Deploying a model**

1. Once your training has finished and a good model has been trained then you can deploy your model.
2. To deploy a model on a single URL image use `python deploy.py --URL [URL_LINK]`
3. To deploy a model on a folder full of images use `python deploy -image_folder path/to/folder`
4. To deploy a model on a single file use `python deploy -image_file path/to/file`

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

**Getting started**

Things to try:

1. Create your own classifiers
2. Try different model architectures (Hint: go to google scholar or arxiv and search for GoogLeNet, VGG-Net, AlexNet, ResNet and follow the waves :) )


**Special Thanks**
Special thanks to Francois Chollet (fchollet) for building the superb [Keras](https://github.com/fchollet/keras) deep learning library.
We couldn't have brought a project ready to be used by non-machine learning people if it wasn't for the ease of use of Keras.

Special thanks to https://github.com/shuvronewscred/ for building the image scraper we adapted for our project.
Original source code can be found at https://github.com/shuvronewscred/google-search-image-downloader
