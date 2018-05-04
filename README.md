# LSTM sentiment analysis on the Yelp-dataset 
[report](report.pdf)

## Preparation for the Group Model

Copy the example file `docker-compose-local-hilbert.yml` into a local version `docker-compose-local.yml` for use on the machine that you use. Then do:

```
mkdir checkpoints logs
docker build -t lstm_sentiment d
docker-compose -f docker-compose.yml -f docker-compose-local.yml up
```

## Setup for the group model

Everything is controlled through environment variables which are set in `docker-compose-local.yml`. In particular, the following ones are particularly relevant:

* `DATA_PATH` (e.g. `/data/small_data_train.json`) (required): set to the path of your training data
* `HOSTNAME` (e.g. `hilbert`): set to a meaningful name for your machine; this will be recorded in comet.ml
* `COMET_API_KEY` (e.g. `...8OXriYOPMvTRpsXwgg8tbcNi...`): set to your comet.ml API key if you want to plot your experiment progress on comet.ml
* `ENABLE_CUDA` (e.g. `False`): set to `True` if you want to want to run the training on a GPU
* `PROFILER` (e.g. `False`): set to `True` if you want to run the Pytorch profiler (much slower)
* `VALIDATION_DATA_PATH` (e.g. `/data/small_data_validate.json`): set to the path of your devset if you want to evaluate your model on the devset after each epoch



## Documentation of the original sentiment model

This project does sentiment analysis on the [yelp-review](https://www.yelp.com/dataset) dataset. This is done by modeling the star rating and the number of votes for the review being _cool_, _funny_ and _helpful_ as a function of the written text in the review.

The models evaluated are different combinations of recurrent neural networks, embeddings and convolutional neural networks. All the models are implemented in [PyTorch](http://pytorch.org/).


## Visualization
For visualization of training and other data [Visdom](https://github.com/facebookresearch/visdom) is used. This means that you have to start a Visdom server before being able to visualize anything.
``` 
python -m visdom.server
```


## Settings
A python script called 'settings.py' contains all the parameters that are meant to be modified for different test. This includes the number of epochs of training that should be performed, the learning rate of the optimizer, the model to be used (defined by a dictionary with the model to be used and the parameters for this model), the dataset to be used, if there should be visualization or not etc. The settings script also takes care of parsing of command line arguments for the other scripts and to see what can be set using these just run for example:
```
python train.py --help
```
For a list of supported arguments.

## Training a model
When the setting have been set properly training a model should be as easy as
```
python train.py --data-path PATH_TO_TRAIN_DATA
```
It is important to keep in mind that one has to provide own word-embeddings if one wishes to train models with external word embeddings.

## Evaluating model on new data
To evaluate a trained model on new data start by making sure that the model and dataset type in the settings script are the same as the one you want to evaluate.

Then evaluation should be performed through:
```
python evaluate.py --load-path PATH_TO_CHECKPOINT_OF_MODEL --data-path PATH_TO_EVAL_DATA
```
