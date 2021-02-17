# Audio Super Resolution Tensorflow 2

TF2 implmentation of the Audio Super Resolution paper by V. Kuleshov, Z. Enam, and S. Ermon. Audio Super Resolution Using Neural Networks. ICLR 2017 

## Requirements

This implementation is based on the following
* `Tensorflow 2.4`
* `Librosa 0.8`
* `Matplolib 3.2.3`
* `Numpy 1.19`
* `Scipy 1.6`

## Model Implemented
Only AudioTFile is implemented here because the original repo stated that it is the best model.

## Setup
To install this package, simply clone the git repo:

`git clone git@github.com:speakupai/audio_super_res_tf2.git;
cd audio-super-res;`

### Creating Data
To process data follow the directions from the original repo https://github.com/kuleshov/audio-super-res. We followed the same instructions so this section is not modified.

## Running the model
To run the model simply run `make` in the terminal. This activates a command that will initiate the model with all the required parameters to train the model on a single speaker. However, it is possible to customize the training as shown below.

## CLI Instructions
Running the model is handled by the run.py script.

`usage: run.py train [-h] --train TRAIN --val VAL [-e EPOCHS]
                    [--batch-size BATCH_SIZE] [--logname LOGNAME]
                    [--layers LAYERS] [--alg ALG] [--lr LR] [--model MODEL] 
                    [--r R] [--piano PIANO] [--grocery GROCERY]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         path to h5 archive of training patches
  --val VAL             path to h5 archive of validation set patches
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  --batch-size BATCH_SIZE
                        training batch size
  --logname LOGNAME     folder where logs will be stored
  --layers LAYERS       number of layers in each of the D and U halves of the
                        network
  --alg ALG             optimization algorithm
  --lr LR               learning rate
  --model               the model to use for training (audiounet, audiotfilm, 
                                                       dnn, or spline). Defaults to audiounet.
  --r                   the upscaling ratio of the data: make sure that the appropriate 
                        datafile have been generated (note: to generate data with different
                        scaling ratios change the SCA parameter in the makefile)
  --piano               false by default--make true to train on piano data 
  --grocery             false by default--make true to train on grocery imputation data
  --speaker              number of speakers being trained on (single or multi). Defaults to single
  --pools_size          size of pooling window
  --strides             size of pooling strides
  --full                false by default--whether to calculate the "full" snr after each epoch. The "full" snr 
                        is the snr acorss the non-patched data file, rather than the average snr over all the 
                        patches which is calculated by default
`
                        

For example, to run the model on data prepared for the single speaker dataset, you would type:

`python run.py train \
  --train ../data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5 \
  --val ../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 \
  -e 120 \
  --batch-size 64 \
  --lr 3e-4 \
  --logname singlespeaker \
  --model audiotfilm \
  --r 4 \
  --layers 4 \
  --piano false \
  --pool_size 8 \
  --strides 8
  --full true`
 
The above run will store checkpoints in ./singlespeaker.lr0.000300.1.g4.b64.
