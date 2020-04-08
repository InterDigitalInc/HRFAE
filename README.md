## HRFAE: High Resolution Face Age Editing

Official implementation for paper *High Resolution Face Age Editing*.

![Teaser image](./arch.png)

## Dependencies

* Python 3.7
* Pytorch 1.1
* Numpy
* Opencv
* TensorboardX
* Tensorboard_logger

You can also create a new environment for this repo by running
```
conda env create -f env.yml
```

## Load and test pretrained network 

1. Download the [pretrained model](https://partage.imt.fr/index.php/s/rXyiDj6xR9irJ6z) to the folder `logs/001/`.

2. Upload test images in the folder `/test/input` and run the test file. The output images will be saved in the folder `/test/output`.
    ```
    python test.py --config 001 --target_age 65
    ```

## Train a new model

1. Pretrained age classifier

    To get age information, we use an age classifier pretrained on [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset. We use the model released from paper [Deep expectation of real and apparent age from a single image without facial landmarks](https://data.vision.ee.ethz.ch/cvl/publications/papers/articles/eth_biwi_01299.pdf) by Rothe et al. 

    To prepare the model, you need to download the original [caffe model](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel) and convert it to PyTorch format. We use the converter [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) released by Vadim Kantorov. Then name the PyTorch model as `dex_imdb_wiki.caffemodel.pt` and put it in the folder `/models`.


2. Preparing your dataset

    Download [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset and unzip it to the `/data/ffhq` directory. 
    
    Download [age label](https://partage.imt.fr/index.php/s/DbSk4HzFkeCYXDt) to the `/data` directory.

    You can also train the model with your own dataset. Put your images in the `/data` directory. With the pretrained classifier, you can create a new label file with the age of each image. 

3. Training
    
    You can modify the training options of the config file in `configs` directory.
    ```
    python train.py --config 001 
    ```

## Reference

## License




