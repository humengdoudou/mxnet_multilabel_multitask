#
# This is the python code for mxnet based single label single task (SLST for short) classification mission
# actually, I test the following code in the DPNs based mxnet version,
# but it is easy to transfer the follow code to official mxnet.
#
# the code use mxnet.image.ImageIter to handle the images and .lst data reader,
# but not the mxnet.io.ImageRecordIter to handle the .rec and .lst data reader
#
# the .lst file organized like the following:
# 685551	12.000000	pavilion/00006962.jpg
# 1309299	24.000000	wood_house/00016810.jpg
# 704968	13.000000	plane/00005464.jpg
# 992439	18.000000	swimming_pool/00003219.jpg
# 3537	    0.000000	aquarium_underwater/00004537.jpg
# 1004156	18.000000	swimming_pool/00002370.jpg
# 1262962	23.000000	window/00003901.jpg
# 1108990	20.000000	tower/00005627.jpg
# 365688	6.000000	crops_field/00003949.jpg
#
# each line denotes a train/val image example
# the 1st column is the image index
# the 2nd column is the image class
# the 3rd column is the relative image path
#
# in the following code, I use the trained model by mxnet_train_resnext_singlelabelsingletask.py to do the forward pass
#
# reference:
# https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/fine-tune.py
#
# Author: hzhumeng01 2018-01-30
# copyright @ XXX


import argparse
import os, sys

# for import the docker based mxnet version
mxnet_root = "/mxnet/"
sys.path.insert(0, mxnet_root + 'python')
import mxnet as mx

import numpy as np

import importlib
import find_mxnet
import time

sys.path.insert(0, "./settings")
sys.path.insert(0, "../")

import logging

IMAGE_DEPTH    = 3

DEBUG = 1

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def load_model(prefix, index, data_names, data_shapes, label_names = None, label_shapes = None, context = mx.gpu()):

    symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, index)
    model = mx.mod.Module(symbol      = symbol,
                          context     = context,
                          data_names  = data_names,
                          label_names = label_names)

    model.bind(for_training = False,
               data_shapes  = data_shapes,
               label_shapes = label_shapes)

    model.set_params(arg_params = arg_params, aux_params = aux_params)
    return symbol, model


def load_classifier(prefix, index, batch_size, img_width, img_height):

    data_names = ['data']
    data_shapes = [('data', (batch_size, IMAGE_DEPTH, img_height, img_width))]

    label_names = ['softmax_label']
    label_shapes = [('softmax_label', (batch_size,))]

    symbol, model = load_model(prefix,
                               index,
                               data_names   = data_names,
                               data_shapes  = data_shapes,
                               label_names  = label_names,
                               label_shapes = label_shapes,
                               context      = mx.gpu(0))

    return symbol, model


def transform(data, augmenters):

    for aug in augmenters:
        data = aug(data)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'score a model on a dataset')
    parser.add_argument('--model',        type = str,   default = '/root/mxnet_dpn/models/mnist224_resnext50_SLST/resnext50', required = True)
    parser.add_argument('--gpus',         type = str,   default = '0,1')
    parser.add_argument('--batch-size',   type = int,   default = 1)
    parser.add_argument('--epoch',        type = int,   default = 3)
    parser.add_argument('--image-shape',  type = str,   default = '3,224,224')
    parser.add_argument('--image-txt',    type = str,   default = '/root/mxnet_dpn/mnist224_test.txt')
    parser.add_argument('--image-folder', type = str,   default = '/root/mxnet_datasets/')
    parser.add_argument('--num-classes',  type = int,   default = 10)
    parser.add_argument('--lr',           type = float, default = 0.01)
    parser.add_argument('--num-epoch',    type = int,   default = 30)
    parser.add_argument('--kv-store',     type = str,   default = 'device', help = 'the kvstore type')
    parser.add_argument('--num-examples', type = int,   default = 60000)
    parser.add_argument('--mom',          type = float, default = 0.9,    help = 'momentulm for sgd')
    parser.add_argument('--wd',           type = float, default = 0.0005, help = 'weight decay for sgd')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # -----1----- init & load model
    image_shape = [int(i.strip()) for i in args.image_shape.split(',')]

    batch_size        = args.batch_size
    img_width         = image_shape[1]
    img_height        = image_shape[2]
    classifier_prefix = args.model
    classifier_index  = args.epoch
    sym, model = load_classifier(classifier_prefix, classifier_index, batch_size, img_width, img_height)

    # -----2----- image pre-process
    cast_aug   = mx.image.CastAug()
    resize_aug = mx.image.ForceResizeAug(size = (img_width, img_height))
    augmenters_classifier = [cast_aug, resize_aug]

    mean = np.array([123.68, 116.28, 103.53])
    std  = np.array([58.395, 57.12,  57.375])

    # -----3----- image list
    with open(args.image_txt, "r") as f_read:
        image_list = f_read.readlines()
    assert len(image_list) > 0, "No valid image specified to detect"

    test_count = 0

    # -----4----- classify
    for image_name in image_list:
        image_path = args.image_folder + image_name.strip()

        # -----4.1----- image read & convert
        with open(image_path, 'rb') as fp:
            img_content = fp.read()
        img_ndarray = mx.img.imdecode(img_content)

        # -----4.2----- image preprocess
        data_aug       = transform(img_ndarray, augmenters_classifier)
        data_normal    = mx.image.color_normalize(data_aug, mx.nd.array(mean), mx.nd.array(std))  # mean and std, denpends on training schedule
        data_transpose = mx.nd.transpose(data_normal, axes=(2, 0, 1))                             # mean and std
        # data_transpose = mx.nd.transpose(data_aug, axes=(2, 0, 1))                              # no mean and std
        data_expand    = mx.nd.expand_dims(data_transpose, axis = 0)

        # -----4.3----- forward
        model.forward(mx.io.DataBatch((data_expand,)))

        # -----4.4----- output
        pred_prob = model.get_outputs()[0].asnumpy()
        pred = np.argmax(pred_prob, axis = 1)

        if DEBUG:
            print (image_path, pred)
            gt_label = int(image_path.split('/')[-2])
            if gt_label == pred[0]:
                test_count += 1
            print test_count

    print "acc: ", test_count * 1.0 / len(image_list)