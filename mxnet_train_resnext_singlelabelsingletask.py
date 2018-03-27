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
# reference:
# https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/fine-tune.py
#
# Author: hzhumeng01 2018-01-22
# copyright @ XXX


import argparse
import os, sys

# for import the docker based mxnet version
mxnet_root = "/mxnet/"
sys.path.insert(0, mxnet_root + 'python')
import mxnet as mx

import importlib
import find_mxnet
import time

sys.path.insert(0, "./settings")
sys.path.insert(0, "../")

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# for fine-tuning
def get_fine_tune_model(sym, arg_params, num_classes, layer_name, batchsize):

    all_layers = sym.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data = net, num_hidden = num_classes, name = 'fc')
    net = mx.symbol.SoftmaxOutput(data = net, name = 'softmax')

    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


# learing rate step size setup
def multi_factor_scheduler(begin_epoch, epoch_size, step=[5, 10, 15], factor=0.1):

    step_ = [epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step = step_, factor = factor) if len(step_) else None


def train_model(model, gpus, batch_size, image_shape, epoch=0, num_epoch = 20, kv = 'device'):

    train = mx.image.ImageIter(
        batch_size   = args.batch_size,
        data_shape   = (3, 224, 224),
        label_width  = 1,
        path_imglist = args.data_train,
        path_root    = args.image_train,
        part_index   = kv.rank,
        num_parts    = kv.num_workers,
        shuffle      = True,
        data_name    = 'data',
        label_name   = 'softmax_label',
        aug_list     = mx.image.CreateAugmenter((3, 224, 224), resize=224, rand_crop=True, rand_mirror=True, mean=True, std=True)
    )

    val = mx.image.ImageIter(
        batch_size   = args.batch_size,
        data_shape   = (3, 224, 224),
        label_width  = 1,
        path_imglist = args.data_val,
        path_root    = args.image_val,
        part_index   = kv.rank,
        num_parts    = kv.num_workers,
        data_name    = 'data',
        label_name   = 'softmax_label',
        aug_list     = mx.image.CreateAugmenter((3, 224, 224), resize=224, mean=True, std=True)
    )

    kv = mx.kvstore.create(args.kv_store)

    prefix = model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # flatten0: for resnext-50-symbol.json
    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, args.num_classes, 'flatten0', args.batch_size)

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr_scheduler = multi_factor_scheduler(args.epoch, epoch_size)

    optimizer_params = {
        'learning_rate': args.lr,
        'momentum': args.mom,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler}

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    model = mx.mod.Module(
        context = devs,
        symbol  = new_sym
    )

    checkpoint = mx.callback.do_checkpoint(args.save_result)

    eval_metric = ['accuracy']

    model.fit(
        train_data         = train,
        begin_epoch        = epoch,
        num_epoch          = num_epoch,
        eval_data          = val,
        eval_metric        = eval_metric,
        kvstore            = kv,
        optimizer          = 'sgd',
        optimizer_params   = optimizer_params,
        arg_params         = new_args,
        aux_params         = aux_params,
        initializer        = initializer,
        allow_missing      = True,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 20),
        epoch_end_callback = checkpoint
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'train a model on a dataset')
    parser.add_argument('--model',       type = str,   default = '/root/mxnet_dpn/models/models_org/resnext-50', required = True)
    parser.add_argument('--gpus',        type = str,   default = '0')
    parser.add_argument('--batch-size',  type = int,   default = 32)
    parser.add_argument('--epoch',       type = int,   default = 30)
    parser.add_argument('--image-shape', type = str,   default = '3,224,224')
    parser.add_argument('--data-train',  type = str,   default = '/root/mxnet_dpn/mxnet/tools/mnist224_train.lst')
    parser.add_argument('--image-train', type = str,   default = '/root/mxnet_datasets/')
    parser.add_argument('--data-val',    type = str,   default = '/root/mxnet_dpn/mxnet/tools/mnist224_test.lst')
    parser.add_argument('--image-val',   type = str,   default = '/root/mxnet_datasets/')
    parser.add_argument('--num-classes', type = int,   default = 10)
    parser.add_argument('--lr',          type = float, default = 0.01)
    parser.add_argument('--num-epoch',   type = int,   default = 30)
    parser.add_argument('--kv-store',    type = str,   default = 'device', help = 'the kvstore type')
    parser.add_argument('--save-result', type = str,   default = '/root/mxnet_dpn/models/mnist224_resnext50_SLST/resnext50',
                        help = 'the save path')
    parser.add_argument('--num-examples',type = int,   default = 60000)
    parser.add_argument('--mom',         type = float, default = 0.9,    help = 'momentulm for sgd')
    parser.add_argument('--wd',          type = float, default = 0.0005, help = 'weight decay for sgd')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    kv = mx.kvstore.create(args.kv_store)

    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)

    hdlr = logging.FileHandler(args.save_result + '/train.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logging.info(args)

    train_model(
        model       = args.model,
        gpus        = args.gpus,
        batch_size  = args.batch_size,
        image_shape = '3,224,224',
        epoch       = args.epoch,                # eg: epoch = 5, begin training in 5th epoch, like fine-tuning in caffe
        num_epoch   = args.num_epoch,
        kv          = kv
    )
