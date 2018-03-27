#
# This is the python code for mxnet based single label multi task (SLMT for short) classification mission
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
# so I use the single label to do the multi task classifitation
# the 1st task is class each image like in mxnet_train_resnext_singlelabelsingletask.py
# the 2nd task is determine whether the output label is larger than 10, if >10, output 1, else 0.
#
# reference:
# 1 https://github.com/apache/incubator-mxnet/blob/master/example/multi-task/example_multi_task.py
# 2 https://github.com/hariag/mxnet-multi-task-example/blob/master/multi-task.ipynb
# 3 https://github.com/miraclewkf/multi-task-MXNet/blob/master/train_multitask.py
# 4 https://github.com/cypw/DPNs
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

MULTITASK_LABEL2_THRES = 5

class MultiTask_iterator(mx.io.DataIter):

    '''multi label mnist iterator'''
    def __init__(self, data_iter):
        super(MultiTask_iterator, self).__init__('multitask_iter')
        self.data_iter  = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # Different labels should be used here for actual application
        return [('softmax_multitask1_label', provide_label[1]), \
                ('softmax_multitask2_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch  = self.data_iter.next()
        label1 = batch.label[0]

        # new label based on the original label, output 0 or 1 here
        label2 = mx.nd.array(label1.asnumpy() > MULTITASK_LABEL2_THRES).astype('float32')

        return mx.io.DataBatch(data  = batch.data,
                               label = [label1, label2],
                               pad   = batch.pad,
                               index = batch.index)


# define multi task accuracy
class MultiTask_Accuracy(mx.metric.EvalMetric):

    def __init__(self, num = None, output_names = None):
        self.num = num
        super(MultiTask_Accuracy, self).__init__('multi_accuracy', num)
        self.output_names = output_names

    def reset(self):
        ''' Resets the internal evaluation result to initial state.'''
        self.num_inst   = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst   += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i]   += len(pred_label.flat)

    def get(self):
        if self.num is None:
            return super(MultiTask_Accuracy, self).get()
        else:
            return zip(*(('%s-task%d' % (self.name, i), float('nan') if self.num_inst[i] == 0
            else self.sum_metric[i] / self.num_inst[i])
                         for i in range(self.num)))


# for fine-tuning for the SLMT
def get_fine_tune_model(sym, num_classes_mt1, num_classes_mt2, layer_name):

    all_layers = sym.get_internals()
    net = all_layers[layer_name + '_output']

    # task1
    fc_multitask1  = mx.symbol.FullyConnected(data = net, num_hidden = num_classes_mt1, name = 'fc_multitask1')
    smo_multitask1 = mx.symbol.SoftmaxOutput(data = fc_multitask1, name = 'softmax_multitask1')

    # task2
    fc_multitask2  = mx.symbol.FullyConnected(data = net, num_hidden = num_classes_mt2, name = 'fc_multitask2')
    smo_multitask2 = mx.symbol.SoftmaxOutput(data = fc_multitask2, name = 'softmax_multitask2')

    softmax_group = mx.symbol.Group([smo_multitask1, smo_multitask2])

    return softmax_group


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
        aug_list     = mx.image.CreateAugmenter((3, 224, 224), resize=224, mean=True, std=True)
    )

    train = MultiTask_iterator(train)
    val   = MultiTask_iterator(val)

    kv = mx.kvstore.create(args.kv_store)

    prefix = model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # flatten0: for resnext-50-symbol.json
    new_sym = get_fine_tune_model(sym, args.num_classes_mt1, args.num_classes_mt2, 'flatten0')

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
        context     = devs,
        symbol      = new_sym,
        data_names  = ['data'],
        label_names = ['softmax_multitask1_label', 'softmax_multitask2_label']
    )

    checkpoint = mx.callback.do_checkpoint(args.save_result)

    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(MultiTask_Accuracy(num = 2, output_names = ['softmax_multitask1_output','softmax_multitask2_output']))

    model.fit(
        train_data         = train,
        begin_epoch        = epoch,
        num_epoch          = num_epoch,
        eval_data          = val,
        eval_metric        = eval_metric,
        validation_metric  = eval_metric,
        kvstore            = kv,
        optimizer          = 'sgd',
        optimizer_params   = optimizer_params,
        arg_params         = arg_params,
        aux_params         = aux_params,
        initializer        = initializer,
        allow_missing      = True,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 20),
        epoch_end_callback = checkpoint
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'train a model on a dataset')
    parser.add_argument('--model',       type = str,   default = '/root/mxnet_dpn/models/models_org/resnext-50', required = True)
    parser.add_argument('--gpus',        type = str,   default = '1')
    parser.add_argument('--batch-size',  type = int,   default = 32)
    parser.add_argument('--epoch',       type = int,   default = 0)
    parser.add_argument('--image-shape', type = str,   default = '3,224,224')
    parser.add_argument('--data-train',  type = str,   default = '/root/mxnet_dpn/mxnet/tools/mnist224_train.lst')
    parser.add_argument('--image-train', type = str,   default = '/root/mxnet_datasets/')
    parser.add_argument('--data-val',    type = str,   default = '/root/mxnet_dpn/mxnet/tools/mnist224_test.lst')
    parser.add_argument('--image-val',   type = str,   default = '/root/mxnet_datasets/')
    parser.add_argument('--num-classes-mt1', type = int,   default = 10)
    parser.add_argument('--num-classes-mt2', type = int,   default = 2)
    parser.add_argument('--lr',          type = float, default = 0.01)
    parser.add_argument('--num-epoch',   type = int,   default = 30)
    parser.add_argument('--kv-store',    type = str,   default = 'device', help = 'the kvstore type')
    parser.add_argument('--save-result', type = str,   default = '/root/mxnet_dpn/models/mnist224_resnext50_SLMT/resnext50',
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