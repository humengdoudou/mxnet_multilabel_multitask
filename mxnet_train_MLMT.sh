python mxnet_train_resnext_multilabelmultitask.py \
--epoch 0 \
--model /root/mxnet_dpn/models/models_org/resnext-50 \
--batch-size 32 \
--num-classes-mt1 26 \
--num-classes-mt2 11 \
--num-classes-mt3 8 \
--num-classes-mt3 8 \
--num-classes-mt3 4 \
--data-train  /root/mxnet_dpn/mxnet/tools/coat_train.lst \
--image-train /root/mxnet_datasets/ \
--data-val    /root/mxnet_dpn/mxnet/tools/coat_val.lst \
--image-val   /root/mxnet_datasets/ \
--num-labels  5 \
--num-examples 246 \
--lr 0.001 \
--gpus 1 \
--num-epoch 30 \
--save-result /root/mxnet_dpn/models/mnist224_resnext50_MLMT/resnext50
