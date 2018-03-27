python mxnet_test_resnext_multilabelsingletask.py \
--epoch 3 \
--model /root/mxnet_dpn/models/mnist224_resnext50_MLST/resnext50 \
--batch-size 1 \
--num-classes 10 \
--image-txt    /root/mxnet_dpn/mnist224_test_MLST.txt \
--image-folder   /root/mxnet_datasets/ \
--gpus 1