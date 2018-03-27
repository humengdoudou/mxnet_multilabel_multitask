python mxnet_test_resnext_multilabelmultitask.py \
--epoch 30 \
--model /root/mxnet_dpn/models/mnist224_resnext50_MLMT/resnext50 \
--batch-size 1 \
--num-classes 10 \
--image-txt      /root/mxnet_dpn/coat_MLMT_test.txt \
--image-folder   /root/mxnet_datasets/ \
--gpus 1