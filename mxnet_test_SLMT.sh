python mxnet_test_resnext_singlelabelmultitask.py \
--epoch 5 \
--model /root/mxnet_dpn/models/mnist224_resnext50_SLMT/resnext50 \
--batch-size 1 \
--num-classes 10 \
--image-txt      /root/mxnet_dpn/mnist224_test.txt \
--image-folder   /root/mxnet_datasets/ \
--gpus 1