# mxnet based multilabel multitask project


tags： mxnet multilabel multitask

---

### 0. background

this project contains 4 sub-projects， and they are all based mxnet：

1. single label single task (SLST for short);
2. single label multi task (SLMT for short);
3. multi label single task (MLST for short);
4. multi label multi task (MLMT for short);
 
and all codes are all written in python, actually, I test the following code in the DPNs based mxnet version, but it is easy to transfer the follow code to official mxnet.

the code use mxnet.image.ImageIter to handle the images and .lst data reader, but not the mxnet.io.ImageRecordIter to handle the .rec and .lst data reader.

And i use the resnext50 for finetuning.


### 1. SLST

The relevant code files are mxnet_train_SLST.sh, mxnet_test_SLST.sh, mxnet_train_resnext_singlelabelsingletask.py, mxnet_test_resnext_singlelabelsingletask.py.

and I use the mxnet_train_SLST.sh to call mxnet_train_resnext_singlelabelsingletask.py by using themnist224_train.lst, mnist224_test.lst in datalisttxt folder.

Once the training finished, the test procedure is operated by calling mxnet_test_SLST.sh, mxnet_test_resnext_singlelabelsingletask.py, and i use mnist224_test.txt in datalisttxt folder.

the .lst file organized like the following:
```
685551	12.000000	pavilion/00006962.jpg
1309299	24.000000	wood_house/00016810.jpg
704968	13.000000	plane/00005464.jpg
992439	18.000000	swimming_pool/00003219.jpg
3537	0.000000	aquarium_underwater/00004537.jpg
1004156	18.000000	swimming_pool/00002370.jpg
1262962	23.000000	window/00003901.jpg
1108990	20.000000	tower/00005627.jpg
365688	6.000000	crops_field/00003949.jpg
```

each line denotes a train/val image example
```
the 1st column is the image index
the 2nd column is the image class
the 3rd column is the relative image path
```

running the code is straighrforward, simply exec mxnet_train_SLST.sh/mxnet_test_SLST.sh, and i use the resized 224*224 mnist .jpg images to do train/test.

### 2. SLMT

you can refer to the SLST filename to use the SLMT, so i do not mention again.

the .lst file organized like the following:
```
1309299	24.000000	wood_house/00016810.jpg
704968	13.000000	plane/00005464.jpg
992439	18.000000	swimming_pool/00003219.jpg
3537	0.000000	aquarium_underwater/00004537.jpg
1004156	18.000000	swimming_pool/00002370.jpg
1262962	23.000000	window/00003901.jpg
1108990	20.000000	tower/00005627.jpg
365688	6.000000	crops_field/00003949.jpg
```

each line denotes a train/val image example
```
the 1st column is the image index
the 2nd column is the image class
the 3rd column is the relative image path
```

so I use the single label to do the multi task classifitation the 1st task is class each image like in mxnet_train_resnext_singlelabelsingletask.py, the 2nd task is determine whether the output label is larger than 10, if >10, output 1, else 0.

actually, the SLMT is a bit confused, in the following code:

```python
def next(self):
    batch  = self.data_iter.next()
    label1 = batch.label[0]

    # new label based on the original label, output 0 or 1 here
    label2 = mx.nd.array(label1.asnumpy() > MULTITASK_LABEL2_THRES).astype('float32')

    return mx.io.DataBatch(data  = batch.data,
                           label = [label1, label2],
                           pad   = batch.pad,
                           index = batch.index)
```

we can find that the two labels are derived from the single label in label2 by determining whether the single label is lager than MULTITASK_LABEL2_THRES or not, 

so in real training procedure, it is a multi-task multi-label job.

I clarify the SLMT just for telling how to do multitask mission by using single label.

### 3. MLST

the .lst file organized like the following:
```
685551	1.000000	0.000000	1.000000	0.000000	pavilion/00006962.jpg
1309299	0.000000	1.000000	0.000000	1.000000	wood_house/00016810.jpg
704968	0.000000	0.000000	1.000000	0.000000	plane/00005464.jpg
992439	1.000000	0.000000	1.000000	1.000000	swimming_pool/00003219.jpg
1004156	1.000000	1.000000	0.000000	1.000000	aquarium_underwater/00004537.jpg
1262962	0.000000	1.000000	1.000000	0.000000	swimming_pool/00002370.jpg
1108990	1.000000	0.000000	0.000000	0.000000	window/00003901.jpg
365688	1.000000	0.000000	0.000000	0.000000	crops_field/00003949.jpg
```

each line denotes a train/val image example
```
the 1st column is the image index
the 2nd~5th column is the image class
the 6th column is the relative image path
```

basically, we set the multilabel as multi-hot label, which means each column(eg: above 2nd~5th presents one class,
and the image belongs to the specified class or not, and each class will simplified as a binary classifier.


num-labels in mxnet_train_MLST.sh shows the number of labels

### 4. MLMT

the .lst file organized like the following:
```
685551	1.000000	6.000000	1.000000	0.000000	0.000000	pavilion/00006962.jpg
1309299	10.000000	5.000000	0.000000	1.000000	2.000000	wood_house/00016810.jpg
704968	6.000000	11.000000	1.000000	0.000000	4.000000	plane/00005464.jpg
992439	8.000000	0.000000	4.000000	1.000000	5.000000	swimming_pool/00003219.jpg
1004156	4.000000	2.000000	4.000000	1.000000	6.000000	aquarium_underwater/00004537.jpg
1262962	16.000000	1.000000	3.000000	0.000000	8.000000	swimming_pool/00002370.jpg
1108990	0.000000	14.000000	2.000000	0.000000	5.000000	window/00003901.jpg
365688	25.000000	9.000000	0.000000	0.000000	9.000000	crops_field/00003949.jpg
```
each line denotes a train/val image example
```
the 1st column is the image index
the 2nd~6th column is the image class
the 7th column is the relative image path
```

different from the multi-hot label in mxnet_train_resnext_multilabelsingletask.py, each column is a multi-classifier.

For example, 2nd column has 26 classes, and in this way,
it looks like the singlelabel in mxnet_train_resnext_singlelabelsingletask.py;

3rd column has 15 classes, but is a different task from 2nd and the rest,
which means the classes in each column are independent with each other.

and 5th column is a binary classification task, which is similar with mxnet_train_resnext_multilabelsingletask.py in each column

### 5. references

1. https://github.com/apache/incubator-mxnet/blob/master/example/multi-task/example_multi_task.py
2. https://github.com/hariag/mxnet-multi-task-example/blob/master/multi-task.ipynb
3. http://blog.csdn.net/linmingan/article/details/78360854
4. https://github.com/cypw/DPNs
5. https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/fine-tune.py
6. https://github.com/miraclewkf/multi-task-MXNet
7. https://github.com/miraclewkf/multilabel-MXNet

### 6. revise details

| revise id   |  revise time  |  revise version  |  reviser  | revise comments |
| :-----:  | :-----:    | :----:     | :-----:  | :----:   |
| 1        | 2018-01-24 |   V1.0     |   humengdoudou   |          |
| 2        |            |            |          |          |
| 3        |            |            |          |          |
| 4        |            |            |          |          |
