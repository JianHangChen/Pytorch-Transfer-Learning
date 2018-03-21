from torchvision import models,transforms,datasets
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import sys
import os

# import matplotlib.pyplot as plt



#input:
#python3 train.py --data /tiny/imagenet/dir/ --save /dir/to/save/model/

print('please copy input like:')
print('python3 train.py --data /tiny-imagenet-200/ --save /hw6model/')

if(sys.argv[1] == '--data'):
    data_dir = '.'+ sys.argv[2] #'tiny-imagenet-200'
    print('datadir is:', data_dir)
else:
    print('please copy input like:')
    print('python3 train.py --data /tiny-imagenet-200/ --save /hw6model/')
    sys.exit()

if(sys.argv[3] == '--save'):
    model_dir = '.'+sys.argv[4]
    print('model_dir is:', model_dir)
else:
    print('please input like:')
    print('python3 train.py --data /tiny-imagenet-200/ --save /hw6model/')
    sys.exit()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(model_dir)

# data processing

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

prep1 = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            transforms.Scale(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])




dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), prep1)
         for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=128, #64
                                               shuffle=False, num_workers=6)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()

#  Creating alex Model

model_alex = models.alexnet(pretrained=True)
model_vgg = model_alex



# Modifying last layer and setting the gradient false to all layers

for param in model_vgg.parameters():
    param.requires_grad = False
# model_vgg.classifier._modules['6'].out_features = 200
model_vgg.classifier._modules['6'] = nn.Linear(4096, 200)

# for param in model_vgg.classifier._modules['6'].parameters():
#     param.requires_grad = True
if use_gpu:
	model_vgg = model_vgg.cuda()

print(model_vgg)

# Calculating preconvoluted features
def preconvfeat(dataset):
    conv_features = []
    labels_list = []
    counter = 0
    initial_time = time.time()
    last_time = initial_time
    save_size = 800

    for data in dataset:
        counter += 1

        print('batch:',counter)

        print('current time:', time.time()-initial_time)
        print('1 batch time:', time.time()-last_time)
        last_time = time.time()
        inputs,labels = data

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs , labels = Variable(inputs),Variable(labels)



        x = model_vgg.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())


    conv_features = np.concatenate([[feat] for feat in conv_features])
    # torch.save(conv_features, 'conv_feat_train'+ str(save_file_number) + '.pt')
    # torch.save(labels_list, 'labels_train' + str(save_file_number) + '.pt')

    return (conv_features,labels_list)


preconvoluted = 0
train = 1
test = 0 #always0



print('finish loading raw data')
print('begin training')
if(not preconvoluted):
# time
    if train:
        print('begin preconvoluted training feature')
        start = time.time()
        conv_feat_train,labels_train = preconvfeat(dset_loaders['train'])
        print(time.time() - start)

# time
    if test:
        print('begin preconvoluted test feature')
        start = time.time()
        conv_feat_val,labels_val = preconvfeat_test(dset_loaders['val'])
        print(time.time() - start)

        torch.save(conv_feat_val, 'conv_feat_val.pt')
        torch.save(labels_val, 'labels_val.pt')



# Training fully connected module

# Creating loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.SGD(model_vgg.classifier[6].parameters(),lr = lr,momentum = 0.9)

# Creating Data generator
def data_gen(conv_feat,labels,batch_size=128,shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0,len(conv_feat),batch_size):
        yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])

def train_model(model,size,conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True):
    for epoch in range(epochs):
        batches = data_gen(conv_feat=conv_feat,labels=labels,shuffle=shuffle)
        total = 0
        running_loss = 0.0
        running_corrects = 0
        for inputs,classes in batches:
            inputs = torch.from_numpy(inputs)
            classes = torch.from_numpy(classes)
            if use_gpu:
                inputs = inputs.cuda()
                classes = classes.cuda()
            inputs , classes = Variable(inputs),Variable(classes)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs,classes)
            if train:
                if optimizer is None:
                    raise ValueError('Pass optimizer for train mode')
                optimizer = optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('I am testing')
            _,preds = torch.max(outputs.data,1)
            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc))



# model_vgg.load_state_dict(torch.load('model_cpu.pth'))
# print('previous training successful load cpu model')

if(use_gpu):
    model_vgg.cuda()


print('begin training last layer')



if train:
    # print('loading training features')
    # conv_feat_train = torch.load('conv_feat_train' + str(0) + '.pt')
    # labels_train = torch.load('labels_train' + str(0) + '.pt')
    epochs = 20
    for epoch in range(epochs):
        print()
        print('this is epoch:',epoch+1)
        print()

        start = time.time()
        (train_model(model=model_vgg.classifier,size=dset_sizes['train'],conv_feat=conv_feat_train,labels=labels_train,
            epochs=1,optimizer=optimizer,train=True,shuffle=True))
        print(time.time()-start)


        model_vgg.cpu()
        torch.save(model_vgg.state_dict(),model_dir+'model_cpu.pth')
        print('successful save cpu model')
        if(use_gpu):
            model_vgg.cuda()





# Loading Preconvoluted features


# print(model_vgg)

# if test:
#     print('loading testing features')
#     conv_feat_val = torch.load('conv_feat_val.pt')
#     labels_val = torch.load('labels_val.pt')
#     print(labels_val)
#     print(conv_feat_val.shape)


# print('inference for last layer')
# start = time.time()
# train_model(conv_feat=conv_feat_val,labels=labels_val,model=model_vgg.classifier,size=dset_sizes['val'],train=False,shuffle=True)
# print(time.time()-start)
# print('everything is fine')