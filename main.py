from LwF import LwFmodel
from ResNet import resnet18_cbam
parser=2
numclass=int(20/parser)
task_size=int(20/parser)
feature_extractor=resnet18_cbam()
img_size=32
batch_size=200
memory_size=2000
epochs=1
learning_rate=0.01

import torch
from torchvision import datasets, transforms
import myDatasets
normal_transforms = {
    'train': transforms.Compose([
        transforms.Resize(35),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize(35),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}
svhn = {x: datasets.SVHN(root='SVHN', split=x if x == 'train' else 'test',
                                        download=True, transform=normal_transforms[x])
         for x in ['train', 'val']}
cifar = {x: myDatasets.CIFAR10(root='CIFAR10', extend =10, train=True if x == 'train' else False,
                                        download=True, transform=normal_transforms[x])
         for x in ['train', 'val']}
svhn_loader = {x: torch.utils.data.DataLoader(svhn[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
jointdata  = {x: myDatasets.ConcatDataset(svhn[x], cifar[x]) for x in ['train', 'val']}

model=LwFmodel(jointdata, numclass,feature_extractor,batch_size,epochs,learning_rate,task_size)

for i in range(parser):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)