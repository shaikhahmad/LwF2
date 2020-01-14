from LwF import LwFmodel
from ResNet import resnet18_cbam
parser=10
numclass= 10 #int(40/parser)
task_size=int(40/parser)
feature_extractor=resnet18_cbam()
img_size=32
batch_size=200
task_size=int(40/parser)
memory_size=2000
epochs=2
learning_rate=2.0

model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size)

import torch
from torchvision import datasets, transforms
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
# svhn_loader = {x: torch.utils.data.DataLoader(svhn[x], batch_size=batch_size,
#                                                shuffle=True, num_workers=4)
#                 for x in ['train', 'val']}

for i in range(4):
    model.beforeTrain(svhn)
    accuracy=model.train()
    model.afterTrain(accuracy)