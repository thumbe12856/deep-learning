import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import dataset


transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = dataset.CIFAR10('./data', train=True, transform=transform_train)
test_dataset = dataset.CIFAR10('./data', train=False, transform=transform_test)
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dataiter = iter(test_loader)
image, label = dataiter.next()
image = viz.images(image[:10]/2+0.5, nrow=10, padding=3, env='cifar10')
text = viz.text('||'.join('%6s' % classes[label[j]] for j in range(10)))

