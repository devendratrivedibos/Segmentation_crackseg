import torchvision.models as models
import torch

# This will download the ImageNet pre-trained weights to your cache automatically
resnet101 = models.resnet101(weights='DEFAULT')

# If you specifically need to save it as 'resnet101-imagenet.pth' locally:
torch.save(resnet101.state_dict(), r'D:\Devendra_Files\segmentation_training\training_code\pretrained_weights\resnet101-imagenet.pth')