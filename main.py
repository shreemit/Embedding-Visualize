from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from deep_features import DeepFeatures
import torch

def stack(tensor, times=3):
  return(torch.cat([tensor]*times, dim=0))

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

BATCH_SIZE = 100

tfs = transforms.Compose([transforms.Resize((221,221)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485], std=[0.229]),
                          stack])

DATA_FOLDER = r'./Data/MNIST'
IMGS_FOLDER = '/Users/shreemit/Developer/cv_visualize/Embedding-Visualize/Outputs/Images'
EMBS_FOLDER = '/Users/shreemit/Developer/cv_visualize/Embedding-Visualize/Outputs/Embeddings'
TB_FOLDER = '/Users/shreemit/Developer/cv_visualize/Embedding-Visualize/Outputs/Tensorboard'
DEVICE = 'cpu'
# Load the Fashion MNIST dataset
# trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.FashionMNIST(DATA_FOLDER, download=True, train=False, transform=tfs)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# model = efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)
# model = model.to('mps')
# print(model)
# model.classifier = Identity()
# model.eval()

resnet152 = models.resnet152(pretrained=True).to(DEVICE)

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

resnet152.fc = Identity() # Remove the prediction head
resnet152.eval() # Setup for inferencing


DF = DeepFeatures(model = resnet152 , 
                  imgs_folder = IMGS_FOLDER, 
                  embs_folder = EMBS_FOLDER, 
                  tensorboard_folder = TB_FOLDER, 
                  experiment_name='EXPERIMENT1')

batch_imgs, batch_labels = next(iter(testloader))

DF.write_embeddings(x = batch_imgs.to(DEVICE))

DF.create_tensorboard_log()
