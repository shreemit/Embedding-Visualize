from torchvision import transforms
from torchvision import datasets
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch

def stack(tensor, times=3):
  return(torch.cat([tensor]*times, dim=0))

BATCH_SIZE = 100

# Define the data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Fashion MNIST dataset
# trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)

print(model)
model.classifier = torch.nn.Identity()
model.eval()

# batch_imgs, batch_labels = next(iter(testloader))
# embs = model(batch_imgs)

# print("Input images: " + str(batch_imgs.shape))
# print("Embeddings: " + str(embs.shape))


