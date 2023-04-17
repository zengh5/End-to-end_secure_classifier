
import torch
from torchvision import datasets, transforms
# from train_utils import train, test_epoch
from train_utils_AddDe import test_epoch_AddDe
import numpy as np
import torch.nn as nn
import torchvision.models as models
#######

device = torch.device("cuda")
## 0 Path of the adversarial/benign images
# Benign images
data = "OriImages"
# # BIM attack
# data = "AdverImages/oracle/BIM"
# # MI attack
# data = "AdverImages/oracle/MI"
# # CW attack
# data = "AdverImages/oracle/CW"

batch_size = 5

## 1 Models
# modelC : classification model
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print("=> creating model '{}'".format('resnet18'))
modelC = models.__dict__['resnet18'](num_classes=20)
print(modelC)

# Oracle classifier, standard training
saved = torch.load('trained_models/best_res18_c20_oracle.pth')

modelC.load_state_dict(saved)
modelC = nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    modelC
)
modelC.eval()
modelC = modelC.to(device)


## 2 Data
# Data transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
])

test_set = datasets.ImageFolder(data, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

## 3 Detect
# a probe image is declared to be adversarial when the prediction of itself and the prediction of its
# AddDe-processed version is different

# As the increase of sigmas, both the FPR and TPR increase.
sigmas = [4.0, 10.0, 20.0]
inconsistent = np.zeros_like(sigmas)
for i in range(len(sigmas)):
    test_results = test_epoch_AddDe(
        model=modelC,
        loader=test_loader,
        is_test=True,
        noise_sigma=sigmas[i] / 255.
    )
    _, inconsistent[i] = test_results

done = 1
