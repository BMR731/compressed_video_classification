from torchvision import transforms
import numpy as np

WID = 112
HEI = 112

transform_rgb_residual = transforms.Compose([
    # transforms.ColorJitter(1.0, 1.0, 1.0, 0.25),
    # transforms.RandomRotation(5),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomResizedCrop(size=(WID, HEI), scale=(0.75, 1.0)),
    transforms.Resize(size=(WID,HEI)),
    transforms.ToTensor(), # you must know now the value has been scaled (0,1)
])

transform_mv = transforms.Compose([
    # transforms.Resize(np.random.randint(WID, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(size=(WID, HEI), scale=(0.75, 1.0)),
    transforms.ToTensor(),
])

transform_infer = transforms.Compose([
    # transforms.Resize(size=(int(256*0.8),int(340*0.8))),
    transforms.Resize(size=(WID, HEI)),
    # transforms.CenterCrop(size=(WID, HEI)),
    transforms.ToTensor(),
])

