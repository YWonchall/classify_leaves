import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from scipy import stats
from tqdm import tqdm
import utils
import config
from dataset import LeavesDataset, num_to_class


def predict(net, test_loader, device, num_gpus=1): 
    net, devices = utils.to_devices(net, device,num_gpus)
    net.eval()
    preds = []
    with torch.no_grad():
        for X in tqdm(test_loader):
            X = X.to(devices[0])
            outputs = net(X)
            preds.extend(outputs.argmax(dim=-1).cpu().numpy().tolist())
    return preds


if __name__ == '__main__':
    df_test = pd.read_csv(config.test_path)
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    test_dataset = LeavesDataset(
        df_test, config.imgs_path, mode='test',transform=val_test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )

    net = models.resnet18(num_classes=config.num_classes)
    allmodels_preds = []
    for k in range(config.num_folds):
        bestmodels_path = config.bestmodels_folder + f'resnet18_best_fold_{k}.pth'
        net.load_state_dict(torch.load(bestmodels_path))
        singlemodels_preds = predict(net, test_loader, config.device)    
        allmodels_preds.append(singlemodels_preds)

    predictions = stats.mode(np.array(allmodels_preds))[0][0]
    preds = []
    for i in predictions:
        preds.append(num_to_class[i])
    savefile_path = './submission1.csv'
    df_test['label'] = pd.Series(preds)
    df_test.to_csv(savefile_path, index=False)
