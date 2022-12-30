import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
import utils
import config
from dataset import LeavesDataset


def kfold(df_data, k=5, shuffle=False):
    KF = StratifiedKFold(n_splits=k)
    for fold, (train_index,valid_index) in enumerate(KF.split(df_data['image'], df_data['label'])):   
        train_dataset = LeavesDataset(
            df_data.loc[train_index], config.imgs_path, mode='train',transform=train_transform
        )
        valid_dataset = LeavesDataset(
            df_data.loc[valid_index], config.imgs_path, mode='train',transform=val_test_transform
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers
        )
        yield fold, train_loader, valid_loader

def train(net, train_loader, val_loader, num_epochs, lr, wd, 
          device, bestmodels_path, finetune=False, num_gpus=1): 
    net, devices = utils.to_devices(net, device, num_gpus=num_gpus)

    if finetune:
        params_1x = [param for name, param in net.named_parameters()
                        if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.Adam(
            [
                {'params': params_1x},
                {'params': net.fc.parameters(), 'lr': lr * 10}
            ],
            lr=lr, weight_decay=wd
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    
    loss = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(num_epochs):     
        net.train()
        train_metric = utils.Accumulator(3)
        with tqdm(train_loader) as pbar:
            for X, y in pbar:
                optimizer.zero_grad()
                X, y = X.to(devices[0]), y.to(devices[0])
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    train_metric.add(l * X.shape[0], utils.accuracy(y_hat, y), X.shape[0])
                train_l = train_metric[0] / train_metric[2]
                train_acc = train_metric[1] / train_metric[2]      
                pbar.set_description(
                    f"Train : {epoch} Loss: {train_l:.4f} Acc: {train_acc:.4f} "
                )
            
        net.eval()
        val_metric = utils.Accumulator(3)
        with tqdm(val_loader) as pbar, torch.no_grad():
            for X, y in pbar:
                X, y = X.to(devices[0]), y.to(devices[0])
                y_hat = net(X)
                l = loss(y_hat, y)
                val_metric.add(l * X.shape[0], utils.accuracy(y_hat, y), X.shape[0])
                val_l = val_metric[0] / val_metric[2]
                val_acc = val_metric[1] / val_metric[2]
                pbar.set_description(
                    f"Valid : {epoch} Loss: {val_l:.4f} Acc: {val_acc:.4f} "
                )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), bestmodels_path)
            print('saving model with acc {:.3f}'.format(best_acc))

def get_net(num_classes, finetune=False, load_params=False, load_path=None):
    if load_params:
        net = models.resnet18(num_classes=num_classes)
        net.load_state_dict(torch.load(load_path))
    else:
        if finetune:
            net = models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            nn.init.xavier_uniform_(net.fc.weight);
        else:        
            net = models.resnet18(num_classes)
            net.apply(utils.init_weights)
    return net


if __name__ == '__main__':
    df_train = pd.read_csv(config.train_path)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])    
    #net = get_finetune_net(num_classes=176)
    for k, train_loader, valid_loader in kfold(df_train, k=5, shuffle=False):  
        print(f"{k} fold")
        bestmodels_path = config.bestmodels_folder + f'resnet18_best_fold_{k}.pth'
        checkpoints_path = config.checkpoints_folder + f'resnet18_checkpoints_fold_{k}.pth'
        net = get_net(num_classes=config.num_classes, finetune=config.finetune,
                        load_params=config.load_params, load_path=checkpoints_path)
        train(net, valid_loader, valid_loader, config.num_epochs, 
             config.lr, config.wd, config.device, bestmodels_path, finetune=config.finetune)
        torch.save(net.state_dict(), checkpoints_path)
