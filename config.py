num_classes = 176   # 类别数目
num_folds = 5       # kfold
batch_size = 128
lr = 3e-4
lr_scale = 20       # 衰减epoch
wd = 5e-3
num_epochs = 1
num_workers = 4     # 数据读取进程数
finetune = True     # 是否采用预训练模型
load_params = True  # 是否加载本地参数
device = 'gpu'      # 采用 'gpu' or 'cpu' 训练
num_gpus = 1     # gpu数目
checkpoints_folder = './check_points/'  
bestmodels_folder = './best_models/'

# 数据集相关路径
train_path = './data/train.csv'
valid_path = ''
test_path = './data/test.csv'
imgs_path = './data/'
