import gc
from torch.utils.data import DataLoader
from tools import *
from load_data import *
from pred import *
from train import *
import argparse


#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='seed value')
# Data parameters
parser.add_argument('--concat_nframes', type=int, default=1, help='the number of frames to concat with, n must be odd (total 2k+1 = n frames)')
parser.add_argument('--train_ratio', type=float, default=0.8, help='the ratio of data used for training, the rest will be used for validation')
# Training parameters
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_epoch', type=int, default=5, help='the number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--model_path', type=str, default='./model.ckpt', help='the path where the checkpoint will be saved')
# Model parameters
parser.add_argument('--hidden_layers', type=int, default=1, help='the number of hidden layers')
parser.add_argument('--hidden_dim', type=int, default=256, help='the hidden dimension')
args = parser.parse_args()
# Compute input_dim based on parsed concat_nframes
args.input_dim = 39 * args.concat_nframes


same_seeds(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

#******************************************************************************
# load training and validation data
#******************************************************************************
train_X, train_y = preprocess_data(split='train', feat_dir='C:/Users/shayg/Datasets/libriphone/libriphone/feat', phone_path='C:/Users/shayg/Datasets/libriphone/libriphone', concat_nframes=args.concat_nframes, train_ratio=args.train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='C:/Users/shayg/Datasets/libriphone/libriphone/feat', phone_path='C:/Users/shayg/Datasets/libriphone/libriphone', concat_nframes=args.concat_nframes, train_ratio=args.train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

#******************************************************************************
# Create Dataloader objects
#******************************************************************************
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

#==============================================================================
# Start training
#==============================================================================
train(args.input_dim, args.hidden_layers, args.hidden_dim, args.learning_rate, train_set, train_loader,
      val_set, val_loader, args.num_epoch, args.model_path)
# remove loaders to save memory
del train_loader, val_loader
gc.collect()

#******************************************************************************
# load testing data
#******************************************************************************
test_X = preprocess_data(split='test', feat_dir='C:/Users/shayg/Datasets/libriphone/libriphone/feat', phone_path='C:/Users/shayg/Datasets/libriphone/libriphone', concat_nframes=args.concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

#******************************************************************************
# load model
#******************************************************************************
model = Classifier(input_dim=args.input_dim, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim).to(device)
model.load_state_dict(torch.load(args.model_path))

#******************************************************************************
# make prediction
#******************************************************************************
pred(model, test_loader)
