"#!/usr/bin/env python"
"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 - PyTorch Code


Author(s):
Yi-Chung Chen (ychen921@umd.edu)
Master of engineering in Robotics,
University of Maryland, College Park
"""


import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torchvision.transforms as T

import argparse
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Misc.MiscUtils import *
from Misc.DataUtils import *
from DataReader import *
from Network.Network import *
from Network.ResNet import *
from Network.DenseNet import *
from Network.ResNeXt import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

test_folder = '../CIFAR10/Test'
test_label_file = './TxtFiles/LabelsTest.txt'

def plot_acc_loss(loss_values, Accuracies):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # Plot loss over epoch
    ax1.plot(loss_values)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_ylim(0, 2.5)
    
    # Plot accuracy over epoch
    ax2.plot(Accuracies)
    ax2.set_xlabel('Epochs', fontsize=13)
    ax2.set_ylabel('Accuracy (%)', fontsize=13)
    ax2.set_ylim(0, 100)
    
    plt.suptitle('Testing loss & acc', fontsize=18)
    plt.savefig('../Save_fig/'+'TestLossAccuracy.png')
    plt.show()
    
def Eval(model, data_loader, device, test_pred_path, test_GT_path, epoch, NumEpochs):
    
    total_samples = 0
    total_correct = 0
    
    Epoch_loss = []
    Epoch_acc = []
    
    pred_list = []
    GT_list = []
    
    inference_time = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(data_loader)):
            
            GT_list += label.tolist() 
            
            image = (image/255.0).to(device)
            label = label.to(device)
            
            
            outputs = model(image.float())
            _, pred = torch.max(outputs, 1)
            
            # Compute loss and its gradient
            loss = F.cross_entropy(outputs, label)
            Epoch_loss.append(loss.item())
            
            pred_list += pred.tolist()
            
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)
            
    end_time = time.time()
    inference_time = (end_time - start_time)
    
    # Compute loss and acc
    AccThisEpoch = 100 * total_correct / total_samples
    LossThisEpoch = sum(Epoch_loss) / len(Epoch_loss)
    
    print('Epoch:{}, Loss:{}, Accuracy:{}%, Inference time:{}\n'.format(epoch, LossThisEpoch, AccThisEpoch, inference_time))
    
    if epoch == int(NumEpochs-1):
        with open(test_pred_path, 'w') as file:
            for p in pred_list:
                file.write(f'{p}\n')

        with open(test_GT_path, 'w') as file:
            for p in GT_list:
                file.write(f'{p}\n')
        
    return AccThisEpoch, LossThisEpoch


def Read_labels(LabelsPath, test_pred_path):
    
    GT_labels = []
    pred_labels = []
    
    with open(LabelsPath, 'r') as file:
        labels = file.readlines()
        GT_labels = [int(x.replace('\n', '')) for x in labels]
    
    with open(test_pred_path, 'r') as file:
        labels = file.readlines()
        pred_labels = [int(x.replace('\n', '')) for x in labels]
        
    return GT_labels, pred_labels

def ConfusionMatrix(GT_labels, Pred_labels):
    cm = confusion_matrix(GT_labels, Pred_labels)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_disp.plot()
    plt.savefig('../Save_fig'+'/TestConfusionMatrix.png')
    plt.show()
    

def main():
    
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:10')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/Checkpoints', help='Path to load latest model from, Default:C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/Checkpoints')
    Parser.add_argument('--BasePath', dest='BasePath', default='C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/CIFAR10/Test', help='Path to load images from, Default:C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/CIFAR10/Test')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--Model', type=int, default=0, help='Choose a model, 0: ConvNet, 1: ResNet, 2: ResNeXt, 3: DenseNet, Default:0')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    Model_select = Args.Model
    
    # Create a file for storing accuracy and loss per epoch
    if not os.path.exists('../Save_fig'):
        os.mkdir('../Save_fig')
    
    # Path to save predicted labels
    test_pred_path = './TxtFiles/TestPredOut.txt' # Path to save predicted labels
    test_GT_path = './TxtFiles/TestGTOut.txt'
    
    # Data Loader
    transform = T.Compose([
                T.ToTensor(),
                T.Resize((32, 32)),
                #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    test_dataset = CIFAR10Dataset(BasePath, LabelsPath, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # Initialize model and optimizer
    if Model_select == 0:
        model = CIFAR10Model().to(device)
        print('Model select: ConvNet')
    elif Model_select == 1:
        model = ResNet18().to(device)
        print('Model select: ResNet')
    elif Model_select == 2:
        model = ResNeXt().to(device)
        print('Model select: ResNeXt')
    elif Model_select == 3:
        model = Dense_Net().to(device)
        print('Model select: DenseNet')
    else:
        raise Exception("Oops! The model does not exist... (Only 0 to 3)")
    
    loss_values = []
    Accuracies = []
    
    
    for epoch in range(NumEpochs):
        saved_model = ModelPath + '/' + str(epoch) + 'model.pt'
        print('\nLoading checkpoint: \n', saved_model)
    
        # Load checkpoint
        checkpoint = torch.load(saved_model)
        model.load_state_dict(checkpoint['model_state_dict'])
            
        # Evaluation mode
        model.eval()
        
        # Test model  
        Acc_Epoch, Loss_Epoch = Eval(model, test_loader, device, test_pred_path, test_GT_path, epoch, NumEpochs)
        
        # Store acc and loss over epoch
        loss_values.append(Loss_Epoch)
        Accuracies.append(Acc_Epoch)
    
    # Plot testing loss and accuracy
    plot_acc_loss(loss_values, Accuracies)
    
    GT_Labels, Pred_Labels = Read_labels(test_GT_path, test_pred_path)
    ConfusionMatrix(GT_Labels, Pred_Labels)
    
        
if __name__ == '__main__':
    main()