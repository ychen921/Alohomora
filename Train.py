"#!/usr/bin/env python"

"""
CMSC733 Spring 2024: Classical and Deep Learning Approaches for
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
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T

import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
import time

from Misc.MiscUtils import *
from Misc.DataUtils import *
from DataReader import *
from Network.Network import *
from Network.ResNet import *
from Network.DenseNet import *
from Network.ResNeXt import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

dtype = torch.float32

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def test_training_data(model, device, train_dataloader, LabelsPathPred, LabelsPathGT):
    
    total_samples = 0
    total_correct = 0
    
    pred_list = []
    GT_list = []
    
    inference_time = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(train_dataloader)):
            image = (image/255.0).to(device)
            label = label.to(device)
            
            outputs = model(image.float())
            _, pred = torch.max(outputs, 1)
            
            pred_list += pred.tolist()
            GT_list += label.tolist()
            
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)
            
    end_time = time.time()
    inference_time = (end_time - start_time)
            
            
    print('Inference time:{} sec\n'.format(inference_time))
            
    with open(LabelsPathPred, 'w') as file:
        for p in pred_list:
            file.write(f'{p}\n')
    
    with open(LabelsPathGT, 'w') as file:
        for p in GT_list:
            file.write(f'{p}\n')
    

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile) 


def plot_acc_loss(loss_values, Accuracies):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # Plot loss over epoch
    ax1.plot(loss_values)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_ylim(-1, 3)
    
    # Plot accuracy over epoch
    ax2.plot(Accuracies)
    ax2.set_xlabel('Epochs', fontsize=13)
    ax2.set_ylabel('Accuracy (%)', fontsize=13)
    ax2.set_ylim(0, 100)
    
    plt.suptitle('Training loss & acc', fontsize=18)
    plt.savefig('../Save_fig/'+'TrainLossAccuracy.png')
    plt.show()


def Solver(model, optimizer, NumEpochs, train_loader, device, SaveCheckPoint, 
           CheckPointPath, LatestFile, LogsPath, LabelsPathPred, LabelsPathGT):
    
    # Check Lastest file exsiting or not
    if LatestFile is not None:
        # Load necessary info
        checkpoint = torch.load(LatestFile)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        StartEpoch = int(checkpoint['epoch'])+1
        # loss = checkpoint['loss']
        
        # Training mode
        model.train() 
        
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    
    loss_values = []
    Accuracies = []
    
    # Tensorboard
    writer = SummaryWriter(LogsPath)
    
    for epoch in range(StartEpoch, NumEpochs):
        
        AccThisEpoch = 0
        Epoch_loss = []

        num_correct = 0
        num_sample = 0

        for i, (image, label) in enumerate(tqdm(train_loader)):
            image = (image/255.0).to(device)
            label = label.to(device)
            
            # Write the network graph at epoch 0, batch 0
            if epoch == 0 and i == 0:
                writer.add_graph(model, input_to_model=image, verbose=False)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions ofr this batch
            outputs = model(image)
            _, pred = torch.max(outputs, 1)
            
            # Write prediction to txt file
            pred_list = pred.tolist()

            # Accumulate correct predictions
            num_correct += (pred == label).sum().item()
            num_sample += label.size(0)

            # Compute loss and its gradient
            loss = F.cross_entropy(outputs, label)
            Epoch_loss.append(loss.item())

            # Backpropation
            loss.backward()

            # Adjust learning rate
            optimizer.step()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if i % SaveCheckPoint == 0:
                
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(epoch) + 'a' + str(i) + 'model.pt'
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                    }, SaveName)
                # torch.save(model.state_dict(), SaveName)
                print('\n' + SaveName + ' Model Saved...')
            
            # Write training accuracy per batch and the loss value to Tensorboard
            writer.add_scalar('Loss/Train', num_correct/num_sample, i)
            writer.add_scalar('Accuracy/Train', loss.item(), i)

        # Compute loss and acc
        AccThisEpoch = 100 * num_correct / num_sample
        LossThisEpoch = sum(Epoch_loss) / len(Epoch_loss)
        
        # Store acc and loss over epoch
        loss_values.append(LossThisEpoch)
        Accuracies.append(AccThisEpoch)
        
        SaveName = CheckPointPath + str(epoch) + 'model.pt'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, SaveName)
        print(SaveName + ' Model Saved...')
        print('Epoch:{} Loss:{} Accuracy:{}%\n'.format(epoch, LossThisEpoch, AccThisEpoch))
        
    writer.close()
    
    # Plot accuracy and loss over epochs
    plot_acc_loss(loss_values, Accuracies)
    
    # Test training data by latest model
    test_training_data(model, device, train_loader, LabelsPathPred, LabelsPathGT)
    
    
def Read_labels(LabelsPath, test_pred_path):
    
    GT_labels = []
    pred_labels = []
    
    with open(LabelsPath, 'r') as file:
        labels = file.readlines()
        GT_labels = [x.replace('\n', '') for x in labels]
    
   
    with open(test_pred_path, 'r') as file:
        labels = file.readlines()
        pred_labels = [x.replace('\n', '') for x in labels]
        
    return GT_labels, pred_labels

def ConfusionMatrix(GT_labels, Pred_labels):
    cm = confusion_matrix(GT_labels, Pred_labels)
    
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_disp.plot()
    plt.savefig('../Save_fig'+'/TrainConfusionMatrix.png')
    plt.show()
            

def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/CIFAR10', help='Base path of images, Default:C:/Users/steve/Desktop/733/ychen921_hw0/Phase2/CIFAR10')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:10')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Not using in this case(Factor to reduce Train data by per epoch), Default:None')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:64')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--Model', type=int, default=0, help='Choose a model (0: ConvNet, 1: ResNet, 2: ResNeXt, 3: DenseNet), Default:0')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    Model_select = Args.Model
    
    train_folder = BasePath + '/Train'
    train_label_file = './TxtFiles/LabelsTrain.txt'
    
    # Path to save predicted labels
    LabelsPathPred = './TxtFiles/TrainPredOut.txt'
    LabelsPathGT = './TxtFiles/TrainGTOut.txt'  

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    print('Latest file: ', LatestFile)  

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Data Loader
    transform = T.Compose([
                T.ToTensor(),
                T.Resize((32, 32)),
                #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomHorizontalFlip(),
            ])
    train_dataset = CIFAR10Dataset(train_folder, train_label_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=MiniBatchSize, shuffle=True)
    
    # Initialize model and optimizer
    if Model_select == 0:
        model = CIFAR10Model(NumClasses).to(device)
        print('Model select: ConvNet')
    elif Model_select == 1:
        model = ResNet18(in_channels=3 , num_classes=NumClasses).to(device)
        print('Model select: ResNet')
    elif Model_select == 2:
        model = ResNeXt().to(device)
        print('Model select: ResNeXt')
    elif Model_select == 3:
        model = Dense_Net().to(device)
        print('Model select: DenseNet')
    else:
        raise Exception("Oops! The model does not exist... (Only 0 to 3)")
    
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Create a file for storing accuracy and loss per epoch
    if not os.path.exists('../Save_fig'):
        os.mkdir('../Save_fig')
    
    summary(model, (3, 32, 32))
    
    # Train the model
    Solver(model, optimizer, NumEpochs, train_loader, device, SaveCheckPoint, 
           CheckPointPath, LatestFile, LogsPath, LabelsPathPred, LabelsPathGT)
    
    GT_Labels, Pred_Labels = Read_labels(LabelsPathGT, LabelsPathPred)
    ConfusionMatrix(GT_Labels, Pred_Labels)


if __name__ == '__main__':
    main()