# Alohomora
In this project, we build our custom ConvNet, ResNet, ResNext, and DenseNet from Scratch. We train the model based on CIFAR10 data set and evaluate which model has the best performance. With hand-on experience in building residual blocks, resnext blocks, and dense blocks, we are more familiar and comfortable with implementing any neural network from scratch by PyTorch.

## Dataset
A randomized version of the CIFAR-10 dataset with 50000 training images and 10000 test images can be found [here](https://drive.google.com/file/d/1CfbEif07iUgCDfJRCEuXPoRvqqoEbUI_/view?usp=drive_link). More details about the dataset can be found [here](http://www.cs.toronto.edu/~kriz/cifar.html).

## Usage 

### Training
To train the model, you can run the following command. The default of the training epoch was set to 5. This will train a simple ConvNet. You can also select different architectures of image classifiers by `--Model`: `ConvNet: 0, ResNet: 1, ResNeXt: 2, and DenseNet: 3`.
```
python3 Train.py --Model 0
```
If you desire to set a specific number training epoch, you can run the following command `--NumEpochs`:
```
python3 Train.py --NumEpochs 10 --Model 0
```
If you are not putting the dataset in the default directory, you can use this command `--BasePath`.
```
python Train.py --NumEpochs 10 --BasePath {directory of CIFAR10 dataset}/CIFAR10
```

### Testing
To test the model, you can run the following command. Please ensure that the setting of `--NumEpochs` should be the same as the training epoch as well as `--Model`.
```
python3 Test.py --NumEpochs 10 --model 0
```

If you are not putting the dataset in the default directory, you can use this command `--BasePath`.
```
python3 Test.py --NumEpochs 10 --model 0 --BasePath '{directory of CIFAR10 dataset}/CIFAR10/Test'
```
## Performance

The quantitative results are summarized below.
| Model       | Accuracy    | # of Parameters   |  
| :---        |    :----:       |     :----:     | 
| ConvNet   | 79.78 %   | 1,960,547 | 
| ResNet    | 82.99 %   | 167,834   | 
| ResNext   | 83.26 %   | 3,780,554 | 
| DenseNet  | 84.85 %   | 783,770   | 

## References
1. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
2. Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
3. Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
