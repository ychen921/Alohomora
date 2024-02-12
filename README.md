# Alohomora
CMSC733 Homework 0
This project report was composed of two parts. In Phase 1, we implement a simplified version of the pb-lite edge detection technique. In Phase 2, we developed multiple image classification neural networks such as simple ConvNet, ResNet, ResNeXt, and DenseNet based on the CIFAR10 dataset.

### Phase 1 - Shake My Boundary
In phase 1, we will implement a simple pb-lite edge detection algorithm. First, generate four filter banks which are oriented Derivative of Gaussian (DoG), Leung-Malik (LM), Gabor, and Half-disc. By using these filters, we can create a texture map of the image. Also, use KMeans clustering to generate brightness and color. Then, implement chi-square distance combined with sobel and canny edges. we will have pb-lite edges. Please find more details in the `report.pdf`. To implement it, you can use the following command.
```bash
python3 Wrapper.py
```

### Phase 2 - Deep Dive on Deep Learning
In Phase 2, we implement various image classification networks based on the CIFAR10 dataset. The following sections will show the experiment details of simple ConvNet, improved ConvNet, ResNet, ResNext, and DenseNet.

#### Usage
To train the model, you can run the following command. The default of the training epoch was set to 5. This will train a simple ConvNet. You can also train different architectures of image classifiers: `ConvNet: 0, ResNet: 1, ResNeXt: 2, and DenseNet: 3`.
```
python3 Train.py --model 0
```
If you desire to set a specific number training epoch, you can run the following command:
```
python3 Train.py --NumEpochs 10 --model 0
```

To test the model, you can run the following command. Please ensure that the setting of `--NumEpochs` should be the same as the training epoch as well as `--model`.
```
python3 Test.py --NumEpochs 10 --model 0
```
