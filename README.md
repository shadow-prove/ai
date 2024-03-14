# Introduction

Our project is a model trained on CIFAR-10 dataset
(https://www.cs.toronto.edu/~kriz/cifar.html), which classifies the image into one of the
10 classes. The model is deployed as a telegram bot, so anybody could use it at any
time. The model will predict the class of the object in a given image.
The CIFAR-10 dataset is a widely used benchmark dataset in the field of
computer vision and machine learning. It was first released in 2009 by the Canadian
Institute for Advanced Research (CIFAR). Most of the newly released models and
papers achieve very high accuracy rates on this dataset if we look at paperswithcode
website. One of the new models that got a very high accuracy rate is the vision
transformer model which was pretrained on a very large amount of data
(https://arxiv.org/abs/2010.11929). The model had an accuracy rate of 99.42%. While
the accuracy rate is very high, the model itself is very complex and requires a lot of
pretraining. As we don’t have access to such huge datasets and powerful computer
servers, we trained a Convolutional Neural Network model.

Data and methods

The CIFAR-10 dataset is composed of 60,000 32x32 color images distributed
across 10 classes, with each class containing 6,000 images. Notably, the images are of
low resolution. To facilitate effective model training and evaluation, we partitioned the
dataset into three subsets: training, validation, and test data, with an 80-10-10 split ratio.
This resulted in 48,000 images allocated for training and 6,000 images each for
validation and testing purposes.
The 10 classes in CIFAR-10 are:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10.Truck

<img width="521" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/b5b185d8-3a17-48d3-a9ed-2b736cacffb7">

Convolutional Neural Network (CNN) model is a type of deep learning
architecture specifically designed for image classification tasks. It is easier to train than
vision transformer models. Our CNN model consists of convolutional layers, activation
functions, pooling layers, and fully connected layers. The total number of parameters is
122 570 which is small compared to the state of the art models. The advantage of the
small model is that it trains a lot faster and doesn’t require high computational GPU/TPU
servers.

<img width="510" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/6860ee14-f312-424a-a38b-e8c91cad641e">

The model is trained using backpropagation and gradient descent-based optimization
algorithms, using cross-entropy as loss function for 10 epochs. Besides training our own
CNN model, we also fine-tuned a Resnet50 model. ResNet50 is a deep convolutional
neural network architecture that was introduced by Microsoft Research in 2015. It is part
of the ResNet (Residual Network) family of models, which are known for their deep
architectures and the use of residual connections. This model was pre trained on a
large dataset, called Imagenet, which consists of 1 million images. So, we took the
weights of the convolutional layers of the model, and added new fully connected layers
for 10 classification problems and trained the fully connected layers on our dataset.

Results

Our custom CNN model obtained better results than the fine-tuned Resnet50 model.
We think it is due to the fact that Resnet50 has a lot more parameters and the dataset
of 60k images is too small to change all the parameters, so the model could not get the
optimal weights. Our custom CNN model got test accuracy of 71.06%, while the
Resnet50 model got only 64.28%. Below are the graphs, representing the model’s
accuracy rates on train and validation data over the period of 10 epochs. We see that
validation accuracy reached plato, so further epochs wouldn’t change the accuracy rate


<img width="528" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/304e26f8-4e53-4c7f-8f7e-1514358975bf">

<img width="468" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/407dcf14-f31c-44bc-be9c-dd41fb2c9e25">

Below, we can see the confusion matrix and classification report for all 10 classes. It
can be seen that our CNN model classifies airplanes, automobiles and horses very well.
While it encounters some difficulties with ship and deer classes. The model mistakes
the cat and dogs for horses, also trucks as automobiles and ships. It is understandable
that the model misclassifies those classes, as they look very similar.

<img width="497" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/8829b8a0-a6fe-4905-a50a-2c22438cdfc3">

<img width="513" alt="image" src="https://github.com/daniil228367/Python_FX_project/assets/146715901/a9a7e161-041f-4f18-bb16-9522b1607fff">

Discussion

Overall, we trained a decent Convolutional Neural Network model on the
CIFAR-10 dataset, and obtained an accuracy rate of 71%. Our custom CNN model
appeared to be better than fine-tuning pre-trained CNN models as Resnet50. The
possible reason for that is the fact that the CIFAR-10 dataset consists of relatively
low-resolution images (32x32 pixels) with simple objects, which might not require the
complexity of a deep architecture like ResNet50. A smaller custom CNN might be more
suitable for capturing the essential features of such images, especially when the dataset
size is limited. Also, fine-tuning a large, pre-trained model like ResNet50 on a relatively
small dataset such as CIFAR-10 can lead to overfitting. The high capacity of ResNet50
may cause it to memorize the training data, resulting in poor generalization to unseen
examples in the validation and test sets. On the other hand, a smaller custom CNN with
fewer parameters might be less prone to overfitting and generalize better to unseen
data. This can be seen from the training graphs of our custom CNN model and
Resnet50 model. While our model has relatively same training and validation accuracies
during 10 epochs, the Resnet50 model has some overfitting tendency.





