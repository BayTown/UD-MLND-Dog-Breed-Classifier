import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np

def face_detector(img_path):
    '''
    Using a CascadeClassifier from cv2 to check whether or not a face of a person is present in an image.

    Args:
        img_path: path to an image

    Returns:
        Return true, if a face of a person is present on the image. Else return false
    '''
    # Load a cv2 CascadeClassifier
    face_cascade_ext = cv2.CascadeClassifier('model/haarcascade_frontalface_alt2.xml')
    # Load the image from path
    img = cv2.imread(img_path)
    # Convert to gray
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # detect if face is present in the image
    faces = face_cascade_ext.detectMultiScale(gray)
    
    # returns true if length of faces is greater than zero
    return len(faces) > 0

def Get_VGG16_model():
    '''
    Loads pretrained VGG16 model and returns it

    Args:
        None
        
    Returns:
        Pretrained VGG16 model
    '''
    # Define VGG16 model
    VGG16 = models.vgg16(pretrained=True)
    # Set it to eval-mode
    VGG16.eval()

    return VGG16


def VGG16_predict(img_path, VGG16):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image

    # Set device to CPU
    device = torch.device("cpu")

    # Here we define the transformation
    # This is a combination of all image transformations
    # that are to be performed on the input image
    transform = transforms.Compose([transforms.Resize((224, 224)),                   # Resize the image to 224x224 pixels
                                    transforms.ToTensor(),                           # Convert the image to the PyTorch Tensor data type
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalizing the image with specific mean and standard deviation
                                                         std=[0.229, 0.224, 0.225])])
    
    # Load image
    img = Image.open(img_path)
    # Transform image
    img = transform(img)
    # Change Tensor to a one-dimensional one
    img = img.unsqueeze(0)
    # Convert img to a Variable. A PyTorch Variable is a wrapper around a PyTorch Tensor.
    img = Variable(img)
    # Transform img to CPU-Datatype
    img = img.to(device)
    # Get the prediction of the model
    prediction = VGG16(img)
    
    # Get the max-Value of the Tensor-matrix and return as integer
    _, index = torch.max(prediction, 1)
    
    return index.item() # predicted class index


def dog_detector(img_path, VGG16):
    '''
    This function uses a pretrained VGG16-model to make a prediction if a dog in an image is present or not
    
    Args:
        img_path: path to an image

    Returns:
        Returns "True" if a dog is detected in the image stored at img_path
    '''
    
    # First index of dogs in dictionary
    first_idx_dict = 151
    # Last index of dogs in dictionary
    last_idx_dict = 268
    
    # Get Integer value of the prediction
    out = VGG16_predict(img_path, VGG16)
    
    # True if returned index of out is in the range of the dogs-labels in the dictionary. If not allocation is false.
    dog_present = True if out >= first_idx_dict and out <= last_idx_dict else False
    
    return dog_present # true/false


def get_model():
    '''
    This function load a pretrained model, add new classifier layers, load the weights to the model and return this model
        
    Args:
        None

    Returns:
        Returns the pretrained model with new classifier layers and weights
    '''
    # Load the pretrained ResNeXt101-Model from pytorch
    model_transfer = models.resnext101_32x8d(pretrained=True)
    # Add a Dropout layer
    model_transfer.add_module('drop', nn.Dropout(0.3))
    # Add a fully-connected layer - This will be the last layer
    model_transfer.add_module('fc1', nn.Linear(in_features=1000, out_features=134, bias=True))

    # Replacing the last 3 layers for fine tuning
    # Parameters of newly constructed modules have requires_grad=True by default
    model_transfer.fc = nn.Linear(2048, 1000, bias=True)
    model_transfer.drop = nn.Dropout(0.3)
    model_transfer.fc1 = nn.Linear(in_features=1000, out_features=134, bias=True)

    # Set device to CPU
    device = torch.device('cpu')
    # load the trained weights
    model_transfer.load_state_dict(torch.load('model/model_transfer_resnext101.pt', map_location=device))
    # Put model to evaluation mode
    model_transfer.eval()

    return model_transfer


def predict_breed_transfer(img_path, model_transfer):
    '''
    A function that takes a path to an image as input
    and returns top-k predictions of the dog breed that is predicted by the model.
        
    Args:
        img_path:          path to an image
        model_transfer:    model for prediction

    Returns:
        Returns top-k predictions and top-k labels in an array
    '''
    # Load class names
    class_names_path = 'data/class_names.txt'
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()

    # Load model
    model = model_transfer

    # Set topk
    topk = 3

    # Define image transformation
    # Define standard normalization for all transormations as used in training
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],         # Normalizing the image with specific mean and standard deviation
                                          std=[0.229, 0.224, 0.225])
    transformation = transforms.Compose([transforms.Resize(256),             # Resize the image to 256x256 pixels
                                         transforms.CenterCrop(224),         # Crops the given Image at the center to 224x224 pixels
                                         transforms.ToTensor(),              # Convert the image to the PyTorch Tensor data type
                                         normalization])
    
    # Set device to CPU
    device = torch.device("cpu")
    
    # Load image
    img = Image.open(img_path)
    # Transform image
    img = transformation(img).float()
    # Change Tensor to a one-dimensional one
    img = img.unsqueeze(0)
    # Convert img to a Variable. A PyTorch Variable is a wrapper around a PyTorch Tensor.
    img = Variable(img)
    # Transform img to CPU-Datatype
    img = img.to(device)
    # Get prediction
    prediction = model(img)
    
    # Get Top-k predictions, default is 3
    topk_predictions = torch.topk(prediction, topk)
    
    # Use Softmax function to extract probabilities
    soft = nn.Softmax(dim=1)
    topk_probabilities = soft(topk_predictions.values).cpu().detach().numpy()[0]
    
    # Get Top-k indices
    topk_indices = topk_predictions.indices.cpu().detach().numpy()[0]
    topk_labels = [class_names[k] for k in topk_indices]

    # return class_name
    return [topk_probabilities, topk_labels]