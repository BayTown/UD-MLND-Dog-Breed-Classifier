# Udacity MLND Capstone Project Dog Breed Classifier

[![Website dog-breed-classifier](https://img.shields.io/website-up-down-green-red/http/shields.io.svg?style=flat-square&logo=appveyor)](http://dog-breed-classifier.us-east-1.elasticbeanstalk.com/index)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=flat-square&logo=appveyor)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square&logo=appveyor)](https://lbesson.mit-license.org/)

## Project description

In this project I developed a CNN for the recognition of dog breeds.
Based on a picture of a dog, an algorithm will give an estimate of the breed of the dog.
If the image of a person is given, the algorithm should reproduce the most similar dog breed.

I have also developed a web application to facilitate access for users. This web application was developed with Flask and made available on AWS Beanstalk.

## Requirements

This project was done on a Linux-OS ([Ubuntu 19.10](https://ubuntu.com/download/desktop)) with an [Anaconda distribution](https://www.anaconda.com/).

> Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing (data science, machine learning > > applications, large-scale data processing, predictive analytics, etc.)
> [Wikipedia](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution))

To start directly with the correct requirements for Anaconda I have added a file called environment.yml to the repo. You can create a
[conda virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with this file:

```bash
conda env create -f environment.yml
```

The first line of the yml file sets the new environment's name.
And then you can activate this new environment:

```bash
conda activate envmlcv
```

If you do not want to use the environment.yml to create a new virtual environment then you need to install with the command `conda install ...` the following packages:

```bash
numpy
glob
pandas
opencv
matplotlib
tqdm
torch
torchvision
PIL
```

## Explanation in depth

## Webapp

This web application was developed with Flask and optimized for AWS Beanstalk and prepared for use. In this simple application I provide the trained model for the prediction. The user can simply upload an image and it will be forwarded to the result page after the inference/prediction process. On this page the user is shown the results and the uploaded image.

You can find the complete webapp and more informations in the folder `webapp`

The site will be available until __19/02/2020__  
[Link to the Webapp](http://dog-breed-classifier.us-east-1.elasticbeanstalk.com/index)  
![Screenshot Webapp](https://user-images.githubusercontent.com/32474126/74352894-72e32a00-4db9-11ea-8f9d-fd0261802ad9.png)
![Screenshot_webapp_result](https://user-images.githubusercontent.com/32474126/74354084-21d43580-4dbb-11ea-8209-4e03e9e0ecb4.png)
