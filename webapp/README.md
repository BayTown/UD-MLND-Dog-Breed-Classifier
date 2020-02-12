# Web Applicaton built with Flask

[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://dog-breed-classifier.us-east-1.elasticbeanstalk.com/index)

## Description

This web application was developed with flask and was trimmed and prepared for deployment from AWS Beanstalk.
In this simple application I provide the trained model for prediction. The end user can easily upload his picture and it will be forwarded to the result page after the prediction process. On this page the end user is presented with the results and the uploaded image.

The site will be available until __19/02/2020__

![Screenshot Webapp][screenshot]

## Requirements

The requirements can be found in this path in the file `requirements.txt`.

In the deploy process at AWS Beanstalk these dependencies are automatically installed.
The only important thing is that this `requirements.txt` is located in the root path.

## Flask

The following tutorials have helped me with the development of my first Flask application:

[Getting started with Python Flask framework (Part 1)](https://medium.com/techkylabs/getting-started-with-python-flask-framework-part-1-a4931ce0ea13)

[Getting Started with Python Flask framework (Part 2)](https://medium.com/techkylabs/getting-started-with-python-flask-framework-part-2-5838ddc5d9a7)

[Getting started with Python Flask framework (part 3)](https://medium.com/techkylabs/getting-started-with-python-flask-framework-part-3-1f0e355c9be5)

## Deploy on AWS Beanstalk

Here is important information that helped me to set up the web application on AWS Beanstalk:

[What is AWS Elastic Beanstalk?](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Welcome.html)

If this is also your first time, you can follow this tutorial on AWS:
[Getting started using Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/GettingStarted.html)

## .elasticbeanstalk - The tricky thing

In the deployment process after AWS Beanstalk I encountered a few obstacles.

### The dependencies of the pip package PILLOW

PILLOW requires JPEG image codecs which must be available on the operating system.  
Before the requirements are installed within the deployment process you have to tell AWS that you have to install additional packages on the Linux-OS. These installations are done with a file whose name you can choose freely but must have the extension `.config`. The file must be located in the folder `.ebextensions`. In my case I have named the file `01-flask.config`.  
Here is a part of the file:

    packages:
     yum:
      libjpeg-turbo-devel: []
    libpng-devel: []

Here you must pay attention to the indentation of the lines. There must not be any tabulators. Only spaces.

### WSGIApplicationGroup

At the end I still had problems. After deploy the website was not reachable. There was still an entry missing in the .ebextensions/01-flask-config described above:

    container_commands:
     AddGlobalWSGIGroupAccess:
      command: "if ! grep -q 'WSGIApplicationGroup %{GLOBAL}' ../wsgi.conf ; then echo 'WSGIApplicationGroup %{GLOBAL}' >> ../wsgi.conf; fi;"

After that it worked.
Here i have more information:
[WSGIApplicationGroup](https://modwsgi.readthedocs.io/en/develop/configuration-directives/WSGIApplicationGroup.html)

## Improvements

The inference could have been designed with a RESTful-API but unfortunately I didn't have the time. This is also a question of cost, because with AWS you need one instance for the inference and one instance for the web server.

[screenshot]: https://github.com/BayTown/UD-MLND-Dog-Breed-Classifier/tree/master/webapp/static/uploads/Screenshot_webapp.png "Screenshot Webapp"
