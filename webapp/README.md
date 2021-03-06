# Web Applicaton built with Flask

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=flat-square&logo=appveyor)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square&logo=appveyor)](https://lbesson.mit-license.org/)

## Description

This web application was developed with Flask and optimized for AWS Beanstalk and prepared for use. In this simple application I provide the trained model for the prediction. The user can simply upload an image and it will be forwarded to the result page after the inference/prediction process. On this page the user is shown the results and the uploaded image.

![Screenshot Webapp](https://user-images.githubusercontent.com/32474126/74352894-72e32a00-4db9-11ea-8f9d-fd0261802ad9.png)
![Screenshot_webapp_result](https://user-images.githubusercontent.com/32474126/74354084-21d43580-4dbb-11ea-8209-4e03e9e0ecb4.png)

## Requirements

The requirements for pip can be found in this path in the file `requirements.txt`.

These were extracted from the virtual environment with the following command:

```bash
$pip freeze > requirements.txt
```

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

```yaml
packages:
 yum:
  libjpeg-turbo-devel: []
  libpng-devel: []
```

Here you must pay attention to the indentation of the lines. There must not be any tabulators. Only spaces.

### WSGIApplicationGroup

At the end I still had problems. After deploy the website was not reachable. There was still an entry missing in the .ebextensions/01-flask-config described above:

```yaml
container_commands:
 AddGlobalWSGIGroupAccess:
  command: "if ! grep -q 'WSGIApplicationGroup %{GLOBAL}' ../wsgi.conf ; then echo 'WSGIApplicationGroup %{GLOBAL}' >> ../wsgi.conf; fi;"
```

After that it worked.
Here i have more information:
[WSGIApplicationGroup](https://modwsgi.readthedocs.io/en/develop/configuration-directives/WSGIApplicationGroup.html)

## Improvements

The inference could have been designed with a RESTful-API but unfortunately I didn't have the time. This is also a question of cost, because with AWS you need one instance for the inference and one instance for the web server.
