from flask import Flask, request, render_template, url_for, redirect
from commons import face_detector, dog_detector, predict_breed_transfer, Get_VGG16_model, get_model
from imagehelper import fix_orientation
import numpy as np
import uuid
from PIL import Image
import io
import os
import sys
import traceback

# Define path to folder 'uploads' in static files
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set Allowed file extensions
application.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

# Set Pytorch Home direcotry path to root. Pytorch is using this path to save pretrained models
# Prevent permission errors on AWS Beanstalk 
os.environ['TORCH_HOME'] = './'

# Load models at startup
VGG16 = Get_VGG16_model()
model_transfer = get_model()


def allowed_image(filename):
    '''
    This function checks whether a file is an image.
    It checks whether the file extension is in the 'ALLOWED_IMAGE_EXTENSIONS' Variable
    Args:
        filename: path to an image
    Returns:
        Returns if file extension indicates an image, else false.
    '''
    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in application.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():
    '''
    Function for route 'index'. If GET then render just the index.html.
    If POST (form submit) then get uploaded image and detect
    if human or dog is present on the image. If this is the case,
    then this function will give the Filepath of the image to the predict function.
    If not a 'Sorry'-Page will be shown.
    Args:
        None
    Returns:
        If dog or humans is detected: render template for result with the topk-predictions and labels and also with url of the image and a variable if a human or a dog is present.
        If neither is detected: Render a template with a 'Sorry'-Page
    '''

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':

        # If request.files is empty then reload index-page
        if request.files is None:
            return render_template('index.html')

        # If parameter 'file' is not in request.files then reload index-page
        if 'file' not in request.files:
            return render_template('index.html')

        # Load request content
        file = request.files['file']

        # If upload was made without content then reload index-page
        if file.filename == '':
            return render_template('index.html')

        # Checks if the filename extension is not allowed
        if allowed_image(file.filename) == False:
            return render_template('index.html')


        # Create unique filename and save the file temporary to the static upload folder
        unique_filename = str(uuid.uuid4()) + '_' + file.filename
        path_filename = os.path.join(application.config['UPLOAD_FOLDER'], unique_filename)
        file.save(path_filename)

        # Fix orientation of the saved image and save inplace
        fix_orientation(path_filename)

        # Build ULR for the image
        part_url_path = 'uploads/' + unique_filename
        img_url = url_for('static', filename=part_url_path)
        
        dog_detected = False
        human_detected = False
        human_dog_text = ''

        # Detect dog or human
        if dog_detector(path_filename, VGG16):
            dog_detected = True
            human_dog_text = 'Dog'
            #print('Hello dog! Here are the predictions for your dog breed')
        elif face_detector(path_filename):
            human_detected = True
            human_dog_text = 'Human'
            #print('Hello human! If you were a dog, you could look like')
        else:
            #print('Oh... No dog or human could be identified.')
            return redirect(url_for('neither'))

        # Get and print topk-predictions
        if dog_detected or human_detected:
            topk_predictions = predict_breed_transfer(path_filename, model_transfer)
            #for k in range(3):
            #    print('Top {0:2} prediction:{1:7.2f}% - {2}'.format(k+1, topk_predictions[0][k]*100, topk_predictions[1][k]))

        # Render template for result with the topk-predictions and labels
        # and also with url of the image and a variable if a human or a dog is present.
        return render_template('result.html', top1_pred=np.round(topk_predictions[0][0]*100, 2),
                                              top2_pred=np.round(topk_predictions[0][1]*100, 2),
                                              top3_pred=np.round(topk_predictions[0][2]*100, 2),
                                              top1_label=topk_predictions[1][0],
                                              top2_label=topk_predictions[1][1],
                                              top3_label=topk_predictions[1][2],
                                              img_path=img_url,
                                              human_dog=human_dog_text)


@application.route('/neither')
def neither():
    return render_template('neither.html')

@application.errorhandler(Exception)
def handle_exception(e):
    return render_template('error.html', etext=e)

if __name__ == '__main__':
    application.run(debug=True)