from flask import Flask, request, render_template
from commons import face_detector, dog_detector, predict_breed_transfer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        print('request files')
        print(request.files)

        if request.files is None:
            print('Upload without a selected file')
            return render_template('index.html')

        if 'file' not in request.files:
            print('file not uploaded')
            return render_template('index.html')

        file = request.files['file']
        image = file.read()

        if image == b'':
            print('no file selected to upload')
            return render_template('index.html')

        dog_detected = False
        human_detected = False


        # Detect dog or human
        if dog_detector(image):
            dog_detected = True
            print('Hello dog! Here are the predictions for your dog breed')
        elif face_detector_ext(image):
            human_detected = True
            print('Hello human! If you were a dog, you could look like')
        else:
            print('Oh... No dog or human could be identified.')
        # Get and print predictions
        if dog_detected or human_detected:
            topk_predictions = predict_breed_transfer(image)
            for k in range(3):
                print('Top {0:2} prediction:{1:7.2f}% - {2}'.format(k+1, topk_predictions[0][k]*100, topk_predictions[1][k]))
        
        return render_template('result.html', top1_pred=topk_predictions[0][0]*100,
                                              top2_pred=topk_predictions[0][1]*100,
                                              top3_pred=topk_predictions[0][2]*100,
                                              top1_label=topk_predictions[1][0],
                                              top2_label=topk_predictions[1][1],
                                              top3_label=topk_predictions[1][2])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)