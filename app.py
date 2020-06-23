import glob
import os
import PIL.Image
from flask import Flask, request
import numpy as np
from keras.preprocessing import image
app = Flask(__name__)
import tensorflow as tf

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/cache/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
data_img=(glob.glob("uploads/*.*"))
# Load the previously saved weights
model =tf.keras.models.load_model("model/model.h5")
print(model.outputs)
@app.route('/test',methods=['POST'])
def test():
    img = request.files['image']
    print(img.filename)
    print(type(img))
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(saved_path)
    img = image.load_img(saved_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(saved_path)
    os.remove(saved_path)
    print(classes[0])
    if classes[0] > 0:
        #non_covid
        return "00"
    else:
        #covid
        return "11"

if __name__ == '__main__':
    app.run()
