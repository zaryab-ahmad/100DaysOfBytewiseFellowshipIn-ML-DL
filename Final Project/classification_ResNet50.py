from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('retinal_vessel_classifier_resnet50.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    img_path = 'temp_image.jpg'
    file.save(img_path)
    
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    class_names = ['Diabetic_Maculopathy', 'Macular_Edema', 'Normal']
    predicted_class = class_names[np.argmax(predictions)]
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
