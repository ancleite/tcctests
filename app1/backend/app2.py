import joblib
import os
from flask import Flask, request, jsonify
from google.cloud import storage
from PIL import Image
import io
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import tempfile
import cv2


# Carregar o modelo VGG16 pré-treinado
vgg_model = VGG16(weights='imagenet', include_top=False)

app = Flask(__name__)

temp_dir = tempfile.mkdtemp()

# Configuração do Google Cloud Storage
BUCKET_NAME = 'ensemble-models-tcc'

def load_model(name_model):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{name_model}.joblib")
    
    # Baixar o modelo para um diretório temporário
    temp_model_path = os.path.join(temp_dir, f"{name_model}.joblib")
    blob.download_to_filename(temp_model_path)
    
    # Carregar o modelo
    model = joblib.load(temp_model_path)
    return model

def extract_features(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    features = vgg_model.predict(x)
    return features.flatten()

# Carregar os modelos na inicialização
model_corn = load_model("clf_corn")
model_tomato = load_model("clf_tomato")
model_potato = load_model("clf_potato")

label_decoders = {
    'potato': {
        0: 'early blight',
        1: 'late blight',
        2: 'healthy'
    },
    'tomato': {
        0: 'bacterial spot',
        1: 'early blight',
        2: 'late blight',
        3: 'leaf mold',
        4: 'septoria leaf spot',
        5: 'Spider_mites Two-spotted_spider_mite',
        6: 'target spot',
        7: 'Yellow_Leaf_Curl_Virus',
        8: 'mosaic virus',
        9: 'healthy'
    },
    'corn': {
        0: 'Cercospora_leaf_spot Gray_leaf_spot',
        1: 'Common_rust',
        2: 'Northern_Leaf_Blight',
        3: 'healthy'
    }
}



def segment_leaf(image):
    # Converter a imagem para o espaço de cor HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Definir intervalos para a cor verde da folha (você pode ajustar os valores conforme necessário)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Criar uma máscara que isola apenas a cor verde na imagem
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Aplicar a máscara à imagem original
    result = cv2.bitwise_and(image, image, mask=mask)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar o maior contorno (presumindo que seja a folha)
        largest_contour = max(contours, key=cv2.contourArea)

        # Encontrar a caixa delimitadora do maior contorno
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Cortar a imagem para manter apenas a folha detectada
        cropped_leaf = result[y:y+h, x:x+w]

        return cropped_leaf
    else:
        # Retorna a imagem original se nenhum contorno foi encontrado
        return image



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    model_name = request.form.get('name')

    if model_name not in ['corn', 'tomato', 'potato']:
        return jsonify({'error': 'Invalid model name'}), 400
    
    # Processar a imagem
    try:
        img = Image.open(image_file.stream).convert('RGB')
        img = np.array(img)

        # Identificar a folha na imagem
        leaf_segmented = segment_leaf(img)

        # Redimensionar a folha segmentada para 224x224
        leaf_segmented_resized = cv2.resize(leaf_segmented, (224, 224))

        # Converter de volta para um array numpy
        img_array = np.array(leaf_segmented_resized)
        
        img_features = extract_features(img_array)
        img_features = img_features.reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500
    
    # Escolher o modelo
    if model_name == 'corn':
        model = model_corn
        decoder = label_decoders['corn']
    elif model_name == 'tomato':
        model = model_tomato
        decoder = label_decoders['tomato']
    else:
        model = model_potato
        decoder = label_decoders['potato']
    
    # Realizar a predição
    try:
        encoded_prediction = model.predict(img_features).tolist()
        # Decodificar a predição usando o dicionário apropriado
        decoded_prediction = [decoder.get(i, 'Unknown') for i in encoded_prediction]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'predictions': decoded_prediction})


if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=8080)
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(host='0.0.0.0', port=3030)
    #app.run()
