from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Predictor:
    def __init__(self, class_labels):
        self.class_labels = class_labels

    def load_selected_model(self, model_config):
        """
        Load the model based on selected configuration.
        :param model_config: dict with architecture, batch size, learning rate, and layer setting
        :return: Loaded Keras model
        """
        subdir = 'models'
        architecture = model_config["architecture"]
        batch_size = model_config["batch_size"]
        learning_rate = model_config["learning_rate"]
        layer_setting = model_config["layer_setting"].lower()

        # Generate the model file name dynamically
        filename = f'{architecture}_{layer_setting.upper()}_model_lr_{learning_rate}_batch_{batch_size}.h5'
        path = f'{subdir}/{filename}'

        try:
            return load_model(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{filename}' not found in '{subdir}'.")

    def preprocess_image(self, img, target_size=(224, 224)):
        """
        Preprocess the image for prediction.
        :param img: PIL Image object
        :param target_size: Target size for the image (default is (224, 224))
        :return: Preprocessed image array
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict_image(self, img_file, model):
        """
        Predict the class of the uploaded image using the loaded model.
        :param img_file: File object of the uploaded image
        :param model: Keras model
        :return: Predicted class label and confidence scores
        """
        img = Image.open(img_file)
        img_array = self.preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        probabilities = predictions[0]
        return self.class_labels[predicted_class[0]], probabilities
