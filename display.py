import streamlit as st
from PIL import Image

class ImageDisplay:
    def get_uploaded_image(self, img_file):
        # Open and resize the uploaded image
        img = Image.open(img_file)
        img = img.resize((300, 300))
        return img  # Return image instead of displaying it

    def get_evaluation_images(self, model_config):
        # Retrieve evaluation images based on model configuration
        architecture = model_config["architecture"]
        batch_size = model_config["batch_size"]
        learning_rate = model_config["learning_rate"]
        layer_setting = model_config["layer_setting"]

        # Generate file names dynamically based on configuration
        acc_filename = f"grafik/{architecture}_{layer_setting.upper()}_accuracy_{batch_size}_{learning_rate}.png"
        loss_filename = f"grafik/{architecture}_{layer_setting.upper()}_loss_{batch_size}_{learning_rate}.png"
        cm_filename = f"grafik/{architecture}_{layer_setting.upper()}_confusion_matrix_{batch_size}_{learning_rate}.png"

        try:
            # Load images dynamically
            acc_img = Image.open(acc_filename)
            loss_img = Image.open(loss_filename)
            cm_img = Image.open(cm_filename)
        except FileNotFoundError:
            st.error(f"Gambar evaluasi untuk konfigurasi berikut tidak ditemukan:\n"
                     f"- Arsitektur: {architecture}\n"
                     f"- Batch Size: {batch_size}\n"
                     f"- Learning Rate: {learning_rate}\n"
                     f"- Layer Setting: {layer_setting}")
            return None

        return {
            "accuracy": acc_img,
            "loss": loss_img,
            "confusion_matrix": cm_img
        }
    "models/Xception_UNFREEZE_model_lr_0.001_batch_16.h5"
    "models/Xception_UNFREEZE_model_lr_0.001_batch_16.h5"
