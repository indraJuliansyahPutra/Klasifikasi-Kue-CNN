import streamlit as st
from predict import Predictor
from display import ImageDisplay

def main():
    # Set judul aplikasi Streamlit
    st.title("Klasifikasi Gambar Kue")

    # Menu di sidebar untuk memilih opsi
    menu = st.sidebar.selectbox("Menu", ["Performa Model", "Prediksi"])

    # Daftar label kelas (nama-nama kue)
    class_labels = ['Dadar Gulung', 'Kastengel', 'Klepon', 'Lapis', 'Lumpur', 'Putri Salju', 'Risoles', 'Serabi']

    # Objek untuk menampilkan gambar
    image_display = ImageDisplay()

    # Objek untuk melakukan Prediksi
    prediction = Predictor(class_labels)

    if menu == "Performa Model":
        # Pilih model yang akan dievaluasi
        col1, col2 = st.columns(2)
        with col1:
            eval_model_choice = st.selectbox("Pilih Arsitektur Model", ["Xception", "VGG-19"])
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])

        with col2:
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001])
            layer_setting = st.selectbox("Layer Setting", ["Unfreeze", "Freeze"])

        # Tombol untuk menampilkan grafik evaluasi
        if st.button("Tampilkan"):
            # Dapatkan konfigurasi lengkap model
            model_config = {
                "architecture": eval_model_choice,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "layer_setting": layer_setting,
            }
            # Tampilkan grafik evaluasi berdasarkan konfigurasi
            eval_images = image_display.get_evaluation_images(model_config)

            # Display the evaluation images
            st.write("### Akurasi dan Loss Model")
            col1, col2 = st.columns(2)
            with col1:
                st.image(eval_images["accuracy"], caption=f"Akurasi {eval_model_choice}_{batch_size}_{learning_rate}_{layer_setting}", use_container_width=True)
            with col2:
                st.image(eval_images["loss"], caption=f"Loss {eval_model_choice}_{batch_size}_{learning_rate}_{layer_setting}", use_container_width=True)
            st.write("### Confusion Matrix")
            st.image(eval_images["confusion_matrix"], caption=f"Confusion Matrix {eval_model_choice}_{batch_size}_{learning_rate}_{layer_setting}", use_container_width=True)

    elif menu == "Prediksi":
        # Unggah gambar terlebih dahulu
        uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Pilih model setelah gambar dipilih
            # Membagi jadi 2 bagian
            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("Pilih Arsitektur Model", ["Xception", "VGG19"])
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
            with col2:
                learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001])
                layer_setting = st.selectbox("Layer Setting", ["Unfreeze", "Freeze"])

            
            with col1:
                # Tampilkan gambar yang diunggah
                img = image_display.get_uploaded_image(uploaded_file)
                st.image(img, caption='Gambar yang diunggah.', use_container_width=True)

            with col2:
                # Muat model sesuai pilihan pengguna
                model_config = {
                    "architecture": model_choice,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "layer_setting": layer_setting,
                }
                model = prediction.load_selected_model(model_config)

                # Tambahkan tombol "Prediksi"
                if st.button("Prediksi", use_container_width=True):
                    # Prediksi kelas gambar
                    label, probabilities = prediction.predict_image(uploaded_file, model)

                    # Tampilkan hasil prediksi kelas
                    if max(probabilities) > 0.5:
                        st.markdown(f"<h2 style='color: #4CAF50; text-align: center; font-size: 28px'><b>Hasil Prediksi: Kue {label}</b></h2>", unsafe_allow_html=True)
                        st.write("Probabilities untuk masing-masing kelas:")
                        col3, col4 = st.columns(2)
                        half = len(class_labels) // 2
                        with col3:
                            for i in range(half):
                                st.write(f"{class_labels[i]}: {probabilities[i]:.2f}")
                        with col4:
                            for i in range(half, len(class_labels)):
                                st.write(f"{class_labels[i]}: {probabilities[i]:.2f}")
                    else:
                        st.markdown(f"<h2 style='color: #d14d4d; text-align: center; font-size: 28px'><b>Probabilitas tertinggi dibawah 0.5, model tidak cukup yakin dengan hasil prediksi. </b></h2>", unsafe_allow_html=True)
                        col3, col4 = st.columns(2)
                        half = len(class_labels) // 2
                        with col3:
                            for i in range(half):
                                st.write(f"{class_labels[i]}: {probabilities[i]:.2f}")
                        with col4:
                            for i in range(half, len(class_labels)):
                                st.write(f"{class_labels[i]}: {probabilities[i]:.2f}")

if __name__ == "__main__":
    app = main()
