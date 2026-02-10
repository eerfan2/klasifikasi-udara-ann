# Aplikasi Klasifikasi Kualitas Udara dengan ANN

Aplikasi ini menggunakan **Artificial Neural Network (ANN)** untuk mengklasifikasikan kualitas udara berdasarkan parameter sensor seperti PM2.5, PM10, NO2, CO, O3, dan SO2.

## Fitur
- **Dataset Sintetis**: Script untuk menghasilkan dummy data jika dataset asli belum tersedia.
- **Model ANN**: Melatih model deep learning sederhana menggunakan TensorFlow/Keras.
- **Web Interface**: Antarmuka interaktif menggunakan Streamlit untuk prediksi real-time.

## Instalasi

1.  **Pastikan Python Terinstall**
    Aplikasi ini dibuat dengan Python 3.10.

2.  **Install Dependencies**
    Jalankan perintah berikut di terminal:
    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan

1.  **Generate Data (Optional)**
    Jika belum ada data di folder `data/`, jalankan:
    ```bash
    python src/generate_data.py
    ```

2.  **Training Model**
    Latih model ANN dengan menjalankan:
    ```bash
    python src/train_model.py
    ```
    Model akan disimpan di folder `model/`.

3.  **Jalankan Aplikasi Web**
    Untuk membuka antarmuka prediksi:
    ```bash
    streamlit run app.py
    ```

## Struktur Folder
- `data/`: Menyimpan dataset CSV.
- `model/`: Menyimpan model `.h5` dan scaler `.pkl` setelah training.
- `src/`: Script untuk generate data dan training model.
- `app.py`: Main script untuk aplikasi Streamlit.
