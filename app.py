import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Page config
st.set_page_config(
    page_title="Klasifikasi Kualitas Udara",
    page_icon="🍃",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #dff0d8;
        color: #3c763d;
    }
    .stWarning {
        background-color: #fcf8e3;
        color: #8a6d3b;
    }
    .stError {
        background-color: #f2dede;
        color: #a94442;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'air_quality_ann_model.h5')
    scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

def predict_row(row_values, model, scaler):
    input_data = np.array([row_values])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    classes = ['Baik', 'Sedang', 'Tidak Sehat']
    return classes[class_idx], confidence

def main():
    st.title("🍃 Klasifikasi Kualitas Udara (ANN)")
    st.markdown("Aplikasi ini menggunakan **Artificial Neural Network** untuk mengklasifikasikan kualitas udara berdasarkan parameter sensor.")
    
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("Model atau Scaler tidak ditemukan! Harap jalankan training terlebih dahulu.")
        st.info("Jalankan `python src/train_model.py` di terminal.")
        return

    # Tabs for Single Prediction and Batch Prediction
    tab1, tab2 = st.tabs(["🧩 Prediksi Satuan", "📂 Prediksi Massal (Upload)"])

    with tab1:
        st.subheader("Input Parameter Sensor")
        col1, col2 = st.columns(2)
        
        with col1:
            pm10 = st.number_input("PM10 (pm_10) - µg/m³", min_value=0.0, value=20.0)
            pm25 = st.number_input("PM2.5 (pm_duakomalima) - µg/m³", min_value=0.0, value=15.0)
            so2 = st.number_input("SO2 (so2) - µg/m³", min_value=0.0, value=5.0)
            
        with col2:
            co = st.number_input("CO (co) - mg/m³", min_value=0.0, value=0.5)
            o3 = st.number_input("O3 (o3) - µg/m³", min_value=0.0, value=30.0)
            no2 = st.number_input("NO2 (no2) - µg/m³", min_value=0.0, value=10.0)
            
        st.write("---")
        
        if st.button("Prediksi Kualitas Udara"):
            # Prepare input in strict order: ['pm_10', 'pm_duakomalima', 'so2', 'co', 'o3', 'no2']
            input_data = np.array([[pm10, pm25, so2, co, o3, no2]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            classes = ['Baik', 'Sedang', 'Tidak Sehat']
            result_class = classes[class_idx]
            
            st.header(f"Hasil: {result_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.caption("*Nilai confidence diperoleh dari probabilitas kelas tertinggi yang dihasilkan oleh model ANN, dan tidak selalu merepresentasikan kepastian absolut.*")
            
            # Show probability bar
            prob_df = pd.DataFrame(prediction, columns=classes)
            st.bar_chart(prob_df.T)
            
            if class_idx == 0:
                st.success("Kualitas udara Bagus. Aman untuk beraktivitas di luar.")
            elif class_idx == 1:
                st.warning("Kualitas udara Sedang. Hati-hati bagi kelompok sensitif.")
            else:
                st.error("Kualitas udara Tidak Sehat! Hindari aktivitas di luar ruangan.")

    with tab2:
        st.subheader("Upload File Data (Excel/CSV)")
        st.info("Pastikan file memiliki kolom: **pm_10, pm_duakomalima, so2, co, o3, no2**")
        st.markdown("_Tips: Aplikasi akan otomatis mencoba mengganti nama kolom seperti 'PM10' menjadi 'pm_10' jika ditemukan._")
        
        uploaded_file = st.file_uploader("Upload file Excel atau CSV", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Auto-rename logic
                rename_map = {
                    'PM10': 'pm_10',
                    'PM2.5': 'pm_duakomalima',
                    'pm2.5': 'pm_duakomalima',
                    'Pm10': 'pm_10',
                    'Pm2.5': 'pm_duakomalima',
                    'SO2': 'so2',
                    'CO': 'co',
                    'O3': 'o3',
                    'NO2': 'no2'
                }
                # Check for columns and rename if match found
                df = df.rename(columns=rename_map)
                
                required_cols = ['pm_10', 'pm_duakomalima', 'so2', 'co', 'o3', 'no2']
                
                st.write("Preview Data (setelah penyesuaian):")
                st.dataframe(df.head())
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Kolom berikut tidak ditemukan dalam file (setelah auto-rename): {', '.join(missing_cols)}")
                    st.warning("Harap namai kolom sesuai format: pm_10, pm_duakomalima, so2, co, o3, no2")
                else:
                    if st.button("Proses Prediksi Massal"):
                        with st.spinner("Sedang memproses..."):
                            results = []
                            confidences = []
                            
                            # Prepare data for batch prediction (Specify order explicitly)
                            X_input = df[required_cols].values
                            X_scaled = scaler.transform(X_input)
                            
                            predictions = model.predict(X_scaled)
                            
                            classes = ['Baik', 'Sedang', 'Tidak Sehat']
                            
                            for pred in predictions:
                                class_idx = np.argmax(pred)
                                results.append(classes[class_idx])
                                confidences.append(np.max(pred) * 100)
                            
                            # Assign results
                            df['Hasil Prediksi'] = results
                            df['Confidence (%)'] = confidences
                            
                            # Validation Logic (Optional)
                            possible_gt_cols = ['Kategori', 'Quality', 'Label', 'Actual', 'Kelas', 'categori', 'category']
                            gt_col = next((c for c in df.columns if c in possible_gt_cols), None)
                            
                            extra_cols = ['Hasil Prediksi', 'Confidence (%)']
                            
                            if gt_col:
                                def check_suitability(row):
                                    actual = row[gt_col]
                                    pred = row['Hasil Prediksi']
                                    
                                    if isinstance(actual, (int, float, np.number)):
                                        mapping = {0: 'Baik', 1: 'Sedang', 2: 'Tidak Sehat'}
                                        actual_str = mapping.get(int(actual), str(actual))
                                    else:
                                        actual_str = str(actual)
                                        
                                    return "Sesuai" if actual_str.lower() == pred.lower() else "Tidak Sesuai"

                                df['Kesesuaian'] = df.apply(check_suitability, axis=1)
                                extra_cols.insert(1, 'Kesesuaian')
                            
                            # Reorder columns
                            cols = extra_cols + [c for c in df.columns if c not in extra_cols]
                            df = df[cols]
                            
                            st.success("Prediksi Selesai!")
                            st.dataframe(df)
                            
                            # Convert back to CSV for download
                            csv = df.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                "Download Hasil Prediksi (CSV)",
                                csv,
                                "hasil_prediksi_udara.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            st.caption("*Nilai confidence diperoleh dari probabilitas kelas tertinggi yang dihasilkan oleh model ANN, dan tidak selalu merepresentasikan kepastian absolut.*")
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

    st.write("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 50px;'>
        Model ANN dilatih menggunakan data indeks polutan udara dan diimplementasikan untuk tujuan akademik.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
