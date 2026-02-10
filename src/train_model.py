import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'air_quality_dummy.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    X = df.drop('Quality', axis=1)
    y = df['Quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels (already int, but good practice for categorical)
    # y is 0, 1, 2. No need for one-hot if using sparse_categorical_crossentropy
    
    # 3. Build ANN Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # 3 classes: Baik, Sedang, Tidak Sehat
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Train Model
    print("Training model...")
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # 5. Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['Baik', 'Sedang', 'Tidak Sehat']))
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baik', 'Sedang', 'Tidak Sehat'], yticklabels=['Baik', 'Sedang', 'Tidak Sehat'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # 6. Save Model & Scaler
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'air_quality_ann_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_model()
