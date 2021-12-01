# Breast Cancer Image Classification with DL

A deep learning project that classifies breast ultrasound images into three categories: **Benign**, **Malignant**, and **Normal** using a DenseNet121 architecture.

## 🎯 Project Overview

This project implements a convolutional neural network based on DenseNet121 to automatically classify breast ultrasound images. The model achieves high accuracy in distinguishing between different breast tissue conditions, which can assist medical professionals in early detection and diagnosis.

## 📊 Dataset

The dataset contains **158 breast ultrasound images** divided into three classes:

- **Benign**: 90 images (57.0%)
- **Malignant**: 42 images (26.6%) 
- **Normal**: 26 images (16.4%)

### Dataset Structure
```
test_images/
├── benign/          # 90 images
├── malignant/       # 42 images
└── normal/          # 26 images
```

## 🏗️ Model Architecture

- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Input Size**: 256x256 pixels
- **Classes**: 3 (Benign, Malignant, Normal)
- **Transfer Learning**: Utilizes pre-trained weights for feature extraction
- **Fine-tuning**: Custom classification head for breast cancer detection

## 📈 Performance Metrics

The model achieves the following performance on the test set:

- **Test Accuracy**: 94.94%
- **Test Loss**: 0.1365
- **Evaluation**: Confusion matrix and classification report available

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Streamlit App

1. **Start the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Upload an image**: Use the file uploader to select a breast ultrasound image (JPG, JPEG, or PNG)

3. **Get predictions**: The model will automatically classify the image and display the result

### Model Loading

The app automatically downloads and combines the pre-trained model from GitHub. The model is split into 34 parts for efficient distribution and is reconstructed in memory during runtime.

## 📁 Project Structure

```
Breast-Cancer-Image-Classification-with-DenseNet121/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── streamlit_app.py            # Streamlit web application
├── model_training.ipynb        # Model training notebook
├── model_evaluation.ipynb      # Model evaluation notebook
├── test_images/                # Test dataset
│   ├── benign/                 # Benign images
│   ├── malignant/              # Malignant images
│   └── normal/                 # Normal images
└── splitted_model/             # Model files (split into parts)
    ├── model.h5.part01
    ├── model.h5.part02
    └── ... (34 parts total)
```

## 🔧 Technical Details

### Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities
- **H5py**: HDF5 file handling
- **Requests**: HTTP library

### Image Preprocessing

1. **Resize**: Images are resized to 256x256 pixels
2. **Normalization**: Pixel values are normalized to [0, 1] range
3. **Batch Processing**: Single image is expanded to batch dimension

### Model Features

- **Caching**: Model loading is cached for faster subsequent predictions
- **Error Handling**: Robust error handling for image processing
- **Real-time Prediction**: Instant classification results
- **User-friendly Interface**: Simple drag-and-drop interface

## 📚 Usage Examples

### Using the Streamlit App

1. Open your web browser and navigate to the Streamlit app
2. Click "Choose a breast ultrasound image" 
3. Select an image file from your computer
4. View the uploaded image and prediction result

### Class Mapping

```python
class_mapping = {
    0: 'Benign',
    1: 'Malignant', 
    2: 'Normal'
}
```

## 🔬 Model Evaluation

The model evaluation includes:

- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Detailed precision, recall, and F1-score metrics
- **Individual Predictions**: Sample predictions with confidence scores

## ⚠️ Important Notes

- **Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.
- **Model Limitations**: The model is trained on a specific dataset and may not generalize to all types of breast ultrasound images.
- **Data Privacy**: Ensure patient privacy when using medical images.

