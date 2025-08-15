# 🎭 3-Class Face Mask Detection System

A comprehensive deep learning system for detecting face mask compliance with **3 classes**: *With Mask*, *Without Mask*, and *Mask Worn Incorrectly*.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 🎯 **Key Features**

- ✅ **3-Class Detection**: With mask, without mask, mask worn incorrectly
- ✅ **Interactive Jupyter Notebook**: Train and experiment cell-by-cell
- ✅ **Transfer Learning**: EfficientNet, ResNet50V2, MobileNetV3 support
- ✅ **Real-time Detection**: Live webcam face mask detection
- ✅ **Data Visualization**: Comprehensive dataset analysis and training plots
- ✅ **Robust Architecture**: Error handling with CNN fallback
- ✅ **Class Balancing**: Automatic handling of imbalanced datasets

## 📊 **Dataset Compatibility**

**Primary Format**: Folder-organized datasets
```
Dataset/
├── with_mask/          # Images with correctly worn masks
├── without_mask/       # Images without masks
└── mask_incorrect/     # Images with incorrectly worn masks
```

**Tested On**: 17,964 images (5,988 per class) - perfectly balanced dataset

### **🔗 Dataset Source**
This project was developed and tested using the **Face Mask Detection Dataset** from Kaggle:

**📋 Dataset**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)  
**👤 Author**: vijaykumar1799  
**📊 Size**: 853 images with PASCAL VOC annotations + 17,964 organized images  
**🏷️ Classes**: 3-class system (with_mask, without_mask, mask_weared_incorrect)  
**✅ Quality**: High-quality, real-world images perfect for training robust models

## 🚀 **Quick Start**

### **1. Installation**
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn numpy pandas
```

### **2. Launch Jupyter Notebook**
```bash
jupyter notebook face_mask_detection_notebook.ipynb
```

### **3. Configure Your Dataset**
In Cell 1, update the dataset path:
```python
DATA_DIR = r"D:\Downloads\archive (1)\Dataset"  # Your dataset path
```

### **4. Run Training**
- **Quick Test**: Set `EPOCHS = 5` in Cell 8, run Cells 1-9 (5 minutes)
- **Full Training**: Use `EPOCHS = 30` for best accuracy (30-45 minutes)

### **5. Test Your Model**
- **Webcam**: Uncomment `test_webcam()` in Cell 13
- **Single Image**: Use `test_single_image()` in Cell 12

## 📋 **Notebook Structure**

| Cell | Purpose | Runtime | Description |
|------|---------|---------|-------------|
| 1 | **Setup & Config** | 10s | Import libraries, configure parameters |
| 2 | **Dataset Analysis** | 30s | Analyze folder structure and image counts |
| 3 | **Data Visualization** | 10s | Create class distribution charts |
| 4 | **Data Generators** | 1m | Setup data augmentation and validation |
| 5 | **Image Preview** | 30s | Display sample images from each class |
| 6 | **Build Model** | 1m | Create neural network architecture |
| 7 | **Model Summary** | 10s | Show model details and architecture |
| 8 | **Training Config** | 10s | Set epochs, callbacks, class weights |
| 9 | **🔥 Phase 1 Training** | 15-30m | Train classification head |
| 10 | **🔧 Fine-tuning** | 10-20m | Fine-tune entire model (optional) |
| 11 | **📊 Evaluation** | 2m | Generate confusion matrix and metrics |
| 12 | **Single Image Test** | As needed | Test model on individual images |
| 13 | **🎬 Webcam Testing** | As needed | Real-time face mask detection |

## 🧠 **Model Architectures**

### **Supported Models**
- **EfficientNetB0** (Most Accurate)
- **ResNet50V2** (Most Stable) 
- **MobileNetV3Large** (Fastest)
- **Custom CNN** (Fallback for compatibility issues)

### **Model Selection**
```python
# In Cell 1, choose your architecture:
ARCHITECTURE = "resnet"        # Recommended for stability
ARCHITECTURE = "efficientnet"  # Best accuracy (may have TF issues)
ARCHITECTURE = "mobilenet"     # Fastest inference
```

## 🎨 **Real-time Detection Colors**

When using webcam detection:
- 🟢 **Green Box** = With Mask (Correct)
- 🔴 **Red Box** = Without Mask
- 🟡 **Yellow Box** = Mask Incorrect (Improperly worn)

## 📈 **Expected Performance**

### **Training Results**
- **Accuracy**: 85-95% (with balanced dataset)
- **Training Time**: 30-60 minutes (full training)
- **Dataset Size**: Optimized for 10K+ images

### **Inference Speed**
- **CPU**: 5-15 FPS
- **GPU**: 15-30 FPS
- **Webcam**: Real-time detection

## 🔧 **Configuration Options**

### **Training Parameters**
```python
# Quick experimentation
EPOCHS = 5
FINE_TUNE_EPOCHS = 0
BATCH_SIZE = 16

# Production training
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
BATCH_SIZE = 32

# Large dataset training
EPOCHS = 50
FINE_TUNE_EPOCHS = 25
BATCH_SIZE = 64
```

### **Data Augmentation**
Built-in augmentation includes:
- Rotation (±25°)
- Width/Height shift (±20%)
- Zoom (±15%)
- Horizontal flip
- Brightness variation (80-120%)

## 📁 **Project Structure**

```
Detection-of-Face-Mask/
├── face_mask_detection_notebook.ipynb  # Main training notebook
├── README.md                           # This file
└── .git/                              # Git repository
```

## 🛠️ **Advanced Usage**

### **Custom Dataset**
1. Organize your images into 3 folders
2. Update `DATA_DIR` in Cell 1
3. Run Cells 2-4 to analyze your data
4. Adjust `BATCH_SIZE` based on dataset size

### **Hyperparameter Tuning**
```python
# Experiment with learning rates
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005))

# Adjust dropout rates
layers.Dropout(0.4)  # Increase for overfitting

# Modify dense layer sizes
layers.Dense(1024, activation='relu')  # Larger for complex datasets
```

### **Multi-GPU Training**
```python
# For multiple GPUs
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
```

## 📊 **Monitoring Training**

The notebook provides real-time visualization:
- **Accuracy plots**: Training vs validation curves
- **Loss plots**: Monitor overfitting
- **Class distribution**: Dataset balance analysis
- **Confusion matrix**: Per-class performance
- **Sample predictions**: Visual model testing

## 🚨 **Troubleshooting**

### **Common Issues**

**"TensorFlow shape mismatch"**
```python
# Switch to ResNet architecture
ARCHITECTURE = "resnet"
```

**"Out of memory"**
```python
# Reduce batch size
BATCH_SIZE = 16
```

**"Low accuracy"**
```python
# Increase training epochs
EPOCHS = 50
FINE_TUNE_EPOCHS = 25
```

**"Can't find dataset"**
```python
# Check your path
DATA_DIR = r"C:\Your\Actual\Dataset\Path"
```

## 🎯 **Use Cases**

- **COVID-19 Compliance Monitoring**
- **Workplace Safety Systems**
- **Public Health Research**
- **Educational Demonstrations**
- **Smart Building Access Control**

## 📚 **Requirements**

### **System Requirements**
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (CUDA-compatible for acceleration)

### **Python Dependencies**
```
tensorflow>=2.8.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
jupyterlab>=3.0.0
```

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch
3. Experiment with the notebook
4. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **TensorFlow/Keras** for deep learning framework
- **OpenCV** for computer vision capabilities  
- **Jupyter** for interactive development environment
- **Transfer Learning** models from TensorFlow Hub
- **Dataset**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) by vijaykumar1799 on Kaggle

## 🔗 **Related Projects**

- [Face Detection with OpenCV](https://github.com/opencv/opencv)
- [TensorFlow Computer Vision](https://www.tensorflow.org/tutorials/images)
- [Keras Applications](https://keras.io/api/applications/)

---

**Built with ❤️ for public health and safety**

*For questions or issues, please open a GitHub issue or contact the maintainers.*
