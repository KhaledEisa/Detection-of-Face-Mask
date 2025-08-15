# State-of-the-Art Face Mask Detection System 🎭🤖

A professional-grade face mask detection system using modern deep learning techniques with **real datasets**.

## 🚀 Features

- **🧠 Multiple Architectures**: EfficientNet, ResNet50V2, MobileNetV3
- **📊 Professional Training**: Transfer learning, data augmentation, class balancing
- **🔍 Smart Dataset Analysis**: Automatic dataset structure detection
- **📈 Comprehensive Evaluation**: Accuracy, precision, recall, confusion matrices
- **⚡ Real-time Inference**: Live webcam detection
- **💾 Model Persistence**: Save and load trained models
- **🎯 Optimized Parameters**: Automatic parameter suggestion based on dataset size

## 📁 Project Structure

```
face-mask-detection/
├── face_mask_detection_sota.py    # Main training script
├── dataset_setup.py               # Dataset analysis helper
├── quick_inference.py              # Quick testing script
├── requirements.txt                # Dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your dataset** (one of these structures):

**Option A: Pre-split dataset**
```
your_dataset/
├── train/
│   ├── mask/
│   └── no_mask/
├── val/
│   ├── mask/
│   └── no_mask/
└── test/
    ├── mask/
    └── no_mask/
```

**Option B: Simple structure (auto-split)**
```
your_dataset/
├── mask/
└── no_mask/
```

## 🎯 Quick Start

### 1. Analyze Your Dataset
```bash
python dataset_setup.py path/to/your/dataset
```

This will:
- ✅ Analyze dataset structure
- ✅ Count images per class  
- ✅ Check class balance
- ✅ Suggest optimal training parameters
- ✅ Generate training command

### 2. Train the Model

Use the suggested command from step 1, or manually:

```bash
python face_mask_detection_sota.py \
    --data_dir "path/to/your/dataset" \
    --architecture efficientnet \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --fine_tune_epochs 20
```

### 3. Test Your Model

**Webcam testing:**
```bash
python quick_inference.py --webcam
```

**Single image testing:**
```bash
python quick_inference.py --image your_photo.jpg
```

## 🧠 Model Architectures

| Architecture | Best For | Parameters | Speed |
|--------------|----------|------------|-------|
| **MobileNetV3** | Small datasets, Mobile deployment | ~5M | ⚡⚡⚡ |
| **EfficientNet** | Balanced performance | ~4M | ⚡⚡ |
| **ResNet50V2** | Large datasets, High accuracy | ~23M | ⚡ |

## 📊 Training Process

### Phase 1: Transfer Learning (50 epochs)
- ✅ Freeze base model weights
- ✅ Train custom classification head
- ✅ Learning rate: 0.001

### Phase 2: Fine-tuning (20 epochs)  
- ✅ Unfreeze all layers
- ✅ Fine-tune entire model
- ✅ Lower learning rate: 0.0001

### Advanced Features:
- 🔄 **Data Augmentation**: Rotation, zoom, flip, brightness
- ⚖️ **Class Balancing**: Automatic class weight calculation
- 📉 **Early Stopping**: Prevent overfitting
- 📈 **Learning Rate Scheduling**: Adaptive learning rate
- 💾 **Model Checkpointing**: Save best model automatically

## 📈 Performance Monitoring

The system automatically generates:

- **📊 Training History**: Accuracy, loss, precision, recall plots
- **🎯 Confusion Matrix**: Detailed classification results  
- **📋 Classification Report**: Per-class metrics
- **📝 Training Log**: CSV file with epoch-by-epoch metrics
- **⚙️ Model Configuration**: JSON file with training parameters

## 🎮 Command Reference

### Training Commands

**Basic training:**
```bash
python face_mask_detection_sota.py --data_dir your_dataset
```

**Custom parameters:**
```bash
python face_mask_detection_sota.py \
    --data_dir your_dataset \
    --architecture resnet \
    --img_size 256 \
    --batch_size 16 \
    --epochs 100 \
    --fine_tune_epochs 30
```

### Inference Commands

**Live webcam:**
```bash
python quick_inference.py --webcam
```

**Specific model:**
```bash
python quick_inference.py --webcam --model my_custom_model.h5
```

**Image testing:**
```bash
python quick_inference.py --image test_photo.jpg --model best_face_mask_model.h5
```

### Dataset Analysis

**Full analysis:**
```bash
python dataset_setup.py your_dataset_path
```

## 🔧 Optimization Tips

### For Small Datasets (<1000 images):
- Use **MobileNet** architecture
- Smaller image size (128x128)
- More data augmentation
- Lower batch size (16)

### For Large Datasets (>10000 images):
- Use **ResNet** architecture  
- Higher image size (256x256)
- Larger batch size (64+)
- More training epochs

### For GPU Memory Issues:
- Reduce batch size
- Use MobileNet architecture
- Reduce image size
- Enable mixed precision training

## 📋 Expected Results

With a **good dataset** (2000+ balanced images):

| Metric | Expected Range |
|--------|----------------|
| **Accuracy** | 92-98% |
| **Precision** | 90-97% |
| **Recall** | 90-97% |
| **Training Time** | 30-60 minutes |

## 🔍 Troubleshooting

### Common Issues:

**"No module named tensorflow"**
```bash
pip install tensorflow==2.15.0
```

**"Out of memory" errors**
```bash
# Reduce batch size
python face_mask_detection_sota.py --batch_size 8
```

**Low accuracy**
- Check dataset quality and balance
- Increase training epochs
- Try different architecture
- Add more data augmentation

**Webcam not working**
```bash
# Try different camera index
python quick_inference.py --webcam --camera 1
```

## 📊 File Outputs

After training, you'll get:
- ✅ `best_face_mask_model.h5` - Best model (highest validation accuracy)
- ✅ `face_mask_model_final.h5` - Final model (last epoch)
- ✅ `training_history.png` - Training curves visualization
- ✅ `confusion_matrix.png` - Classification results
- ✅ `training_log.csv` - Detailed training metrics
- ✅ `training_config.json` - Training parameters
- ✅ `suggested_training_command.txt` - Optimal command for your dataset

## 🎭 Real-Time Detection Features

- **🎯 Accurate Detection**: State-of-the-art model performance
- **⚡ Fast Processing**: Optimized for real-time use
- **📊 Confidence Scores**: See prediction confidence
- **🎨 Visual Feedback**: Color-coded bounding boxes
- **📸 Screenshot Capture**: Save results with 's' key
- **🔄 Easy Controls**: 'q' to quit, 's' to save

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **OpenCV Community** for computer vision tools
- **EfficientNet Authors** for the efficient architecture
- **Transfer Learning Research** for enabling fast training

---

🎉 **Ready to train with your real dataset!** Follow the Quick Start guide above.
