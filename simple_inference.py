#!/usr/bin/env python3
"""
Simple Inference for 3-Class Face Mask Detection
For models trained on folder-organized datasets
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path

def test_webcam_simple(model_path="best_simple_3class_model.h5", img_size=224):
    """Test simple 3-class model with webcam"""
    print(f"📹 Testing simple 3-class model with webcam...")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("💡 Make sure you've trained the model first!")
        return
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Class names and colors
    classes = ['With Mask', 'Without Mask', 'Mask Incorrect']
    colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Green, Red, Yellow
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("🎬 Simple 3-Class detection started!")
    print("🎨 Colors: Green=With Mask, Red=Without Mask, Yellow=Incorrect")
    print("⌨️ Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_region = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_region, (img_size, img_size))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            predictions = model.predict(face_batch, verbose=0)[0]
            class_idx = np.argmax(predictions)
            predicted_class = classes[class_idx]
            confidence = predictions[class_idx]
            
            # Choose color based on prediction
            color = colors[class_idx]
            
            # Draw results
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Main label
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show all probabilities
            y_offset = y + h + 25
            for i, (class_name, prob) in enumerate(zip(classes, predictions)):
                text = f"{class_name}: {prob:.2f}"
                text_color = colors[i] if prob > 0.3 else (128, 128, 128)
                cv2.putText(frame, text, (x, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Info overlay
        cv2.putText(frame, "Simple 3-Class Face Mask Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Green=Mask, Red=No Mask, Yellow=Incorrect", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Simple 3-Class Mask Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"simple_result_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Simple 3-class detection completed")

def test_single_image_simple(model_path, image_path, img_size=224):
    """Test simple 3-class model on a single image"""
    print(f"📸 Testing simple 3-class model on: {image_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Preprocess
    image_resized = cv2.resize(image, (img_size, img_size))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Predict
    predictions = model.predict(image_batch, verbose=0)[0]
    classes = ['With Mask', 'Without Mask', 'Mask Incorrect']
    
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    print(f"🎯 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.4f}")
    print(f"📈 All scores:")
    for class_name, score in zip(classes, predictions):
        print(f"   {class_name}: {score:.4f}")
    
    # Show result
    color = (0, 255, 0) if predicted_class == 'With Mask' else (0, 0, 255) if predicted_class == 'Without Mask' else (0, 255, 255)
    cv2.putText(image, f"{predicted_class}: {confidence:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Simple 3-Class Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Simple 3-Class Face Mask Testing")
    parser.add_argument("--model", type=str, default="best_simple_3class_model.h5", 
                       help="Path to trained simple 3-class model")
    parser.add_argument("--image", type=str, help="Test single image")
    parser.add_argument("--webcam", action="store_true", help="Test with webcam")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    
    args = parser.parse_args()
    
    print("🧪 SIMPLE 3-CLASS TESTING")
    print("=" * 35)
    print("🏷️ Classes: With Mask, Without Mask, Mask Incorrect")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Make sure you've trained the simple model first!")
        print("   Run: python face_mask_detection_simple.py")
        return
    
    if args.image:
        # Test single image
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        test_single_image_simple(args.model, args.image, args.img_size)
    elif args.webcam:
        # Test with webcam
        test_webcam_simple(args.model, args.img_size)
    else:
        print("❓ Please specify either --image or --webcam")
        print("Examples:")
        print("  python simple_inference.py --webcam")
        print("  python simple_inference.py --image my_photo.jpg")

if __name__ == "__main__":
    main() 