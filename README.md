**Wall and Ceiling Segmentation AR App**
A real-time Augmented Reality (AR) mobile application for Diamond Paints, enabling users to visualize paint colors on interior walls and ceilings with 96% segmentation accuracy and >0.9 IoU. Built using a DeepLabV3-inspired model with a pre-trained MobileNetV2 backbone, optimized for mobile deployment with int8 quantization and 20% pruning, achieving ~0.7–1s inference on mid-range devices. Integrated into a React Native app, enhancing realism by 30% with dynamic lighting and texture recognition.
Training and validation metrics: Loss (0.05–0.1), Accuracy (0.96), IoU (>0.9) over 50 epochs.
Table of Contents

1. Project Overview
2. Model Architecture
3. Repository Structure
4. Prerequisites
5. Setup Instructions
6. Training the Model
7. Evaluating the Model
8. Visualizing Metrics
9. Converting to TFLite
10. Using the Model Locally
11. Deploying to React Native
12. Results
13. License
14. Credits

1. Project Overview
This project develops an AI-driven AR app for Diamond Paints, allowing users to:

Segment walls (green), ceilings (red), and background (blue) in real-time using a smartphone camera.
Overlay virtual paint colors from Diamond Paints’ catalog, supporting interior design decisions.
Enhance realism by 30% (user testing) with dynamic lighting and texture recognition.

Key Achievements:

96% segmentation accuracy, >0.9 IoU on ~5,000 images.
0.7–1s inference on mid-range devices (1–2MB TFLite model).
Optimized with int8 quantization (2–3x faster) and 20% pruning (~5–10% faster).
Deployed in React Native for iOS/Android, competing with apps like Dulux Visualizer.

Model Architecture

Backbone: Pre-trained MobileNetV2 for efficient feature extraction.
Architecture: DeepLabV3-inspired with simplified ASPP (64 filters) and boundary refinement (32 filters).
Input: 224x224 RGB images, 3 classes (Background, Wall, Ceiling).
Loss: Combined IoU-Dice-Focal loss (0.4IoU + 0.4Dice + 0.2*Focal).
Training: 50 epochs, ~5,000 images, Adam with cosine decay (1e-3 initial learning rate).
Optimizations: Int8 quantization, 20% pruning, separable convolutions (~15–20% faster).

Repository Structure
Wall-Ceiling-Segmentation-AR/
├── optimized_wall_segmentation_with_fallback.py  # Training and TFLite conversion
├── evaluate_model_metrics.py                    # Evaluate validation loss, accuracy, IoU
├── visualize_validation_metrics.py              # Plot training/validation metrics
├── test_wall_segmentation_tflite.py             # Test TFLite model inference
├── WallCeilingSegmentation.js                   # React Native integration
├── Final_Wall_Segmentation.tflite               # Pre-trained TFLite model (~1–2MB)
├── validation_metrics.png                       # Training metric plots
├── README.md                                    # This file
├── LICENSE                                      # MIT License

Prerequisites

Hardware: Mid-range GPU (e.g., NVIDIA GTX 1060) for training; CPU for inference.
OS: Windows/Linux/Mac (tested on Windows 10).
Dependencies:pip install tensorflow==2.11.0 tensorflow-model-optimization==0.7.3 numpy==1.23.5 opencv-python==4.11.0 albumentations==1.3.0 matplotlib==3.7.1


Node.js: For React Native (v18.x recommended).
Data: ~5,000 annotated images (not included due to size; structure below):Annotated_Dataset/
├── train/
│   ├── images/augmented/images/*.jpg
│   ├── masks/augmented/masks/*.png
├── validation/
│   ├── images/*.jpg (~224 images)
│   ├── masks/*.png (~224 masks)



Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/Wall-Ceiling-Segmentation-AR.git
cd Wall-Ceiling-Segmentation-AR


Install Dependencies:
pip install -r requirements.txt

Create requirements.txt:
tensorflow==2.11.0
tensorflow-model-optimization==0.7.3
numpy==1.23.5
opencv-python==4.11.0
albumentations==1.3.0
matplotlib==3.7.1


Prepare Dataset:

Organize ~5,000 images/masks as shown above.
Adjust paths in optimized_wall_segmentation_with_fallback.py:train_images_dir = 'path/to/Annotated_Dataset/train/images'
val_images_dir = 'path/to/Annotated_Dataset/validation/images'
val_masks_dir = 'path/to/Annotated_Dataset/validation/masks'




Download Pre-trained TFLite Model:

Final_Wall_Segmentation.tflite is included in the repo.
Optionally, train your own model (see below).



Training the Model

Run Training:
python optimized_wall_segmentation_with_fallback.py


Output: Final_Wall_Model_1.0.keras, training_history.npy.
Time: ~1–2 hours on a mid-range GPU (596 steps/epoch, 50 epochs).
Metrics: ~96% accuracy, >0.9 IoU, ~0.05–0.1 validation loss.


Save History:

Ensures training_history.npy for visualization:np.save('training_history.npy', history.history)





Evaluating the Model

Run Evaluation:python evaluate_model_metrics.py


Output: evaluation_results.txt:Validation Loss: ~0.05–0.1
Validation Accuracy: ~0.96
Validation IoU: >0.9


Input: Final_Wall_Model_1.0.keras, validation dataset (~224 images).



Visualizing Metrics

Run Visualization:python visualize_validation_metrics.py


Output: validation_metrics.png (loss, accuracy, IoU curves).
Input: training_history.npy.



Converting to TFLite

Run Conversion (if not using pre-trained Final_Wall_Segmentation.tflite):

Conversion is included in optimized_wall_segmentation_with_fallback.py.
Output: Final_Wall_Segmentation.tflite (~1–2MB).
Calibration: Uses 200 representative samples for int8 quantization.


Verify TFLite Model:
python test_wall_segmentation_tflite.py


Output: Sample predictions (blue=Background, green=Wall, red=Ceiling).
Inference Time: ~0.7–1s on mid-range devices.



Using the Model Locally

Test TFLite Model:

Use test_wall_segmentation_tflite.py for local inference:python test_wall_segmentation_tflite.py


Input: Any 224x224 RGB image (e.g., interior photo).
Output: Segmentation mask (blue=Background, green=Wall, red=Ceiling).
Requirements: TensorFlow 2.11.0, OpenCV.


Example:
import tensorflow as tf
import cv2
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='Final_Wall_Segmentation.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = cv2.imread('test_image.jpg')
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
mask = np.argmax(output[0], axis=-1)

# Visualize mask
mask_color = np.zeros((224, 224, 3), dtype=np.uint8)
mask_color[mask == 0] = [255, 0, 0]  # Background: Blue
mask_color[mask == 1] = [0, 255, 0]  # Wall: Green
mask_color[mask == 2] = [0, 0, 255]  # Ceiling: Red
cv2.imwrite('output_mask.png', mask_color)



Deploying to React Native

Setup React Native:
npm install -g react-native-cli
npx react-native init WallCeilingAR
cd WallCeilingAR
npm install @tensorflow/tfjs @tensorflow/tfjs-react-native


Copy TFLite Model:

Place Final_Wall_Segmentation.tflite in WallCeilingAR/assets/.


Integrate Model:

Use WallCeilingSegmentation.js (included) in your React Native app.
Example usage:import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { loadTFLiteModel } from '@tensorflow/tfjs-react-native';

async function segmentImage(image) {
  const model = await loadTFLiteModel('assets/Final_Wall_Segmentation.tflite');
  // Preprocess image (224x224, normalize)
  const tensor = tf.tensor4d([image], [1, 224, 224, 3]);
  // Run inference
  const output = await model.predict(tensor);
  // Process mask (0=Background, 1=Wall, 2=Ceiling)
  return output.argMax(-1).dataSync();
}




Build and Run:
npx react-native run-android  # or run-ios


Output: AR app visualizing paint colors on segmented walls/ceilings.



Results

Metrics:
Validation Loss: ~0.05–0.1
Validation Accuracy: ~0.96
Validation IoU: >0.9


Sample Output:Sample TFLite output: Blue=Background, Green=Wall, Red=Ceiling.
Performance:
~0.7–1s inference on mid-range mobile devices.
~1–2MB TFLite model size.
30% realism enhancement (user testing).



License
This project is licensed under the MIT License.
Credits

Author: Bilal Saif
Organization: Diamond Paints
Contact: [bilalsaifcheema7576@gmail.com or https://www.linkedin.com/in/bilalsaif7576/]
Acknowledgments: Built with TensorFlow, TFLite, and React Native for an innovative AR experience.

