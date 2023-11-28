import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import segmentation_models as sm

def predict(model, image_path, input_shape, class_labels):
    # Load the test image
    test_image = load_img(image_path, target_size=input_shape[:2])
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    preprocess_input = sm.get_preprocessing('densenet121')

    # Preprocess the test image
    preprocessed_test_image = preprocess_input(test_image)

    # Perform prediction
    predicted = model.predict(preprocessed_test_image)
    predicted = np.squeeze(predicted)
    predicted = np.argmax(predicted, axis=-1)

    # Convert label values to actual values
    predicted_rgb = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
    for i in range(len(class_labels)):
        if i == 0:
            predicted_rgb[predicted == i] = [0, 0, 0]  # Background (Black)
        elif i == 1:
            predicted_rgb[predicted == i] = [255, 0, 0]  # Car (Red)
        elif i == 2:
            predicted_rgb[predicted == i] = [0, 255, 0]  # Wheel (Green)
        elif i == 3:
            predicted_rgb[predicted == i] = [0, 0, 255]  # Lights (Blue)
        elif i == 4:
            predicted_rgb[predicted == i] = [255, 255, 0]  # Window (Yellow)

    return predicted_rgb

# Set the necessary paths and parameters
model_path = "models/PSPNet_densenet121_model.h5"
image_path = "H:/Datasets/car-segmentation/images/003.png"
input_shape = (384, 384, 3)
class_labels = ['background', 'car', 'wheel', 'lights', 'window']

# Load the trained model
model = tf.keras.models.load_model(model_path, compile=False)

# Predict the mask
predicted = predict(model, image_path, input_shape, class_labels)

# Load the test image and mask
test_image = load_img(image_path, target_size=input_shape[:2])
test_mask_path = "H:/Datasets/car-segmentation/masks/003.png"
test_mask = load_img(test_mask_path, target_size=input_shape[:2], color_mode="grayscale")
test_mask = img_to_array(test_mask) / 255.

# Plot the test image, actual mask, and predicted mask
fig = plt.figure(figsize=(12, 4))
axs = fig.subplots(1, 3)
axs[0].imshow(test_image)
axs[0].set_title('Test Image')
axs[0].axis('off')
axs[1].imshow(test_mask.squeeze(), cmap='gray')
axs[1].set_title('Actual Mask')
axs[1].axis('off')
axs[2].imshow(predicted)
axs[2].set_title('Predicted')
axs[2].axis('off')
plt.tight_layout()
plt.show()

