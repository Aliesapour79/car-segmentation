import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import segmentation_models as sm
import matplotlib.pyplot as plt

# Set the necessary paths and parameters
image_dir = "H:\Datasets\car-segmentation\images"
mask_dir = "H:\Datasets\car-segmentation\masks"
input_shape = (384, 384, 3)
batch_size = 4
epochs = 2
num_classes = 5

# Custom data generator
class DataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size):
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_image_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_files = self.mask_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for image_file, mask_file in zip(batch_image_files, batch_mask_files):
            image = load_img(image_file, target_size=input_shape[:2])
            mask = load_img(mask_file, target_size=input_shape[:2], color_mode="rgb")

            image = img_to_array(image)
            mask = img_to_array(mask)

            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        masks = masks[:, :, :, 0]  # Convert RGB masks to grayscale

        # Convert class masks to one-hot encoded masks
        masks_one_hot = []
        for mask in masks:
            mask_one_hot = np.zeros((mask.shape[0], mask.shape[1], num_classes))
            for i in range(num_classes):
                mask_one_hot[:, :, i] = np.where(mask == i, 1, 0)
            masks_one_hot.append(mask_one_hot)
        masks_one_hot = np.array(masks_one_hot)
        preprocess_input = sm.get_preprocessing('densenet121')
        return preprocess_input(images), masks_one_hot

# Get the list of image and mask files
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".png")]
mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(".png")]

# Split the data into training and validation sets
split_index = int(0.8 * len(image_files))
train_image_files, val_image_files = image_files[:split_index], image_files[split_index:]
train_mask_files, val_mask_files = mask_files[:split_index], mask_files[split_index:]

# Create data generators for training and validation
train_data_generator = DataGenerator(train_image_files, train_mask_files, batch_size=batch_size)
val_data_generator = DataGenerator(val_image_files, val_mask_files, batch_size=batch_size)

# Build the PSPNet model
BACKBONE="densenet121"
model = sm.PSPNet(BACKBONE, input_shape=input_shape, classes=num_classes, activation='softmax')

# Compile the model
model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

# Train the model
history = model.fit(
    train_data_generator,
    validation_data=val_data_generator,
    epochs=epochs
)
# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Plot the training and validation IoU score
plt.plot(history.history['iou_score'], label='Training IoU Score')
plt.plot(history.history['val_iou_score'], label='Validation IoU Score')

plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

model.save("test_model.h5")

