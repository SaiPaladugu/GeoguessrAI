import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the data generator with augmentation and validation split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNetV2-specific preprocessing
    horizontal_flip=True,                    # Flip images horizontally
    rotation_range=10,                       # Rotate up to 10 degrees
    zoom_range=0.1,                         # Zoom in/out by 10%
    validation_split=0.2                     # 20% for validation
)

# Load training data from directory
train_generator = train_datagen.flow_from_directory(
    'street_view_images',                    # Path to image folder
    target_size=(224, 224),                  # Resize images to 224x224
    batch_size=32,                           # Batch size
    class_mode='categorical',                # One-hot encoded labels
    subset='training'                        # Training subset
)

# Load validation data from directory
validation_generator = train_datagen.flow_from_directory(
    'street_view_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'                      # Validation subset
)

# Load pre-trained MobileNetV2 without top layer
base_model = MobileNetV2(
    weights='imagenet',                      # Use ImageNet weights
    include_top=False,                       # Exclude top classification layer
    input_shape=(224, 224, 3)                # Input shape: 224x224 RGB images
)

# Build custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)              # Reduce spatial dimensions
x = Dropout(0.5)(x)                          # Add dropout to prevent overfitting
predictions = Dense(3, activation='softmax')(x)  # Output layer for 3 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer='adam',                        # Adam optimizer
    loss='categorical_crossentropy',         # Loss for multi-class classification
    metrics=['accuracy']                     # Track accuracy
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',                      # Monitor validation loss
    patience=5,                              # Stop after 5 epochs of no improvement
    restore_best_weights=True                # Restore best weights
)
checkpoint = ModelCheckpoint(
    'best_model.h5',                         # Save model to file
    monitor='val_accuracy',                  # Save based on validation accuracy
    save_best_only=True                      # Save only the best model
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,                               # Maximum epochs
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]   # Apply callbacks
)

# Load the best model
model.load_weights('best_model.h5')

# Get class indices mapping
class_indices = train_generator.class_indices  # e.g., {'dubai': 0, 'ottawa': 1, 'tokyo': 2}
index_to_class = {v: k for k, v in class_indices.items()}  # Invert for prediction

# Function to predict city from an image
def predict_city(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_city = index_to_class[predicted_index]
    confidence = prediction[0][predicted_index]
    return predicted_city, confidence

# Test images to predict
test_images = [
    'testing_images/sample_dubai.jpg',
    'testing_images/sample_ottawa.jpg',
    'testing_images/sample_tokyo.jpg'
]

# Run predictions and print results
for image_path in test_images:
    predicted_city, confidence = predict_city(image_path)
    print(f'The predicted city for {image_path} is {predicted_city} with confidence {confidence:.2f}')