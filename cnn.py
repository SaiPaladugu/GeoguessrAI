from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam

# Define the data generator with enhanced augmentation and validation split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNetV2-specific preprocessing
    horizontal_flip=True,                    # Flip images horizontally
    rotation_range=10,                       # Rotate up to 10 degrees
    zoom_range=0.1,                         # Zoom in/out by 10%
    shear_range=0.2,                         # Shear transformations
    width_shift_range=0.2,                   # Horizontal shifts
    height_shift_range=0.2,                  # Vertical shifts
    brightness_range=[0.8, 1.2],             # Brightness adjustments
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

# Build custom top layers with an additional dense layer
x = base_model.output
x = GlobalAveragePooling2D()(x)              # Reduce spatial dimensions
x = Dropout(0.5)(x)                          # Dropout to prevent overfitting
x = Dense(128, activation='relu')(x)         # Dense layer for more complexity
predictions = Dense(3, activation='softmax')(x)  # Output layer for 3 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the last 20 layers of the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True                   # Enable fine-tuning

# Compile the model with a smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-4),      # Smaller learning rate for fine-tuning
    loss='categorical_crossentropy',         # Loss for multi-class classification
    metrics=['accuracy']                     # Track accuracy
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',                      # Monitor validation loss
    patience=10,                             # Increased patience to 10
    restore_best_weights=True                # Restore best weights
)
checkpoint = ModelCheckpoint(
    'best_model.keras',                      # Save model in .keras format
    monitor='val_accuracy',                  # Save based on validation accuracy
    save_best_only=True                      # Save only the best model
)
reduce_lr = ReduceLROnPlateau(               # Reduce learning rate on plateau
    monitor='val_loss',
    factor=0.2,                              # Reduce by factor of 0.2
    patience=3,                              # Wait 3 epochs before reducing
    min_lr=1e-6                              # Minimum learning rate
)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,                               # Increased to 30 epochs
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Save the final model in .keras format
model.save('best_model.keras')
print("Training completed. Model saved to 'best_model.keras'.")