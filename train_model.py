import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'data/training_set' # IMPORTANT: Change this to your dataset path

def create_data_generators(data_dir, target_size, batch_size):
    """Initializes and returns data generators with augmentation."""
    # Rescale all images by 1/255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% of training data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary', # Since we have only 2 classes (cat/dog)
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    print("Class Indices (0=Cat, 1=Dog):", train_generator.class_indices)
    return train_generator, validation_generator

def build_cnn_model(input_shape):
    """Defines the Convolutional Neural Network (CNN) architecture."""
    model = Sequential([
        # 1. Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # 2. Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # 3. Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # 4. Flattening
        Flatten(),

        # 5. Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.5),

        # 6. Output Layer (Sigmoid for binary classification)
        Dense(1, activation='sigmoid') 
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # Binary loss for 2 classes
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # 1. Prepare Data
    train_generator, validation_generator = create_data_generators(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE
    )
    
    # Get the input shape (Height, Width, Channels)
    input_shape = IMAGE_SIZE + (3,) 

    # 2. Build Model
    model = build_cnn_model(input_shape)
    model.summary()

    # 3. Train Model
    print("Starting Model Training...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # 4. Save Model for Deployment
    model_filename = 'cat_dog_model.h5'
    model.save(model_filename)
    print(f"\nModel training complete and saved as: {model_filename}")