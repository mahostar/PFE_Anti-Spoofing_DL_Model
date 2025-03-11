"""
DepthFusion-ViT Model Training
-----------------------------
This script implements the training pipeline for the DepthFusion-ViT architecture,
a lightweight multi-modal anti-spoofing neural network optimized for edge devices.

The model integrates RGB image analysis and pseudo-depth estimation to detect
presentation attacks with high accuracy while maintaining low memory usage.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import json
from pathlib import Path
import random
from tqdm import tqdm

# Set TF memory growth to avoid consuming all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("Using CPU for training")

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, labels, batch_size=16, dim=(224, 224), n_channels=3,
                 n_classes=2, shuffle=True, augment=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_pseudo_depth(self, img):
        """Generate pseudo-depth map from RGB image"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        depth = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        return np.expand_dims(depth, axis=-1)

    def augment_image(self, img):
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = 0.8 + 0.4 * random.random()  # Contrast
            beta = 0.1 * random.random() - 0.05  # Brightness
            img = np.clip(alpha * img + beta * 255, 0, 255).astype(np.uint8)
        
        # Random rotation (slight)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        
        return img

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X_rgb = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_depth = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load RGB image
            img = cv2.imread(ID)
            if img is None:
                img = np.zeros((*self.dim, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, self.dim)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Data augmentation if enabled
            if self.augment:
                img = self.augment_image(img)
            
            # Normalize RGB
            img_normalized = img.astype(np.float32) / 255.0
            
            # Store RGB
            X_rgb[i,] = img_normalized
            
            # Generate and store pseudo-depth
            X_depth[i,] = self.generate_pseudo_depth(img)
            
            # Store class
            y[i] = self.labels[ID]

        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        # Pack into a dictionary for multi-input model
        X = {
            'rgb_input': X_rgb,
            'depth_input': X_depth
        }
        
        return X, y_categorical


def create_micro_vit_block(inputs, embedding_dim, num_heads, mlp_ratio=2.0):
    """Create a MicroViT Transformer block"""
    # Normalization layers
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)
    
    # Skip connection 1
    x = layers.Add()([attention_output, inputs])
    
    # MLP block
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(int(embedding_dim * mlp_ratio), activation='gelu')(y)
    y = layers.Dense(embedding_dim)(y)
    
    # Skip connection 2
    return layers.Add()([x, y])


def create_depth_fusion_vit_model(
    input_shape=(224, 224, 3),
    patch_size=16,
    embedding_dim=96,
    depth=4,
    num_heads=3,
    mlp_ratio=2.0,
    num_classes=2
):
    """Create the lightweight DepthFusion-ViT model architecture"""
    # Input layers
    rgb_input = keras.Input(shape=input_shape, name="rgb_input")
    depth_input = keras.Input(shape=(input_shape[0], input_shape[1], 1), name="depth_input")
    
    # 1. Process RGB input through Vision Transformer
    # Patch embedding
    patch_dim = patch_size * patch_size * input_shape[2]
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    patches = layers.Conv2D(
        filters=embedding_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embedding"
    )(rgb_input)
    
    # Reshape to sequence
    patch_seq = layers.Reshape((num_patches, embedding_dim))(patches)
    
    # Add positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
    patch_seq = patch_seq + pos_embedding
    
    # Transformer blocks
    x = patch_seq
    for i in range(depth):
        x = create_micro_vit_block(
            x, 
            embedding_dim=embedding_dim, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
    
    # Global average pooling
    rgb_features = layers.GlobalAveragePooling1D()(x)
    
    # 2. Process depth input
    depth_features = layers.Conv2D(32, 3, padding='same', activation='relu')(depth_input)
    depth_features = layers.MaxPooling2D()(depth_features)
    depth_features = layers.Conv2D(64, 3, padding='same', activation='relu')(depth_features)
    depth_features = layers.GlobalAveragePooling2D()(depth_features)
    depth_features = layers.Dense(embedding_dim, activation='relu')(depth_features)
    
    # 3. Feature fusion with attention
    combined_features = layers.Concatenate()([rgb_features, depth_features])
    
    # Self-attention on combined features
    attention = layers.Dense(2, activation='softmax')(combined_features)
    attention_rgb = layers.Multiply()([rgb_features, layers.Lambda(lambda x: x[:, 0:1])(attention)])
    attention_depth = layers.Multiply()([depth_features, layers.Lambda(lambda x: x[:, 1:2])(attention)])
    
    # Weighted combination
    fused_features = layers.Add()([attention_rgb, attention_depth])
    
    # 4. Classification head
    x = layers.Dense(embedding_dim, activation='relu')(fused_features)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(
        inputs=[rgb_input, depth_input],
        outputs=outputs,
        name="DepthFusion-ViT"
    )
    
    return model


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    # Cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # Focal loss
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    focal = weight * cross_entropy
    
    return tf.reduce_sum(focal, axis=-1)


def train_model(args):
    """Train the DepthFusion-ViT model"""
    print(f"Preparing to train DepthFusion-ViT model...")
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"DepthFusion-ViT_{timestamp}"
    
    # Create model directory
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Get image paths and labels
    train_images, train_labels, val_images, val_labels = prepare_dataset(args.data_dir)
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create data generators
    train_generator = DataGenerator(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        dim=(args.img_size, args.img_size),
        n_classes=2,
        shuffle=True,
        augment=True
    )
    
    val_generator = DataGenerator(
        val_images,
        val_labels,
        batch_size=args.batch_size,
        dim=(args.img_size, args.img_size),
        n_classes=2,
        shuffle=False,
        augment=False
    )
    
    # Create model
    model = create_depth_fusion_vit_model(
        input_shape=(args.img_size, args.img_size, 3),
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=2
    )
    
    # Model summary
    model.summary()
    
    # Save model architecture diagram
    keras.utils.plot_model(
        model, 
        to_file=os.path.join(model_dir, 'model_architecture.png'),
        show_shapes=True
    )
    
    # Compile model
    if args.use_focal_loss:
        loss_fn = focal_loss
    else:
        loss_fn = 'categorical_crossentropy'
    
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        ),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Learning rate scheduler
    def lr_scheduler(epoch, lr):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return args.learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            # Cosine decay
            decay_epochs = args.epochs - warmup_epochs
            epoch_decay = epoch - warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_decay / decay_epochs))
            return args.learning_rate * cosine_decay
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=1
        ),
        keras.callbacks.LearningRateScheduler(lr_scheduler),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger(os.path.join(model_dir, 'training_log.csv'))
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Save model configuration
    config = {
        'model_name': model_name,
        'timestamp': timestamp,
        'input_shape': [args.img_size, args.img_size, 3],
        'patch_size': args.patch_size,
        'embedding_dim': args.embedding_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'early_stopping_patience': args.patience,
        'train_samples': len(train_images),
        'val_samples': len(val_images),
        'use_focal_loss': args.use_focal_loss,
    }
    
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed. Model saved to {model_dir}")
    
    # Convert model to TFLite format
    if args.convert_to_tflite:
        convert_to_tflite(model, model_dir, config)


def prepare_dataset(data_dir):
    """Prepare dataset by loading image paths and labels"""
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise ValueError(f"Dataset directories not found. Expected 'real' and 'fake' subdirectories in {data_dir}")
    
    # Get all image paths
    real_images = [str(path) for path in Path(real_dir).glob("**/*.jpg")]
    fake_images = [str(path) for path in Path(fake_dir).glob("**/*.jpg")]
    
    # Create labels (0 for fake, 1 for real)
    real_labels = {img: 1 for img in real_images}
    fake_labels = {img: 0 for img in fake_images}
    
    # Combine and shuffle
    all_images = real_images + fake_images
    all_labels = {**real_labels, **fake_labels}
    
    # Check if we have enough data
    if len(all_images) < 10:
        raise ValueError(f"Not enough data found. Found only {len(all_images)} images. Need at least 10 for training.")
    
    # Split into train/val sets (80/20)
    combined = list(zip(all_images, [all_labels[img] for img in all_images]))
    random.shuffle(combined)
    all_images, all_labels_list = zip(*combined)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Convert to label dictionaries
    train_labels = {img: all_labels[img] for img in train_images}
    val_labels = {img: all_labels[img] for img in val_images}
    
    return train_images, train_labels, val_images, val_labels


def convert_to_tflite(model, model_dir, config):
    """Convert trained model to TFLite format with quantization"""
    print("Converting model to TFLite format...")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            rgb = np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
            depth = np.random.uniform(0, 1, (1, 224, 224, 1)).astype(np.float32)
            yield [rgb, depth]
    
    converter.representative_dataset = representative_dataset
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save model
    tflite_model_path = os.path.join(model_dir, 'model_quantized.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_model_path}")
    
    # Save model info
    model_info = {
        'model_name': config['model_name'],
        'input_shape': config['input_shape'],
        'file_size_kb': os.path.getsize(tflite_model_path) / 1024,
        'quantization': 'INT8'
    }
    
    with open(os.path.join(model_dir, 'tflite_model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DepthFusion-ViT model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing real/fake subdirectories')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save the trained model')
    
    # Model architecture parameters
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Size of image patches')
    parser.add_argument('--embedding_dim', type=int, default=96,
                       help='Dimension of token embeddings')
    parser.add_argument('--depth', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=2.0,
                       help='MLP expansion ratio')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of cross-entropy')
    
    # Deployment parameters
    parser.add_argument('--convert_to_tflite', action='store_true',
                       help='Convert to TFLite format after training')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args) 