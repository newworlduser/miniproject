import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom preprocessing function for applying multiple filters
def custom_preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
   
    # Pixelation
    def pixelate_image(img, pixelation_scale=10):
        (height, width) = img.shape[:2]
        small_image = cv2.resize(img, (width // pixelation_scale, height // pixelation_scale), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    img = pixelate_image(img)

    # Adjust contrast and brightness
    def adjust_contrast_brightness(img, contrast=2.3, brightness=10):
        return cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

    img = adjust_contrast_brightness(img)

    # Color jittering
    def color_jittering(img, jitter_amount=100):
        h, w, c = img.shape
        noise = np.random.randint(0, jitter_amount, (h, w, 3), dtype='uint8')
        return np.clip(img + noise, 0, 255).astype('uint8')

    img = color_jittering(img)

    # Canny edge detection
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    img = img.astype('float32') / 255.0
    edges = edges.astype('float32') / 255.0

    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# Load and define the model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze first few layers, fine-tune later layers
for layer in base_model.layers[:100]:  # Experiment with unfreezing more layers if needed
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Reduced units for less complexity
x = Dropout(0.5)(x)  # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Set up data generators with more augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    horizontal_flip=True,
    rotation_range=15,          # Random rotation
    width_shift_range=0.1,      # Horizontal translation
    height_shift_range=0.1,     # Vertical translation
    zoom_range=0.2,             # Random zoom
    preprocessing_function=custom_preprocessing
)

train_gen = datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize dataset\\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize dataset\\validate",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test data generator with only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize dataset\\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Compile the model with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)
print("Training completed, history is available:", 'history' in locals())
# Evaluate the model
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc}')


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_gen, steps=len(test_gen))
print(f'Test Accuracy: {test_acc}')
# # Optional: Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(test_gen, steps=test_gen.samples // 32)
# print(f'Test Accuracy: {test_acc}')
# --------------------------------------------------------------------------------------
# test_loss, test_acc = model.evaluate(test_gen, steps=test_gen.samples // 32)
# print(f'Test Accuracy: {test_acc}')
# ---------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = model.predict(test_gen)
y_pred = np.round(y_pred).astype(int)  # Convert probabilities to binary labels
cm = confusion_matrix(test_gen.classes, y_pred)
print(cm)
# -------------------------------------------------------------------------------------------
model.save('deepfake_detectioncnn2.keras')

# Plot Training Performance
if 'history' in locals():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
else:
    print("Training history not available.")