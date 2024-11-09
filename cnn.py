import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
# from google.colab.patches import cv2_imshow  # Only for Colab

# Custom preprocessing function for applying multiple filters
def custom_preprocessing(img):
    # Convert the image from RGB to BGR (for OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Apply pixelation
    def pixelate_image(img, pixelation_scale=10):
        (height, width) = img.shape[:2]
        small_image = cv2.resize(img, (width // pixelation_scale, height // pixelation_scale), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated_image

    img = pixelate_image(img)

    # 2. Adjust contrast and brightness
    def adjust_contrast_brightness(img, contrast=2.3, brightness=10):
        return cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)

    img = adjust_contrast_brightness(img)

    # 3. Apply color jittering (noise addition)
    def color_jittering(img, jitter_amount=100):
        h, w, c = img.shape
        noise = np.random.randint(0, jitter_amount, (h, w, 3), dtype='uint8')
        jittered_image = np.clip(img + noise, 0, 255).astype('uint8')
        return jittered_image

    img = color_jittering(img)

    # 4. Apply Canny edge detection
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)

    # Convert the edges to RGB to match the 3-channel image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Ensure both img and edges are of the same type (e.g., float32)
    img = img.astype('float32') / 255.0  # Ensure image values are scaled to [0, 1]
    edges = edges.astype('float32') / 255.0

    # Combine the original image with the edges for visualization
    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

    # Convert the image back from BGR to RGB (for Keras)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# Define the model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Set up ImageDataGenerator with custom preprocessing function
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalization
    validation_split=0.15,
    horizontal_flip=True,
    preprocessing_function=custom_preprocessing  # Custom preprocessing function
)

# Training data generator
train_gen = datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize real-vs-fake\\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_gen = datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize real-vs-fake\\validate",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Create a separate ImageDataGenerator for the test set with only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Test data generator (with shuffle=False to ensure proper evaluation)
test_gen = test_datagen.flow_from_directory(
    r"F:\\third year mini project codes\\ORIGINAL DATASETS\\resize dataset\\resize real-vs-fake\\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important for evaluation
)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Optional: Train the model
# model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=20,
#     # steps_per_epoch=train_gen.samples // 32,
#     # validation_steps=val_gen.samples // 32
# )
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch):
  if epoch<=8:
    return(0.0005)
  elif (epoch>8 and epoch<=20):
    return(0.0002)
  elif (epoch>20 and epoch<=32):
    return(0.0001)
  elif (epoch>32 and epoch<=40):
    return(0.00005)
  else:
    return(0.00001)


def scheduler2(epoch):
  if epoch < 40:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (40 - epoch))

# adding my own modification to decrease the learning rate further
callback = LearningRateScheduler(scheduler)
callback2 = LearningRateScheduler(scheduler2)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    callbacks=[callback],
   
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
model.save('deepfake_detection_modelcnn.keras')

# Plot Training Performance if history is defined
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