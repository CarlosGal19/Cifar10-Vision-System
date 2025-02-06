import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import matplotlib.pyplot as plt

data, metadata = tfds.load('cifar10', as_supervised=True, with_info=True)

train_data, test_data = data['train'], data['test']
class_names = metadata.features['label'].names

print(class_names)

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_data = train_data.map(normalize)
test_data = test_data.map(normalize)

train_data = train_data.cache()
test_data = test_data.cache()

plt.figure(figsize=(10, 10))
plt.title('Training Images')
for i, (image, label) in enumerate(train_data.take(25)):
    image = image.numpy()
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
plt.close()

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

amount_train_data = metadata.splits['train'].num_examples
amount_test_data = metadata.splits['test'].num_examples

BATCH_SIZE = 50
train_data = train_data.repeat().shuffle(amount_train_data).batch(BATCH_SIZE)
test_data = test_data.batch(amount_test_data)

history = model.fit(train_data, epochs = 45, steps_per_epoch = math.ceil(amount_train_data / BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_data)

figure, axis = plt.subplots(1, 2)

axis[0].plot(history.history['accuracy'])
axis[0].set_title('Model accuracy')
axis[0].set_ylabel('Accuracy')
axis[0].set_xlabel('Epoch')
axis[0].legend(['Train', 'Test'], loc='upper left')

axis[1].plot(history.history['loss'])
axis[1].set_title('Model loss')
axis[1].set_ylabel('Loss')
axis[1].set_xlabel('Epoch')
axis[1].legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.close()

for test_images, test_labels in test_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

plt.figure(figsize=(12, 12))
plt.title('First 16 images from the test set and their predicted labels')
plt.axis('off')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = test_images[i]
    plt.imshow(image, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    color = 'green' if predicted_label == test_labels[i] else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nReal: {test_labels[i]}", color=color)

plt.tight_layout()
plt.show()
plt.close()

model.save(r"C:\Users\Carlos Galindo\Desktop\first_partial_exam_AI\model.h5")
