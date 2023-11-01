# imports
from loaddata import LoadData
import matplotlib.pyplot as plt
from loaddata import PreprocessData
import imageio
from sklearn.model_selection import train_test_split
from model import UNet
from model import get_model
import tensorflow as tf
import numpy as np


path = '/Users/fungi/PycharmProjects/pythonProject2/Data/image'
path2 = '/Users/fungi/PycharmProjects/pythonProject2/Data/mask'

img, msk = LoadData(path, path2)
'''
show_images = 1
for i in range(show_images):
    img_view = imageio.v2.imread(path + '/' + img[i])
    mask_view = imageio.v2.imread(path2 + '/' + msk[i])
    print(img_view.shape)
    print(mask_view.shape)
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(img_view)
    arr[0].set_title('Image ' + str(i))
    arr[1].imshow(mask_view)
    arr[1].set_title('Masked Image ' + str(i))
#plt.show()
'''
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]

X, y = PreprocessData(img, msk, target_shape_img, target_shape_mask, path, path2)
'''
# QC the shape of output and classes in output dataset
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 3 classes : background, pet, outline
print(np.unique(y))
np.set_printoptions(threshold=np.inf)
#print(y[50])

# Visualize the output
image_index = 0
fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(X[image_index])
arr[0].set_title('Processed Image')
arr[1].imshow(y[image_index,:,:,0])
arr[1].set_title('Processed Masked Image ')
plt.show()
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#print(X_train.shape)
#'''
unet = get_model((128, 128),3)  # UNet(input_size=(128, 128, 3), n_filters=32, n_classes=3)
learn_rate = 0.001
opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
unet.compile(optimizer=opt,
             loss="sparse_categorical_crossentropy",  # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
              metrics=['accuracy'])
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=1, min_lr=0.00001)

results = unet.fit(X_train, y_train, batch_size=20, epochs=10, validation_split=0.2)  # , callbacks=reduce_lr

fig, axis = plt.subplots(1, 2, figsize=(20, 5))
axis[0].plot(results.history["loss"], color='r', label = 'train loss')
axis[0].plot(results.history["val_loss"], color='b', label = 'dev loss')
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[1].plot(results.history["accuracy"], color='r', label = 'train accuracy')
axis[1].plot(results.history["val_accuracy"], color='b', label = 'dev accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()
plt.show()

unet.evaluate(X_test, y_test)
def VisualizeResults(index):
    img = X_test[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_test[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_test[index,:,:,0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:,:,0])
    arr[2].set_title('Predicted Masked Image ')

index = 1
VisualizeResults(index)
plt.show()
#'''
