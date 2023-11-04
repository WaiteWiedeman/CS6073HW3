# imports
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# function to calculate dice coefficient
# takes target mask and predicted mask as input and returns the dice coefficient to 3 decimals
def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)  # calculate intersection between target and prediction
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)  # calculate the sum of target and prediction
    dice = 2*intersect/total_sum  # calculate dice coefficient
    return round(dice, 3) #round up to 3 decimal places


# function to plot dice coefficients
# takes target masks and predicted masks as input
def dice_plot(y_test,y_pred):
    dice = np.zeros(len(y_pred))  # make array to store dice coefficients
    # loop calculates dice for all predictions
    for i in range(len(y_pred)):
        dice[i] = dice_coef(y_test[i],y_pred[i])
    print(np.mean(dice))  # print average dice coefficient
    # plot dice coefficient vs number of masks
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(y_pred)),dice)
    plt.xlabel('Ground Truth Image Number')
    plt.ylabel('Dice')
    plt.title('Dice coefficient')


# function to show image and mask
# takes desired number of images to show, path of images, path of masks, images, and masks as input
def vis_input(show_images,path,path2,img,msk):
    for i in range(show_images):
        img_view = imageio.v2.imread(path + '/' + img[i])  # opens image from path
        mask_view = imageio.v2.imread(path2 + '/' + msk[i])  # opens mask from path
        print(img_view.shape)
        print(mask_view.shape)
        # plots images
        fig, arr = plt.subplots(1, 2, figsize=(15, 15))
        arr[0].imshow(img_view)
        arr[0].set_title('Image ' + str(i))
        arr[1].imshow(mask_view)
        arr[1].set_title('Masked Image ' + str(i))


# function shows preprocessed images
# takes image and mask as input and shows images
def vis_preprocess(X,y):
    # print shapes of data
    print("X Shape:", X.shape)
    print("Y shape:", y.shape)
    # print unique content of y, should be 0 and 1
    print(np.unique(y))

    # plots preprocessed images
    image_index = 0
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(X[image_index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y[image_index, :, :, 0])
    arr[1].set_title('Processed Masked Image ')


# function to plot training history of model
# takes training history and name of metrics as input and plots training loss and IoU
def plot_train(results,metric_names):
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label='train loss')
    axis[0].plot(results.history["val_loss"], color='b', label='val loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(results.history[metric_names[0]], color='r', label='Mean IoU')
    axis[1].plot(results.history[metric_names[1]], color='b', label='Val Mean IoU')
    axis[1].set_title('IoU Comparison')
    axis[1].legend()


# function plots the predictions of each model
# takes the image index, test data, and predictions as input and plots the images
def vis_output(index,X_test,y_test,pred_y1,pred_y2,pred_y3):
    pred_mask1 = tf.argmax(pred_y1, axis=-1)
    pred_mask1 = pred_mask1[..., tf.newaxis]
    pred_mask2 = tf.argmax(pred_y2, axis=-1)
    pred_mask2 = pred_mask2[..., tf.newaxis]
    pred_mask3 = tf.argmax(pred_y3, axis=-1)
    pred_mask3 = pred_mask3[..., tf.newaxis]
    print(np.unique(pred_mask1))
    fig, arr = plt.subplots(1, 5, figsize=(15, 15))
    arr[0].imshow(X_test[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_test[index, :, :, 0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask1[:, :, 0])
    arr[2].set_title('4 Layer Predicted Masked Image ')
    arr[3].imshow(pred_mask2[:, :, 0])
    arr[3].set_title('3 Layer Predicted Masked Image ')
    arr[4].imshow(pred_mask3[:, :, 0])
    arr[4].set_title('2 Layer Predicted Masked Image ')
