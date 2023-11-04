# imports
from load_data import LoadData
import matplotlib.pyplot as plt
from load_data import PreprocessData
import imageio
from sklearn.model_selection import train_test_split
from model import UNet4, UNet3, UNet2
import tensorflow as tf
import numpy as np
from visualization import vis_input, vis_preprocess, plot_train, vis_output, dice_plot
from testmodel import test_model
from training import train_model

# the following variables are the paths to the image and mask files
path = '/Users/fungi/PycharmProjects/cs6073hw3/Data/train/image'
path2 = '/Users/fungi/PycharmProjects/cs6073hw3/Data/train/mask'
# call "LoadData" function with paths as input
img, msk = LoadData(path, path2)
# function shows the image and corresponding mask
show_images = 1  # number of images to show
vis_input(show_images,path,path2,img,msk)  # function call
# the following are the desired image and mask dimensions for preprocessing
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]
# call function to preprocess data
X, y = PreprocessData(img, msk, target_shape_img, target_shape_mask, path, path2)
# calls function to display image and mask after preprocessing
vis_preprocess(X,y)
# splits data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
# calls 4-layer UNet
unet4 = UNet4(input_size=(128,128,3), n_filters=32, n_classes=2)
metric_names4 = ['mean_io_u', 'val_mean_io_u']  # names of metrics for plot
# calls 3-layer UNet
unet3 = UNet3(input_size=(128,128,3), n_filters=32, n_classes=2)
metric_names3 = ['mean_io_u_1', 'val_mean_io_u_1']  # names of metrics for plot
# calls 2-layer UNet
unet2 = UNet2(input_size=(128,128,3), n_filters=32, n_classes=2)
metric_names2 = ['mean_io_u_2', 'val_mean_io_u_2']  # names of metrics for plot

learn_rate = 0.0001  # desired learning rate for training
epochs = 50  # desired number of epochs for training
# function called to train each model
results4 = train_model(unet4,X_train,y_train,learn_rate,epochs)
results3 = train_model(unet3,X_train,y_train,learn_rate,epochs)
results2 = train_model(unet2,X_train,y_train,learn_rate,epochs)
# function called to plot model training histories
plot_train(results4,metric_names4)
plot_train(results3,metric_names3)
plot_train(results2,metric_names2)
# function called to get predictions from test data
pred_y4 = test_model(unet4, X_test, y_test)
pred_y3 = test_model(unet3, X_test, y_test)
pred_y2 = test_model(unet2, X_test, y_test)

index = 1  # test image to show predictions for
# function called to display input image, target mask, and all model predictions
vis_output(index,X_test,y_test,pred_y4[0],pred_y3[0],pred_y2[0])
# plots of dice coefficient for each test image
dice_plot(y_test,pred_y4)
dice_plot(y_test,pred_y3)
dice_plot(y_test,pred_y2)
# saving models
unet4.save('UNet4layer.keras')
unet3.save('UNet3layer.keras')
unet2.save('UNet2layer.keras')
# show plots
plt.show()
