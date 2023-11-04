# imports
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


# defining function to train a model
# function takes keras model object, training data, desired learning rate, and desired number of epochs
# as input and returns the training history
def train_model(model,X_train,y_train,learn_rate,epochs):
    opt = tf.keras.optimizers.SGD(learning_rate=learn_rate)  # optimizer to train model with learning rate
    # compile model for training with above optimizer, loss function, and metrics
    model.compile(optimizer=opt,
                 loss="sparse_categorical_crossentropy",
                 metrics=[tf.keras.metrics.MeanIoU(num_classes=2,sparse_y_pred=False)])
    # callback to reduce learn rate when validation loss isn't improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.000001)
    # fit model with training data
    results = model.fit(X_train, y_train, batch_size=20, epochs=epochs, validation_split=0.2, callbacks=reduce_lr)
    # return model history
    return results
