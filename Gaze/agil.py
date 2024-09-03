import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model

BATCH_SIZE = 50
num_epoch = 50
num_action = 18
SHAPE = (84, 84, 1)  # height * width * channel
dropout = 0.0

###############################
# Architecture of the network #
###############################

gaze_heatmaps = L.Input(shape=(SHAPE[0], SHAPE[1], 1))
g = L.BatchNormalization()(gaze_heatmaps)

imgs = L.Input(shape=SHAPE)
x = L.Multiply()([imgs, g])
x_intermediate = x
x = L.Conv2D(32, (8, 8), strides=2, padding='same')(x)
x = L.BatchNormalization()(x)
x = L.Activation('relu')(x)
x = L.Dropout(dropout)(x)

x = L.Conv2D(64, (4, 4), strides=2, padding='same')(x)
x = L.BatchNormalization()(x)
x = L.Activation('relu')(x)
x = L.Dropout(dropout)(x)

x = L.Conv2D(64, (3, 3), strides=1, padding='same')(x)
x = L.BatchNormalization()(x)
x = L.Activation('relu')(x)

# ============================ channel 2 ============================
orig_x = L.Conv2D(32, (8, 8), strides=2, padding='same')(imgs)
orig_x = L.BatchNormalization()(orig_x)
orig_x = L.Activation('relu')(orig_x)
orig_x = L.Dropout(dropout)(orig_x)

orig_x = L.Conv2D(64, (4, 4), strides=2, padding='same')(orig_x)
orig_x = L.BatchNormalization()(orig_x)
orig_x = L.Activation('relu')(orig_x)
orig_x = L.Dropout(dropout)(orig_x)

orig_x = L.Conv2D(64, (3, 3), strides=1, padding='same')(orig_x)
orig_x = L.BatchNormalization()(orig_x)
orig_x = L.Activation('relu')(orig_x)

x = L.Average()([x, orig_x])
x = L.Dropout(dropout)(x)
x = L.Flatten()(x)
x = L.Dense(512, activation='relu')(x)
x = L.Dropout(dropout)(x)
logits = L.Dense(num_action, name="logits")(x)
prob = L.Activation('softmax', name="prob")(logits)

# Only include `prob` in the model's outputs for training
model = Model(inputs=[imgs, gaze_heatmaps], outputs=prob)

opt = K.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)

# LOAD the Atari-HEAD Dataset in your way
from load_data import *

tar_file = './breakout/92_RZ_3504740_Aug-23-11-27-56.tar.bz2'
label_file = './breakout/92_RZ_3504740_Aug-23-11-27-56.txt'
gaze_model = './human_gaze_breakout.npz'

d = Dataset(tar_file, label_file)
d.load_predicted_gaze_heatmap(gaze_model)  # npz file (predicted gaze heatmap)
d.standardize()

model.fit([d.train_imgs, d.train_GHmap], d.train_lbl, BATCH_SIZE, epochs=num_epoch, shuffle=True, verbose=2)
model.save("model.hdf5")
