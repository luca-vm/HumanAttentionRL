'''
Example data loader for Atari-HEAD dataset
This file reads dataset by
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).
'''

import sys, os, re, threading, time, copy
import numpy as np
import tarfile
import cv2


def preprocess(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

class Dataset:
  def __init__(self, tar_fname, label_fname):
    t1=time.time()
    print ("Reading all training data into memory...")

    # Read action labels from txt file
    frame_ids, lbls = [], [] 
    with open(label_fname,'r') as f:
        for line in f:
            if line.startswith("frame_id") or line == "": 
                continue # skip head or empty lines
            dataline = line.split(',') 
            frame_id, lbl = dataline[0], dataline[5]
            if lbl == "null": # end of file
                break
            frame_ids.append(frame_id)
            lbls.append(int(lbl))
    self.train_lbl = np.asarray(lbls, dtype=np.int32)
    self.train_size = len(self.train_lbl)
    self.frame_ids = np.asarray(frame_ids)
    print(self.train_size)

    # Read training images from tar file
    imgs = [None] * self.train_size
    print("Making a temp dir and uncompressing PNG tar file")
    temp_extract_dir = "img_data_tmp/"
    if not os.path.exists(temp_extract_dir):
        os.mkdir(temp_extract_dir)
    tar = tarfile.open(tar_fname, 'r')
    tar.extractall(temp_extract_dir)
    png_files = tar.getnames()
    # get the full path
    temp_extract_full_path_dir = temp_extract_dir + png_files[0].split('/')[0]
    print("Uncompressed PNG tar file into temporary directory: " + temp_extract_full_path_dir)

    print("Reading images...")
    for i in range(self.train_size):
        frame_id = self.frame_ids[i]
        png_fname = temp_extract_full_path_dir + '/' + frame_id + '.png'
        img = np.float32(cv2.imread(png_fname))
        img = preprocess(img)
        imgs[i] = copy.deepcopy(img)
        #print("\r%d/%d" % (i+1,self.train_size)),
        #sys.stdout.flush()

    self.train_imgs = np.asarray(imgs)
    print ("Time spent to read training data: %.1fs" % (time.time()-t1))

  def standardize(self):
    self.mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"

  def load_predicted_gaze_heatmap(self, train_npz):
    train_npz = np.load(train_npz)
    self.train_GHmap = train_npz['heatmap']
    # npz file from pastK models has pastK-fewer data, so we need to know use value of pastK
    pastK = 3
    self.train_imgs = self.train_imgs[pastK:]
    self.train_lbl = self.train_lbl[pastK:]

  def reshape_heatmap_for_cgl(self, heatmap_shape):
    # predicted human gaze was in 84 x 84, needs to be reshaped for cgl
    #heatmap_shape: output feature map size of the conv layer 
    import cv2
    self.temp = np.zeros((len(self.train_GHmap), heatmap_shape, heatmap_shape))
    for i in range(len(self.train_GHmap)):
        self.temp[i] = cv2.resize(self.train_GHmap[i], (heatmap_shape, heatmap_shape), interpolation=cv2.INTER_AREA)
    self.train_GHmap = self.temp

  def generate_data_for_gaze_prediction(self):
    self.gaze_imgs = [None] * (self.train_size - 3)
    #stack every four frames to make an observation (84,84,4)
    for i in range(3, self.train_size):
        stacked_obs = np.zeros((84, 84, 4))
        stacked_obs[:, :, 0] = self.train_imgs[i-3]
        stacked_obs[:, :, 1] = self.train_imgs[i-2]
        stacked_obs[:, :, 2] = self.train_imgs[i-1]
        stacked_obs[:, :, 3] = self.train_imgs[i]
        self.gaze_imgs[i-3] = copy.deepcopy(stacked_obs)

    self.gaze_imgs = np.asarray(self.gaze_imgs)
    print("Shape of the data for gaze prediction: ", self.gaze_imgs.shape)
    
    
    
'''This file reads a trained gaze prediction network by Zhang et al. 2020, and a data file, then outputs human attention map
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).'''

import tensorflow as tf, numpy as np, tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

def my_softmax(x):
    """Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    return K.activations.softmax(x, axis=-1)

def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    epsilon = 1e-10 # introduce epsilon to avoid log and division by zero error
    y_true = K.backend.clip(y_true, epsilon, 1)
    y_pred = K.backend.clip(y_pred, epsilon, 1)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = [1,2,3])

class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 

    def init_model(self, gaze_model_file):
        # Constants
        self.k = 4
        self.stride = 1
        self.img_shape = 84

        # Constants
        SHAPE = (self.img_shape,self.img_shape,self.k) # height * width * channel
        dropout = 0.0
        ###############################
        # Architecture of the network #
        ###############################
        inputs=L.Input(shape=SHAPE)
        x=inputs 
        
        conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
        x = conv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv2=L.Conv2D(64, (4,4), strides=2, padding='valid')
        x = conv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv3=L.Conv2D(64, (3,3), strides=1, padding='valid')
        x = conv3(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
        x = deconv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
    
        deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
        x = deconv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)         
    
        deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
        x = deconv3(x)
    
        outputs = L.Activation(my_softmax)(x)
        self.model=Model(inputs=inputs, outputs=outputs)
        opt=K.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08)
        self.model.compile(loss=my_kld, optimizer=opt)
        
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
  
    def predict_and_save(self, imgs):
        print("Predicting results...")
        self.preds = self.model.predict(imgs) 
        print("Predicted.")
    
        print("Writing predicted gaze heatmap (train) into the npz file...")
        np.savez_compressed("human_gaze_" + self.game_name, heatmap=self.preds[:,:,:,0])
        print("Done. Output is:")
        print(" %s" % "human_gaze_" + self.game_name + '.npz')


tar_file = './breakout/92_RZ_3504740_Aug-23-11-27-56.tar.bz2'
label_file = './breakout/92_RZ_3504740_Aug-23-11-27-56.txt'
name = 'breakout'
gaze_model = './breakout.hdf5'

d = Dataset(tar_file, label_file)
d.generate_data_for_gaze_prediction()

gp = Human_Gaze_Predictor(name) #game name
gp.init_model(gaze_model) #gaze model .hdf5 file provided in the repo
gp.predict_and_save(d.gaze_imgs)