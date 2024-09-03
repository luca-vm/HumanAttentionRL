'''
Example data loader for Atari-HEAD dataset
This file reads dataset by
Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., ... & Ballard, D. (2020, April). Atari-head: Atari human eye-tracking and demonstration dataset. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6811-6820).
'''

import sys, os, re, threading, time, copy
import numpy as np
import tarfile
import cv2

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
      t1 = time.time()
      print("Reading all training data into memory...")

      # Read action labels and gaze positions from the txt file
      frame_ids, lbls = [], []
      gaze_positions = {}

      with open(label_fname, 'r') as f:
            for line in f:
                if line.startswith("frame_id") or line == "":
                    continue  # skip header or empty lines
                dataline = line.split(',')
                frame_id = dataline[0]
                lbl = dataline[5]
                gaze_pos_str = dataline[6:]  # Extract all values after the label

                # Convert the list of gaze positions to floats
                try:
                    gaze_pos = [float(value) for value in gaze_pos_str if value.strip()]
                except ValueError:
                    gaze_pos = []  # Handle any conversion issues

                if lbl == "null":  # end of file
                    continue

                frame_ids.append(frame_id)
                lbls.append(int(lbl))

                # Initialize the list for this frame_id if it doesn't exist
                if frame_id not in gaze_positions:
                    gaze_positions[frame_id] = []

                # Append the gaze positions to the list for this frame_id
                if gaze_pos:
                    gaze_positions[frame_id].extend(gaze_pos)
                    
              
     
      self.train_lbl = np.asarray(lbls, dtype=np.int32)
      self.gaze_positions = gaze_positions
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
      temp_extract_full_path_dir = temp_extract_dir + png_files[0].split('/')[0]
      
      
      print("Uncompressed PNG tar file into temporary directory: " + temp_extract_full_path_dir)

      print("Reading images...")
      for i in range(self.train_size):
          frame_id = self.frame_ids[i]
          png_fname = temp_extract_full_path_dir + '/' + frame_id + '.png'
          img = np.float32(cv2.imread(png_fname))
          img = preprocess(img)
          imgs[i] = copy.deepcopy(img)

      self.train_imgs = np.asarray(imgs)
      print("Time spent to read training data: %.1fs" % (time.time() - t1))

  def standardize(self):
      self.mean = np.mean(self.train_imgs, axis=(0, 1, 2))
      self.train_imgs -= self.mean  # done in-place --- "x-=mean" is faster than "x=x-mean"


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
    self.gaze_imgs = []
    self.gaze_maps = []
    
    for i in range(self.train_size):
        if i < 3:
            # For the first three frames, create a stacked_obs with repeated frames
            stacked_obs = np.zeros((84, 84, 4))
            for j in range(4):
                stacked_obs[:, :, j] = self.train_imgs[max(0, i-j)]
        else:
            # Regular case for stacking four consecutive frames
            stacked_obs = np.zeros((84, 84, 4))
            stacked_obs[:, :, 0] = self.train_imgs[i-3]
            stacked_obs[:, :, 1] = self.train_imgs[i-2]
            stacked_obs[:, :, 2] = self.train_imgs[i-1]
            stacked_obs[:, :, 3] = self.train_imgs[i]

        self.gaze_imgs.append(copy.deepcopy(stacked_obs))

        # Generate gaze map
        gaze_map = np.zeros((84, 84, 1))  # Add an extra dimension for the channel
        gaze_positions = self.gaze_positions.get(self.frame_ids[i], [])
        for x, y in zip(gaze_positions[::2], gaze_positions[1::2]):
            x = int(x * 84 / 160)
            y = int(y * 84 / 210)
            if 0 <= x < 84 and 0 <= y < 84:
                cv2.circle(gaze_map[:, :, 0], (x, y), 1, 1, -1)  # Draw on the first channel
        self.gaze_maps.append(gaze_map)

    self.gaze_imgs = np.asarray(self.gaze_imgs)
    self.gaze_maps = np.asarray(self.gaze_maps)
    print("Shape of the data for gaze prediction: ", self.gaze_imgs.shape)
    print("Shape of gaze maps: ", self.gaze_maps.shape)



import tensorflow as tf
import numpy as np
import cv2
import sys
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential 

def my_softmax(x):
    # Reshape the input tensor to flatten the spatial dimensions (84x84)
    reshaped_x = tf.reshape(x, (-1, 84 * 84))
    
    # Apply the softmax along the last dimension
    softmaxed_x = tf.nn.softmax(reshaped_x, axis=-1)
    
    # Reshape it back to the original shape (None, 84, 84, 1)
    output = tf.reshape(softmaxed_x, tf.shape(x))
    
    return output

def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    epsilon = 1e-10 # introduce epsilon to avoid log and division by zero error
    y_true = K.backend.cast(K.backend.clip(y_true, epsilon, 1), tf.float32)
    y_pred = K.backend.cast(K.backend.clip(y_pred, epsilon, 1), tf.float32)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis=[1, 2, 3])



def create_saliency_model(input_shape=(84, 84, 4)):
    inputs = L.Input(shape=input_shape)
    dropout = 0.0
    
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
    model=Model(inputs=inputs, outputs=outputs)
    
    print("model created")
    return model


def main():
    # tar_file = './ms_pacman/combined_data.tar.bz2'
    # label_file = './ms_pacman/combined_data.txt'
    
    tar_file = './ms_pacman/52_RZ_2394668_Aug-10-14-52-42.tar.bz2'
    label_file = './ms_pacman/52_RZ_2394668_Aug-10-14-52-42.txt'
    
    
    # Load and preprocess data
    dataset = Dataset(tar_file, label_file)
    dataset.generate_data_for_gaze_prediction()
    


    # Create and compile the model
    model = create_saliency_model()
    # print(model.summary())
    
    
    opt = K.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss=my_kld, optimizer=opt)
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # with tf.GradientTape() as tape:
    #     predictions = model(dataset.gaze_imgs)
    #     loss = my_kld(dataset.gaze_maps, predictions)
    # gradients = tape.gradient(loss, model.trainable_variables)
    # print("Gradients stats: ", [g.numpy().min() for g in gradients], [g.numpy().max() for g in gradients])
    	
    
    BATCH_SIZE = 50
    num_epoch = 50
    model.fit(dataset.gaze_imgs, dataset.gaze_maps, BATCH_SIZE, epochs=num_epoch, shuffle=True, verbose=2)
    model.save("gaze.hdf5")
    


if __name__ == "__main__":
   
    main()

