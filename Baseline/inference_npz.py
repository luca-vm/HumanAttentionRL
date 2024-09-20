import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
# from load_onlyinit import *
from load_inf import *
import time

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 3
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.01
epsilon=1e-08


def my_softmax(x):
    # Reshape the input tensor to flatten the spatial dimensions (84x84)
    reshaped_x = tf.reshape(x, (-1, 84 * 84))
    
    # Apply the softmax along the last dimension
    softmaxed_x = tf.nn.softmax(reshaped_x, axis=-1)
    
    # Reshape it back to the original shape (None, 84, 84, 1)
    output = tf.reshape(softmaxed_x, tf.shape(x))
    
    return output


def my_kld(y_true, y_pred):
    """Compute the KL-divergence between two metrics."""
    epsilon = 1e-10
    y_true = tf.clip_by_value(y_true, epsilon, 1.0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    
    # KL divergence calculation based on the provided formula
    kld = tf.reduce_sum(y_true * tf.math.log(epsilon + y_true / (epsilon + y_pred)), axis=[1, 2, 3])
    
    return kld


class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 

    def init_model(self, gaze_model_file, input_shape=(84, 84, 4), regularization_factor=regularization_factor, dropout= dropout):
        imgs = L.Input(shape=input_shape)
    
    
        x=imgs 
        conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv3(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)

        deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)         

        deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv3(x)
        
        #==================== Branch 2 ============================
        opt = L.Input(shape=input_shape)
        
        opt_x=opt 
        opt_conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv1(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv2(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv3(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv1(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)

        opt_deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv2(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)         

        opt_deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv3(opt_x)
        
        
        
        #==================== Branch 3 ============================
        sal = L.Input(shape=input_shape)
        
        sal_x=sal 
        sal_conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv1(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv2(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv3(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv1(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)

        sal_deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv2(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)         

        sal_deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv3(sal_x)
        

        #=================== Avg ==================================
        x = L.Average()([x, opt_x, sal_x])
        outputs = L.Activation(my_softmax)(x)
        self.model=Model(inputs=[imgs, opt, sal], outputs=outputs)
        
        opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
        self.model.compile(loss=my_kld, optimizer=opt, metrics=['accuracy'])
        
      
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
  
    def predict_and_save(self, inputs, frame_ids):
        print("Predicting results...")
        self.preds = self.model.predict(inputs, batch_size=32)  # Use a reasonable batch size for prediction
        print("Predicted.")
    
        print("Writing predicted gaze heatmap into the npz file...")
        np.savez_compressed(f"./{self.game_name}/gaze.npz", heatmap=self.preds[:,:,:,0], frame_ids=frame_ids)
        print("Done. Output is:")
        print(f" {self.game_name}/gaze.npz")





# test_tar_file = './ms_pacman/test_data/test.tar.bz2'
# test_opt_file = './ms_pacman/test_data/test_opt.tar.bz2'
# test_sal_file = './ms_pacman/test_data/test_sal.tar.bz2'
# test_label_file = './ms_pacman/test_data/test.txt'

test_tar_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35.tar.bz2'
test_opt_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35_opt.tar.bz2'
test_sal_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35_sal.tar.bz2'
test_label_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35.txt'


name = 'ms_pacman'
gaze_model = './dummy-j14d7kvz.hdf5'
# file_name = (tar_file.split('.'))[1].split('/')[2])




def run_prediction(test_dataset, gaze_model, name):
    gp = Human_Gaze_Predictor(name)
    gp.init_model(gaze_model)

    test_tf_dataset = test_dataset.get_dataset()

    all_stacked_imgs = []
    all_stacked_opts = []
    all_stacked_sals = []
    all_frame_ids = []

    for (stacked_img, stacked_opt, stacked_sal), _, frame_id in test_tf_dataset:
        all_stacked_imgs.append(stacked_img.numpy())
        all_stacked_opts.append(stacked_opt.numpy())
        all_stacked_sals.append(stacked_sal.numpy())
        all_frame_ids.append(frame_id.numpy().decode())

    inputs = [
        np.array(all_stacked_imgs),
        np.array(all_stacked_opts),
        np.array(all_stacked_sals)
    ]

    gp.predict_and_save(inputs, all_frame_ids)

# Usage
test_dataset = Dataset(test_tar_file, test_opt_file, test_sal_file, test_label_file)
run_prediction(test_dataset, gaze_model, name)