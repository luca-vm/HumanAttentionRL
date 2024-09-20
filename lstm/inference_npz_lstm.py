import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from load_inf_lstm import *

BATCH_SIZE = 128
num_epoch = 40
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.01
epsilon=1e-08
gaze_weight = 0.7


def my_softmax(x):
    reshaped_x = tf.reshape(x, (-1, 84 * 84))
    softmaxed_x = tf.nn.softmax(reshaped_x, axis=-1)
    output = tf.reshape(softmaxed_x, tf.shape(x))
    return output

def my_kld(y_true, y_pred):
    epsilon = 1e-10
    y_true = K.backend.cast(K.backend.clip(y_true, epsilon, 1), tf.float32)
    y_pred = K.backend.cast(K.backend.clip(y_pred, epsilon, 1), tf.float32)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis=[1, 2, 3])

def duration_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def combined_loss(gaze_weight=gaze_weight, duration_weight= 1 - gaze_weight):
    def loss(y_true, y_pred):
        gaze_true = y_true[:, :84*84]
        duration_true = y_true[:, 84*84:]
        
        gaze_pred = y_pred[:, :84*84]
        duration_pred = y_pred[:, 84*84:]

        gaze_true = tf.reshape(gaze_true, [-1, 84, 84, 1])
        gaze_pred = tf.reshape(gaze_pred, [-1, 84, 84, 1])

        gaze_loss = my_kld(gaze_true, gaze_pred)
        dur_loss = duration_loss(duration_true, duration_pred)
        
        return gaze_weight * gaze_loss + duration_weight * dur_loss
    return loss


class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 

    def init_model(self, gaze_model_file, input_shape=(84, 84, 4), regularization_factor=regularization_factor, dropout= dropout):
        imgs = L.Input(shape=input_shape)
        opt = L.Input(shape=input_shape)
        sal = L.Input(shape=input_shape)
        
        def process_branch(input_tensor, name_prefix):
            # Encoder part (Conv layers)
            x = input_tensor
            for i, (filters, kernel_size, strides) in enumerate([(32, 8, 4), (64, 4, 2), (64, 3, 1)]):
                x = L.Conv2D(filters, kernel_size, strides=strides, padding='valid', 
                            kernel_regularizer=regularizers.l2(regularization_factor),
                            name=f'{name_prefix}_conv{i+1}')(x)
                x = L.Activation('relu')(x)
                x = L.BatchNormalization()(x)
                x = L.Dropout(dropout)(x)
            
            # Flatten spatial dimensions while preserving temporal (channel) information
            shape_before_lstm = tf.keras.backend.int_shape(x)
            x = L.Reshape((-1, shape_before_lstm[-1]))(x)  # Shape: (batch_size, timesteps, features)

            # LSTM layer after encoding
            x = L.LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(regularization_factor))(x)
            
            # Fully connected to reshape LSTM output for decoding
            x = L.Dense(shape_before_lstm[1] * shape_before_lstm[2] * 64, activation='relu')(x)
            x = L.Reshape((shape_before_lstm[1], shape_before_lstm[2], 64))(x)

            # Decoder part (Conv2DTranspose layers)
            for i, (filters, kernel_size, strides) in enumerate([(64, 3, 1), (32, 4, 2), (1, 8, 4)]):
                x = L.Conv2DTranspose(filters, kernel_size, strides=strides, padding='valid',
                                    kernel_regularizer=regularizers.l2(regularization_factor),
                                    name=f'{name_prefix}_deconv{i+1}')(x)
                if i < 2:  # Apply activation, batch norm, and dropout for intermediate layers
                    x = L.Activation('relu')(x)
                    x = L.BatchNormalization()(x)
                    x = L.Dropout(dropout)(x)
            return x
        
        # Process each input branch (imgs, opt, sal)
        x = process_branch(imgs, 'imgs')
        opt_x = process_branch(opt, 'opt')
        sal_x = process_branch(sal, 'sal')
        
        # Gaze map prediction
        gaze_map = L.Average()([x, opt_x, sal_x])  # Average the outputs of the three branches
        gaze_map = L.Activation(my_softmax)(gaze_map)
        gaze_map_flat = L.Flatten()(gaze_map)
        
        # Duration prediction branch
        duration_x = L.Concatenate()([L.Flatten()(x), L.Flatten()(opt_x), L.Flatten()(sal_x)])
        duration_x = L.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
        duration_x = L.Dropout(dropout)(duration_x)
        duration_x = L.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
        duration_x = L.Dropout(dropout)(duration_x)
        duration_pred = L.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
        
        # Combine outputs
        combined_output = L.Concatenate()([gaze_map_flat, duration_pred])

        self.model = Model(inputs=[imgs, opt, sal], outputs=combined_output)
            
        opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
        self.model.compile(loss=combined_loss(gaze_weight), optimizer=opt, metrics=['accuracy'])
        
      
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
  
    def predict_and_save(self, inputs, frame_ids):
        print("Predicting results...")
        self.preds = self.model.predict(inputs, batch_size=32)
        print("Predicted.")
    
        print("Writing predicted gaze heatmap and durations into the npz file...")
        
        # Separate gaze heatmap and duration predictions
        gaze_preds = self.preds[:, :84*84]  # First 84*84 elements are the flattened gaze heatmap
        duration_preds = self.preds[:, 84*84:]  # Remaining elements are duration predictions
        
        # Reshape gaze predictions back to 2D heatmaps
        gaze_heatmaps = gaze_preds.reshape(-1, 84, 84)
        
        np.savez_compressed(f"./{self.game_name}/gaze_lstm.npz", 
                            heatmap=gaze_heatmaps, 
                            durations=duration_preds, 
                            frame_ids=frame_ids)
        print("Done. Output is:")
        print(f" {self.game_name}/gaze_lstm.npz")





# test_tar_file = './ms_pacman/test.tar.bz2'
# test_opt_file = './ms_pacman/test_opt.tar.bz2'
# test_sal_file = './ms_pacman/test_sal.tar.bz2'
# test_label_file = './ms_pacman/test.txt'

test_tar_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35.tar.bz2'
test_opt_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35_opt.tar.bz2'
test_sal_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35_sal.tar.bz2'
test_label_file = './ms_pacman/496_RZ_3560871_Jul-19-13-28-35.txt'


name = 'ms_pacman'
gaze_model = './dummy-3puhh3gz.hdf5'
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