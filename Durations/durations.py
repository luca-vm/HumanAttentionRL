
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
# from load_onlyinit import *
from load_D import *
import time


import os
os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 1
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.01
epsilon=1e-08
gaze_weight = 0.7

# Initialize WandB run
run = wandb.init(
    project="Gaze-Prediction-Final",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": num_epoch,
        "optimizer": "Adadelta",
        "learning_rate": lr,
        "rho": r,
        "dropout": dropout,
        "epsilon": epsilon,
        "buffer-size" : BUFF_SIZE,
        "regularization_factor": regularization_factor,
        "gaze_weight": gaze_weight
    }
)


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
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) * 20

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

def iou_metric(y_true, y_pred):
    # Extract only the gaze map portion (first 84*84 elements)
    gaze_true = y_true[:, :84*84]
    gaze_pred = y_pred[:, :84*84]
    
    # Reshape back to image dimensions
    gaze_true = tf.reshape(gaze_true, [-1, 84, 84, 1])
    gaze_pred = tf.reshape(gaze_pred, [-1, 84, 84, 1])
    
    # For y_true, keep the binary nature
    y_true_mask = tf.cast(gaze_true > 0.3, tf.float32)
    
    # For y_pred, use a dynamic threshold
    threshold = tf.reduce_mean(gaze_pred) + tf.math.reduce_std(gaze_pred)
    y_pred_mask = tf.cast(gaze_pred > threshold, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_mask * y_pred_mask, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_mask, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_mask, axis=[1, 2, 3]) - intersection
    
    # Calculate IOU
    iou = tf.where(union > 0, intersection / union, tf.zeros_like(intersection))
    
    return tf.reduce_mean(iou)

def create_saliency_model(input_shape=(84, 84, 4), regularization_factor=0.01, dropout=0.3):
    imgs = L.Input(shape=input_shape)
    opt = L.Input(shape=input_shape)
    sal = L.Input(shape=input_shape)
    
    def process_branch(input_tensor, name_prefix):
        x = input_tensor
        for i, (filters, kernel_size, strides) in enumerate([(32, 8, 4), (64, 4, 2), (64, 3, 1)]):
            x = L.Conv2D(filters, kernel_size, strides=strides, padding='valid', 
                         kernel_regularizer=regularizers.l2(regularization_factor),
                         name=f'{name_prefix}_conv{i+1}')(x)
            x = L.Activation('relu')(x)
            x = L.BatchNormalization()(x)
            x = L.Dropout(dropout)(x)
        
        for i, (filters, kernel_size, strides) in enumerate([(64, 3, 1), (32, 4, 2), (1, 8, 4)]):
            x = L.Conv2DTranspose(filters, kernel_size, strides=strides, padding='valid',
                                  kernel_regularizer=regularizers.l2(regularization_factor),
                                  name=f'{name_prefix}_deconv{i+1}')(x)
            if i < 2:
                x = L.Activation('relu')(x)
                x = L.BatchNormalization()(x)
                x = L.Dropout(dropout)(x)
        return x
    
    x = process_branch(imgs, 'imgs')
    opt_x = process_branch(opt, 'opt')
    sal_x = process_branch(sal, 'sal')
    
    # Gaze map prediction
    gaze_map = L.Average()([x, opt_x, sal_x])
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
    
    model = Model(inputs=[imgs, opt, sal], outputs=combined_output)
    return model


class TrainingLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        iou = logs.get('iou_metric')
        val_iou = logs.get('val_iou_metric')
        
        if val_loss is not None:
            wandb.log({
                "val_loss": val_loss,
                "train_loss": loss,
                "train_iou": iou,
                "val_iou": val_iou
            })
        
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}, val_loss = {val_loss:.4f}, IOU = {iou:.4f}, val_IOU = {val_iou:.4f}")
        
early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
        

        
def main():

    train_tar_file = './ms_pacman/test.tar.bz2'
    train_opt_file = './ms_pacman/test_opt.tar.bz2'
    train_sal_file = './ms_pacman/test_sal.tar.bz2'
    train_label_file = './ms_pacman/test.txt'
    
    val_tar_file = './ms_pacman/val_test.tar.bz2'
    val_opt_file = './ms_pacman/val_test_opt.tar.bz2'
    val_sal_file = './ms_pacman/val_test_sal.tar.bz2'
    val_label_file = './ms_pacman/val_test.txt'
    
    test_tar_file = './ms_pacman/val_test.tar.bz2'
    test_opt_file = './ms_pacman/val_test_opt.tar.bz2'
    test_sal_file = './ms_pacman/val_test_sal.tar.bz2'
    test_label_file = './ms_pacman/val_test.txt'
    
    
    # train_tar_file = './ms_pacman/train_data/train.tar.bz2'
    # train_opt_file = './ms_pacman/train_data/train_opt.tar.bz2'
    # train_sal_file = './ms_pacman/train_data/train_sal.tar.bz2'
    # train_label_file = './ms_pacman/train_data/train.txt'
    
    # # train_tar_file = './ms_pacman/val_data/val.tar.bz2'
    # # train_opt_file = './ms_pacman/val_data/val_opt.tar.bz2'
    # # train_sal_file = './ms_pacman/val_data/val_sal.tar.bz2'
    # # train_label_file = './ms_pacman/val_data/val.txt'
    
    # val_tar_file = './ms_pacman/val_data/val.tar.bz2'
    # val_opt_file = './ms_pacman/val_data/val_opt.tar.bz2'
    # val_sal_file = './ms_pacman/val_data/val_sal.tar.bz2'
    # val_label_file = './ms_pacman/val_data/val.txt'
    
    
    # test_tar_file = './ms_pacman/test_data/test.tar.bz2'
    # test_opt_file = './ms_pacman/test_data/test_opt.tar.bz2'
    # test_sal_file = './ms_pacman/test_data/test_sal.tar.bz2'
    # test_label_file = './ms_pacman/test_data/test.txt'
    
    
    
    
    t1 = time.time()
    
    train_dataset = Dataset(train_tar_file, train_opt_file, train_sal_file, train_label_file)
    val_dataset = Dataset(val_tar_file, val_opt_file, val_sal_file, val_label_file)
    
    
    

    train_tf_dataset = train_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    val_tf_dataset = val_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    
     # Calculate steps per epoch
    train_steps_per_epoch = train_dataset.get_steps_per_epoch(BATCH_SIZE)
    val_steps = val_dataset.get_steps_per_epoch(BATCH_SIZE)
    
    print(f"Time spent loading and preprocessing: {time.time() - t1:.1f}s")

    # Create and compile the model
    model = create_saliency_model()
    
    opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
    model.compile(loss=combined_loss(gaze_weight), optimizer=opt, metrics=['accuracy', iou_metric])
    print("Compiled")
    
    
   # Initialize callbacks
    callbacks = [TrainingLogger(), WandbMetricsLogger(log_freq=5), early_stopping]
    
    print(f"Starting model training. Steps per epoch: {train_steps_per_epoch}")
    history = model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        epochs=num_epoch,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    
    del train_dataset
    del val_dataset
    
    # Load test data
    test_dataset = Dataset(test_tar_file, test_opt_file, test_sal_file, test_label_file)
    test_tf_dataset = test_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    test_steps = test_dataset.get_steps_per_epoch(BATCH_SIZE)

    test_loss, test_acc, test_iou = model.evaluate(test_tf_dataset, steps=test_steps, verbose=2)
    print(f'Test Loss: {test_loss}, Test IOU: {test_iou}')
    wandb.log({"test_loss": test_loss, "test_iou": test_iou})


    # Finish the WandB run
    wandb.finish()

    run_name = run.name
    # Save the model
    model.save(f"{run_name}.hdf5")

  


if __name__ == "__main__":
   
    main()

