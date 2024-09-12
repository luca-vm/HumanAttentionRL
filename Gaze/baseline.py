
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
# from load_onlyinit import *
from load_baseline import *
import time


# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

BATCH_SIZE = 512
BUFF_SIZE = 40000
num_epoch = 50
lr = 0.75
r = 0.95
dropout = 0.4
regularization_factor=0.001
epsilon=1e-08

# Initialize WandB run
run = wandb.init(
    project="Gaze-Prediction-test",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": num_epoch,
        "optimizer": "Adadelta",
        "learning_rate": lr,
        "rho": r,
        "dropout": dropout,
        "epsilon": epsilon,
        "buffer-size" : BUFF_SIZE,
        "regularization_factor": regularization_factor
    }
)


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





def create_saliency_model(input_shape=(84, 84, 4), regularization_factor=regularization_factor, dropout= dropout):
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
    model=Model(inputs=[imgs, opt, sal], outputs=outputs)
    
    
    print("model created")
    return model


# Define a custom callback for logging
class TrainingLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            wandb.log({"val_loss": val_loss})
        
        print(f"Epoch {epoch + 1}: loss = {loss:.4f} val_loss = {val_loss:.4f}")
        
early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
        

        
def main():

    # train_tar_file = './ms_pacman/test.tar.bz2'
    # train_opt_file = './ms_pacman/test_opt.tar.bz2'
    # train_sal_file = './ms_pacman/test_sal.tar.bz2'
    # train_label_file = './ms_pacman/test.txt'
    
    # val_tar_file = './ms_pacman/val_test.tar.bz2'
    # val_opt_file = './ms_pacman/val_test_opt.tar.bz2'
    # val_sal_file = './ms_pacman/val_test_sal.tar.bz2'
    # val_label_file = './ms_pacman/val_test.txt'
    
    # test_tar_file = './ms_pacman/val_test.tar.bz2'
    # test_opt_file = './ms_pacman/val_test_opt.tar.bz2'
    # test_sal_file = './ms_pacman/val_test_sal.tar.bz2'
    # test_label_file = './ms_pacman/val_test.txt'
    
    
    train_tar_file = './ms_pacman/train_data/train.tar.bz2'
    train_opt_file = './ms_pacman/train_data/train_opt.tar.bz2'
    train_sal_file = './ms_pacman/train_data/train_sal.tar.bz2'
    train_label_file = './ms_pacman/train_data/train.txt'
    
    # train_tar_file = './ms_pacman/val_data/val.tar.bz2'
    # train_opt_file = './ms_pacman/val_data/val_opt.tar.bz2'
    # train_sal_file = './ms_pacman/val_data/val_sal.tar.bz2'
    # train_label_file = './ms_pacman/val_data/val.txt'
    
    val_tar_file = './ms_pacman/val_data/val.tar.bz2'
    val_opt_file = './ms_pacman/val_data/val_opt.tar.bz2'
    val_sal_file = './ms_pacman/val_data/val_sal.tar.bz2'
    val_label_file = './ms_pacman/val_data/val.txt'
    
    
    test_tar_file = './ms_pacman/test_data/test.tar.bz2'
    test_opt_file = './ms_pacman/test_data/test_opt.tar.bz2'
    test_sal_file = './ms_pacman/test_data/test_sal.tar.bz2'
    test_label_file = './ms_pacman/test_data/test.txt'
    
    
    
    
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
    # print(model.summary())
    
    
    
    opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
    model.compile(loss=my_kld, optimizer=opt, metrics=['accuracy'])
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
    
    test_dataset = Dataset(test_tar_file, test_opt_file, test_sal_file, test_label_file)
    test_tf_dataset = test_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)

    # Evaluate on test set
    test_steps = test_dataset.get_steps_per_epoch(BATCH_SIZE)
    test_loss = model.evaluate(test_tf_dataset, steps=test_steps, verbose=2)
    print(f'Test Loss: {test_loss}')
    wandb.log({"test_loss": test_loss})

    print(f"Training metrics: loss = {history.history['loss'][-1]:.4f}, val_loss = {history.history['val_loss'][-1]:.4f}")

    # Finish the WandB run
    wandb.finish()

    run_name = run.name
    # Save the model
    model.save(f"{run_name}.hdf5")


if __name__ == "__main__":
   
    main()

