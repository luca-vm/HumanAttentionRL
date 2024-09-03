# import gymnasium as gym
import gym
import tensorflow as tf
import numpy as np
import copy
import cv2
from human_gaze_class import Human_Gaze_Predictor
import matplotlib.pyplot as plt


# Define your custom softmax and KLD functions
def my_softmax(x):
    return tf.keras.activations.softmax(x, axis=-1)

def my_kld(y_true, y_pred):
    epsilon = 1e-10
    y_true = tf.clip_by_value(y_true, epsilon, 1)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=[1, 2, 3])

# Load the trained AGIL model
model_path = r"./model.hdf5"
model = tf.keras.models.load_model(model_path, custom_objects={'my_softmax': my_softmax, 'my_kld': my_kld})

gp = Human_Gaze_Predictor('breakout') #game name
gp.init_model('./breakout.hdf5') #gaze model .hdf5 file provided in the repo

# Inspect model input and output specifications
# print("Model Inputs:", model.inputs)
# print("Model Input Shapes:", [input.shape for input in model.inputs])
# print("Model Outputs:", model.outputs)
# print("Model Output Shapes:", [output.shape for output in model.outputs])

# Initialize the Ms. Pacman environment with render_mode='human'
env = gym.make("ALE/Breakout-v5", render_mode='human')
obs = env.reset()

# def preprocess(image):
#     """Warp frames to 84x84 as done in the Nature paper and later work."""
#     width = 84
#     height = 84
#     frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
#     return frame / 255.0

def updateImages(images, new_image):
    for i in range(3):
        images[i] = images[i + 1]
        
    images[3] = new_image
    
    return images
    
    


def proccess(img, i, images):

    # Regular case for stacking four consecutive frames
    stacked_obs = np.zeros((84, 84, 4))
    stacked_obs[:, :, 0] = images[i-3]
    stacked_obs[:, :, 1] = images[i-2]
    stacked_obs[:, :, 2] = images[i-1]
    stacked_obs[:, :, 3] = images[i]

    img = copy.deepcopy(stacked_obs)

    img = np.asarray(img)

    return img

def preprocess_observation(observation):
    if isinstance(observation, tuple):
        observation = np.array(observation[0])
    else:
        observation = np.array(observation)
    
    if len(observation.shape) == 3:
        # observation = tf.image.rgb_to_grayscale(observation)
        # observation = tf.image.resize(observation, [84, 84])
        # observation = tf.cast(observation, tf.float32) / 255.0
        # observation = tf.expand_dims(observation, axis=0)
        width = 84
        height = 84
        frame = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame / 255.0
    else:
        raise ValueError("Observation has an unexpected shape.")
    

def select_action(model, observation, frame, images):
    curr_image = preprocess_observation(observation)
    
    if frame <= 4:
        for i in range(frame - 1, 4):
            images[i] = curr_image 
    else:
        images = updateImages(images, curr_image)
    
    input_image = proccess(curr_image, frame, images)
    
    print("Processed Input Image Shape:", input_image.shape)
    

    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension, shape becomes (1, 84, 84, 1)
    print("Expanded Input Image Shape:", input_image.shape)
    
    gaze_map = gp.model.predict(input_image)
    gaze_map = np.squeeze(gaze_map, axis=0)  # Shape becomes (84, 84, 1)
    
    
    plt.imshow(gaze_map)
    plt.show()
    
    print("Gaze Map Shape:", gaze_map.shape)
    print("Frame Number:", frame)
    
    # Combine the inputs into a list
    inputs = [input_image, gaze_map]

    # Predict and suppress the progress bar
    outputs = model.predict(inputs, verbose=0)
    
    print(outputs[0])
    
    # Extract action probabilities (assuming it's the second output)
    action_probs = outputs[0]  # Adjust index based on your model's output order
    
    print("Action Probabilities Shape:", action_probs.shape)
    
    # Assuming action_probs should be a 2D array [batch_size, num_actions]
    if action_probs.ndim == 2:
        action = np.argmax(action_probs[0])  # Take the first batch element
    else:
        raise ValueError("Unexpected shape for action probabilities.")
    
    return action




for episode in range(10):
    obs = env.reset()
    total_reward = 0
    done = False
    frame = 0
    images = [None, None, None, None]
    while not done:
        frame +=1
        env.render()
        action, images = select_action(model, obs, frame, images)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()