import tensorflow as tf

import numpy as np
import base64, io, time, gym
import IPython, functools
import matplotlib.pyplot as plt
from tqdm import tqdm

import mitdeeplearning as mdl


env = gym.make("Pong-v0", frameskip=5)
env.seed(1); # for reproducibility


### Define the agent's action function ###

# Function that takes observations as input, executes a forward pass through model,
#   and outputs a sampled action.
# Arguments:
#   model: the network that defines our agent
#   observation: observation which is fed as input to the model
# Returns:
#   action: choice of agent action
def choose_action(model, observation):
    # add batch dimension to the observation
    observation = np.expand_dims(observation, axis=0)

    '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''
    logits = model.predict(observation)  # TODO
    # logits = model.predict('''TODO''')

    # pass the log probabilities through a softmax to compute true probabilities
    prob_weights = tf.nn.softmax(logits).numpy()

    '''TODO: randomly sample from the prob_weights to pick an action.
    Hint: carefully consider the dimensionality of the input probabilities (vector) and the output action (scalar)'''
    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]  # TODO
    # action = np.random.choice('''TODO''', size=1, p=''''TODO''')['''TODO''']

    return action



### Agent Memory ###

class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        '''TODO: update the list of actions with new action'''
        self.actions.append(new_action)  # TODO
        # ['''TODO''']
        '''TODO: update the list of rewards with new reward'''
        self.rewards.append(new_reward)  # TODO
        # ['''TODO''']


memory = Memory()



print("Environment has observation space =", env.observation_space)

n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)

# Helper function that normalizes an np.array x
def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x.astype(np.float32)


### Loss function ###

# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards):
    '''TODO: complete the function call to compute the negative log probabilities'''
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)  # TODO
    # neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits='''TODO''', labels='''TODO''')

    '''TODO: scale the negative log probability by the rewards'''
    loss = tf.reduce_mean(neg_logprob * rewards)  # TODO
    # loss = tf.reduce_mean('''TODO''')
    return loss


### Training step (forward and backpropagation) ###

def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      logits = model(observations)

      '''TODO: call the compute_loss function to compute the loss'''
      loss = compute_loss(logits, actions, discounted_rewards) # TODO
      # loss = compute_loss('''TODO''', '''TODO''', '''TODO''')

  '''TODO: run backpropagation to minimize the loss using the tape.gradient method'''
  grads = tape.gradient(loss, model.trainable_variables) # TODO
  # grads = tape.gradient('''TODO''', model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


### Define the Pong agent ###

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense


# Defines a CNN for the Pong agent
def create_pong_model():
    model = tf.keras.models.Sequential([
        # Convolutional layers
        # First, 16 7x7 filters and 4x4 stride
        Conv2D(filters=16, kernel_size=7, strides=4),

        # TODO: define convolutional layers with 32 5x5 filters and 2x2 stride
        Conv2D(filters=32, kernel_size=5, strides=2),  # TODO
        # Conv2D('''TODO'''),

        # TODO: define convolutional layers with 48 3x3 filters and 2x2 stride
        Conv2D(filters=48, kernel_size=3, strides=2),  # TODO
        # Conv2D('''TODO'''),

        Flatten(),

        # Fully connected layer and output
        Dense(units=64, activation='relu'),
        # TODO: define the output dimension of the last Dense layer.
        # Pay attention to the space the agent needs to act in
        Dense(units=n_actions, activation=None)  # TODO
        # Dense('''TODO''')

    ])
    return model


pong_model = create_pong_model()


### Pong reward function ###

# Compute normalized, discounted rewards for Pong (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor. Note increase to 0.99 -- rate of depreciation will be slower.
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # NEW: Reset the sum if the reward is not 0 (the game has ended!)
        if rewards[t] != 0:
            R = 0
        # update the total discounted reward as before
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)

observation = env.reset()
for i in range(30):
  observation, _,_,_ = env.step(0)
observation_pp = mdl.lab3.preprocess_pong(observation)

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.imshow(observation); ax.grid(False);
ax2.imshow(np.squeeze(observation_pp)); ax2.grid(False); plt.title('Preprocessed Observation');

### Training Pong ###

# Hyperparameters
learning_rate = 1e-4
MAX_ITERS = 10000  # increase the maximum number of episodes, since Pong is more complex!

# Model and optimizer
pong_model = create_pong_model()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# plotting
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=5, xlabel='Iterations', ylabel='Rewards')




memory = Memory()

for i_episode in range(MAX_ITERS):

    plotter.plot(smoothed_reward.get())

    # Restart the environment
    observation = env.reset()
    previous_frame = mdl.lab3.preprocess_pong(observation)

    while True:
        # Pre-process image
        current_frame = mdl.lab3.preprocess_pong(observation)

        '''TODO: determine the observation change
        Hint: this is the difference between the past two frames'''
        obs_change = current_frame - previous_frame  # TODO
        # obs_change = # TODO

        '''TODO: choose an action for the pong model, using the frame difference, and evaluate'''
        action = choose_action(pong_model, obs_change)  # TODO
        # action = # TODO
        # Take the chosen action
        next_observation, reward, done, info = env.step(action)

        '''TODO: save the observed frame difference, the action that was taken, and the resulting reward!'''
        memory.add_to_memory(obs_change, action, reward)  # TODO

        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            smoothed_reward.append(total_reward)

            # begin training
            train_step(pong_model,
                       optimizer,
                       observations=np.stack(memory.observations, 0),
                       actions=np.array(memory.actions),
                       discounted_rewards=discount_rewards(memory.rewards))

            memory.clear()
            break

        observation = next_observation
        previous_frame = current_frame

saved_pong = mdl.lab3.save_video_of_model(
    pong_model, "Pong-v0", obs_diff=True,
    pp_fn=mdl.lab3.preprocess_pong)
mdl.lab3.play_video(saved_pong)