import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
from keras.layers import Dropout
env = gym.make("CartPole-v0")

state_size = env.observation_space.shape[0]
print (state_size)

action_size = env.action_space.n
print (action_size)

batch_size = 32

n_episodes = 1001

#output_dir = "/content/gdrive/My Drive/"
class DQNAgent:
  def __init__ (self, state_size, action_size,n_layer1=25, n_layer2=25):
    self.state_size = state_size
    self.action_size = action_size
    
    self.memory = deque(maxlen=2000)
    
    self.gamma = 0.95
    self.n_layer1= n_layer1
    self.n_layer2= n_layer2
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01
    
    self.learning_rate = 0.001
    
    self.model = self._build_model()
    
  def _build_model(self):
    
    model = Sequential()
    model.add(Dense(self.n_layer1, input_dim = self.state_size, activation= 'relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(self.n_layer2, activation= 'relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(self.action_size, activation='linear'))
    
    model.compile(loss='mse',optimizer= Adam (lr=self.learning_rate))
    
    return model
    
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    
  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])
  
  def replay(self, batch_size):
    
    minibatch = random.sample(self.memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + (self.gamma * np.amax(self.model.predict(next_state)[0]))
      
      target_f = self.model.predict (state)
      target_f[0][action] = target
      
      self.model.fit(state,target_f,epochs=1,verbose=0)
      
      if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
        
  def load(self, name):
    self.model.load_weights(name)
    print(name)
    
  def save(self, name):
    self.model.save_weights(name)
    
agent = DQNAgent(state_size, action_size)

done = False
end_training = False

for e in range(400):
  if end_training ==True:
    break
  state = env.reset()
  state = np.reshape(state, [1, state_size])
  
  for time in range (5000):
    
    #env.render()
    
    action = agent.act(state)
    
    next_state, reward, done, _ = env.step (action)
    
    reward = reward if not done else -10
    
    next_state = np.reshape(next_state, [1,state_size])
    
    agent.remember(state, action, reward, next_state, done)
    
    state = next_state
    
    
    if time == 199:
      #agent.save(output_dir + "cartpole_weights.hdf5")
      #print ("weights saved")
      end_training = True
      print ("Episode: {}/{}, Score: {}, e: {:.2}".format(e,n_episodes,time,agent.epsilon))
      break
    


    if done:
      print ("Episode: {}/{}, Score: {}, e: {:.2}".format(e,n_episodes,time,agent.epsilon))
      
      break
      
  if len(agent.memory)>batch_size:
    agent.replay(batch_size)
done = False
#agent.load(output_dir + "cartpole_weights.hdf5")
#n_episodes=1000
for e in range(10):
  state = env.reset()
  state = np.reshape(state, [1, state_size])
  
  for time in range (5000):
    
    #env.render()
    
    action = agent.act(state)
    
    next_state, reward, done, _ = env.step (action)
    
    reward = reward if not done else -10
    
    next_state = np.reshape(next_state, [1,state_size])
    
    #agent.remember(state, action, reward, next_state, done)
    
    state = next_state
    
    
    if done:
      print ("Episode: {}/{}, Score: {}, e: {:.2}".format(e,n_episodes,time,agent.epsilon))
      
      break    
    from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
import os
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)



frames = []
for i in range(1):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        frames.append(env.render(mode = 'rgb_array'))
        action = agent.act(state)
        state, r, done, _ = env.step(action)
        state = np.reshape(state, [1, state_size])
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    #agent.stop_episode()
env.render()

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML

plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
HTML(ani.to_jshtml()) 