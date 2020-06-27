#(c) Copywrite 2020 Aaron Krumins

# set async_mode to 'threading', 'eventlet', 'gevent' or 'gevent_uwsgi' to
# force a mode else, the best mode is selected automatically from what's
# installed
async_mode = None

#imports for communication between unreal and gymwrapper
import time
from flask import Flask, render_template
import logging
import logging.handlers
from engineio.async_drivers import eventlet
import socketio
from eventlet.support.dns import dnssec, e164, hash, namedict, tsigkeyring, update, version, zone
import json
import random
import numpy as np
import operator
import sys
import os
from gym import spaces
from random import randint

from threading import Timer


import ast

#imports for open AI gym and a compatible machine library (stable baselines, neuroevolution etc)
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

# Set up a specific logger with our desired output level
logging.disable(sys.maxsize)

sio = socketio.Server(logger=True, async_mode = 'eventlet')
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)


thread = None


observations = "NaN"
UEreward = "0"
UEdone = False
maxactions = 0
obsflag = 0



def check_obs(self):
    global obsflag
    if (obsflag == 1):
        # observations recieved from UE, continue training
        obsflag = 0
    else:
        # observations not recieved yet, check again in a half second
        sio.sleep(.06)
        check_obs(self)
        
   


class UnrealEnvWrap(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a env wrapper that recieves any environmental variables from UE and shapes into a format for OpenAI Gym 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}

  
  def __init__(self, ):
    super(UnrealEnvWrap, self).__init__()
    global maxactions

    # Initialize the agent with a random action
    self.agent_pos = randint(0, maxactions)

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = maxactions
    self.action_space = spaces.Discrete(n_actions)
    # The observation will be all environment variables from UE that agent is tracking
    n_actionsforarray = n_actions - 1
    low = np.array([0,0])
    high = np.array([n_actionsforarray, n_actionsforarray])
    self.observation_space = spaces.Box(low, high, dtype=np.float32)


  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent with a random action
    self.observation_space = [0,0]
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.observation_space])

    
    
 #sending actions to UE and recieving observations in response to those actions  
  def step(self, action):
    global observations
    global UEreward
    global UEdone


    #send actions to UE as they are chosen by the RL algorityhm
    straction = str(action)
    sio.emit('recaction', straction)
    #After sending action, we enter a pause loop until we recieve a response from UE with the observations
    check_obs(self)
    
    #load the observations recieved from UE4
    arrayobs = ast.literal_eval(observations)
    self.observation_space = arrayobs
    done = bool(UEdone)
    reward = float(UEreward)

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.array([self.observation_space]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()


  def close(self):
    os._exit(1)
    pass




@sio.event
def disconnect_request(sid):
    sio.disconnect(sid)
    os._exit(1)

@sio.event
def connect(sid, environ):
    print ("Connected To Unreal Engine")


@sio.event
def disconnect(sid):
    print('Disconnected From Unreal Engine, Exiting MindMaker')
    os._exit(1)
    

    
@sio.on('chat message')
def recieve(sid, data):

    global done
    global reward
    global maxactions
    jsonInput = json.loads(data);
    maxactions = jsonInput['maxactions'];
    trainepisodes = jsonInput['trainepisodes']
    evalepisodes = jsonInput['evalepisodes']
    totalepisodes = trainepisodes + evalepisodes 
   
    env = UnrealEnvWrap()
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1)

    # Train the agent with different algorityhms from stable baselines
    
    #model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./DQN_newobservations/")
    model = DQN(MlpPolicy, env, verbose=1)
    #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./A2C_newobservations/")
    #model = A2C(MlpPolicy, env, verbose=1)
    print("Agent training in process...")
    model.learn(total_timesteps=trainepisodes)

    
    # Test the trained agent, (currently not needed, all testing occurs in Unreal itself)
    env.render(mode='console')
    #env.render()

    obs = env.reset()
    print("Training complete, Starting Evaluation of Trained Model:")
    intaction = 0
    #Begin strategic behvaior
    for step in range(evalepisodes):
      action, _ = model.predict(obs, deterministic=True)
      intaction = action[0]
      print("Action: ", intaction)
      obs, reward, done, info = env.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done)
    
    sio.disconnect(sid)




#recieves observations and reward from Unreal Engine    
@sio.on('sendobs')
def sendobs(sid, obsdata):
    global obsflag
    global observations
    global UEreward
    global UEdone

    obsflag = 1
    jsonInput = json.loads(obsdata);
    observations = jsonInput['observations']     
    UEreward = jsonInput['reward'];
    



#This sets up the server connection, with UE acting as the client in a socketIO relationship, will default to eventlet    
if __name__ == '__main__':
    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        print("1 ran")
        app.run(threaded=True)
        
    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi
        logging.disable(sys.maxsize)
        print("MindMaker running, waiting for Unreal Engine to connect")
        eventlet.wsgi.server(eventlet.listen(('', 3000)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi
        try:
            from geventwebsocket.handler import WebSocketHandler
            websocket = True
            print("3 ran")
        except ImportError:
            websocket = False
        if websocket:
            pywsgi.WSGIServer(('', 3000), app, log = None,
                              handler_class=WebSocketHandler).serve_forever()
            print("4 ran")
            log = logging.getLogger('werkzeug')
            log.disabled = True
            app.logger.disabled = True
        else:
            pywsgi.WSGIServer(('', 3000), app).serve_forever()
            print("5 ran")
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :5000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)
        
   
    

