#(c) Copywrite 2020 Aaron Krumins


#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#    along with this program.


#imports for communication between unreal and mindmaker server
import time
import socketio
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
import math

#imports for open AI gym and a compatible machine library (stable baselines, neuroevolution etc)
import gym
import tensorflow as tf
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.deepq.policies import LnMlpPolicy as DqnLnMlpPolicy
from stable_baselines.deepq.policies import CnnPolicy as DqnCnnPolicy	
from stable_baselines.deepq.policies import LnCnnPolicy as DqnLnCnnPolicy	

from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy as SacLnMlpPolicy
from stable_baselines.sac.policies import CnnPolicy as SacCnnPolicy
from stable_baselines.sac.policies import LnCnnPolicy as SacLnCnnPolicy


from stable_baselines.td3.policies import MlpPolicy as Td3MlpPolicy
from stable_baselines.td3.policies import LnMlpPolicy as Td3LnMlpPolicy
from stable_baselines.td3.policies import CnnPolicy as Td3CnnPolicy
from stable_baselines.td3.policies import LnCnnPolicy as Td3LnCnnPolicy



from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR, ACER, SAC, TD3 
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy

#Start the mindmaker client instance 

sio = socketio.Client()



#Define global variables

observations = "NaN"
UEreward = "0"
UEdone = False
maxactions = 0
obsflag = 0
inf = math.inf
actionspace = "Nan"
observationspace = "Nan"
results = os.getenv('APPDATA')       


# Function to check if observations have been recieved from unreal engine environment, if not stay in holding pattern

def check_obs(self):
    global obsflag
    if (obsflag == 1):
        # observations recieved from UE, continue training
        obsflag = 0
    else:
        # observations not recieved yet, check again in a half second
        sio.sleep(.06)
        check_obs(self)
        
   

# Define our custom wrapper from the unreal environment, variables passed in from the UE editor
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
    global conactionspace
    global disactionspace
    #global minaction
    #global maxaction
    global actionspace
    global observationspace

    print (conactionspace)
    

    print (actionspace)
    if conactionspace == True:
        print("continous action space")
        actionspace = "spaces.Box(" + actionspace + ")"
        observationspace = "spaces.Box(" + observationspace + ")"
        self.action_space = eval(actionspace) 
        self.agent_pos = randint(0, 100)   
        self.observation_space = eval(observationspace)
    elif disactionspace == True:        
        
        print("discrete action space")
        actionspace = int(actionspace)
        # Initialize the agent with a random action       
        self.agent_pos = randint(0, actionspace)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = actionspace
        self.action_space = spaces.Discrete(n_actions)
        observationspace = "spaces.Box(" + observationspace + ")"
        # The observation will be all environment variables from UE that agent is tracking
        n_actionsforarray = n_actions - 1
        self.observation_space = eval(observationspace)
    else:
        logmessages = "No action space type selected"
        sio.emit('messages', logmessages)

  
  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    
    self.observation_space = [0]
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.observation_space])

    
    
 #sending actions to UE and recieving observations in response to those actions  
  def step(self, action):
    global observations
    global UEreward
    global UEdone


    #send actions to UE as they are chosen by the algorityhm
    straction = str(action)
    print("action:",straction)
    sio.emit('recaction', straction)
    #After sending action, we enter a pause loop until we recieve a response from UE with the observations
    check_obs(self)
    
    #load the observations recieved from UE4
    arrayobs = ast.literal_eval(observations)
    self.observation_space = arrayobs
    print(arrayobs)
    done = bool(UEdone)
    reward = float(UEreward)
    print("reward", reward)
    print(UEdone)
    if done == True:
        print("Im rrestarting now how fun")
        reward = 0
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
def disconnect_request():
    sio.disconnect()
    os._exit(1)

@sio.event
def connect():
    print ("Connected To MindMaker Server, sending connection request to Unreal Engine")
    request = "connectclient"
    sio.emit('connectreq', request)
    

@sio.event
def disconnect():
    sio.disconnect()
    print('Disconnected From Unreal Engine, Exiting MindMaker')
    os._exit(1)
    

    
@sio.on('initclient')
def recieve(data):
   
    global UEdone
    global reward
    global maxactions
    global conactionspace
    global disactionspace
    #global minaction
    #global maxaction
    global actionspace
    global observationspace
    jsonInput = json.loads(data);
    actionspace = jsonInput['actionspace'];
    observationspace = jsonInput['observationspace'];
    #minaction = jsonInput['minaction'];
    #maxaction = jsonInput['maxaction'];
    #maxactions = jsonInput['maxactions'];
    trainepisodes = jsonInput['trainepisodes']
    evalepisodes = jsonInput['evalepisodes']
    loadmodel = jsonInput['loadmodel']
    savemodel = jsonInput['savemodel']
    modelname = jsonInput['modelname']
    algselected = jsonInput['algselected']
    usecustomparams = jsonInput['customparams']
    a2cparams = jsonInput['a2cparams']
    acerparams = jsonInput['acerparams']
    acktrparams = jsonInput['acktrparams']
    dqnparams = jsonInput['dqnparams']
    # ddpgparams = jsonInput['ddpgparams']
    # ppo1params = jsonInput['ppo1params']
    ppo2params = jsonInput['ppo2params']
    sacparams = jsonInput['sacparams']
    td3params = jsonInput['td3params']
    # trpoparams = jsonInput['trpoparams']
    conactionspace = jsonInput['conactionspace']
    disactionspace = jsonInput['disactionspace']
    totalepisodes = trainepisodes + evalepisodes 
    UEdone = jsonInput['done']
    env = UnrealEnvWrap()
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1)
    print("save model value:", savemodel)
    print("load model value:", loadmodel)
    
    path = results + "\\" + modelname
    
    
    if loadmodel == 'true':
    # Load the trained agent
        if algselected == 'DQN':
            model = DQN.load(path)
        elif algselected == 'A2C':
            model = A2C.load(path)
        elif algselected == 'ACER':
            model = ACER.load(path)
        elif algselected == 'ACKTR':
            model = ACKTR.load(path)
        elif algselected == 'DDPG (Requires Microsoft OpenMPI)':
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            model = DDPG.load(path)
        elif algselected == 'PPO1 (Requires Microsoft OpenMPI)':
            from stable_baselines import PPO1
            model = PPO1.load(path)
        elif algselected == 'PPO2':
            model = PPO2.load(path)
        elif algselected == 'SAC':
            model = SAC.load(path)
        elif algselected == 'TD3':
            model = TD3.load(path)
        elif algselected == 'TRPO (Requires Microsoft OpenMPI)':
            from stable_baselines import TRPO
            model = TRPO.load(path)
        
        print("Loading the Trained Agent")
        logmessages = "Loading the Trained Agent"
        sio.emit('messages', logmessages)
        obs = env.reset()
        intaction = 0
        #Begin strategic behvaior
        for step in range(evalepisodes):
          action, _ = model.predict(obs, deterministic=True)
          intaction = action[0]
          print("Action: ", intaction)
          obs, reward, done, info = env.step(action)
          print('obs=', obs, 'reward=', reward, 'done=', done)
    

    else:
                # Train the agent with different algorityhms from stable baselines
        
        #model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./DQN_newobservations/")
        print("alg selected:", algselected)
        print("use custom:", usecustomparams)
        
        if (algselected == 'DQN') and (usecustomparams == 'true'):
            gammaval = dqnparams["gamma"]
            policyval = dqnparams["policy"]
            act_funcval = dqnparams["act_func"]
            learning_rateval = dqnparams["learning_rate"]
            verboseval = dqnparams["verbose"]
            tensorboard_logval = dqnparams["tensorboard_log"]
            _init_setup_modelval = dqnparams["_init_setup_model"]
            full_tensorboard_logval = dqnparams["full_tensorboard_log"]
            seedval = dqnparams["seed"]
            n_cpu_tf_sessval = dqnparams["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]                   
            buffer_sizeval = dqnparams["buffer_size"]          
            exploration_fractionval = dqnparams["exploration_fraction"]
            exploration_final_epsval = dqnparams["exploration_final_eps"]
            exploration_initial_epsval = dqnparams["exploration_initial_eps"]
            batch_sizeval = dqnparams["batch_size"]
            train_freqval = dqnparams["train_freq"]
            double_qval = dqnparams["double_q"]           
            learning_startsval = dqnparams["learning_starts"]                  
            target_network_update_freqval = dqnparams["target_network_update_freq"]
            prioritized_replayval = dqnparams["prioritized_replay"]
            prioritized_replay_alphaval = dqnparams["prioritized_replay_alpha"]
            prioritized_replay_beta0val = dqnparams["prioritized_replay_beta0"]
            prioritized_replay_beta_itersval = dqnparams["prioritized_replay_beta_iters"]
            prioritized_replay_epsval = dqnparams["prioritized_replay_eps"]
            param_noiseval = dqnparams["param_noise"]

            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
                print("tanh")
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
                print("relu")
            elif act_funcval == 'tf.nn.leaky_relu' :
                print("leaky_relu")
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))            
            print(policyval)
            policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = DQN(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = ast.literal_eval(buffer_sizeval), exploration_fraction = exploration_fractionval, exploration_final_eps = exploration_final_epsval, exploration_initial_eps = exploration_initial_epsval, batch_size = batch_sizeval, train_freq = train_freqval, double_q = double_qval, learning_starts = learning_startsval, target_network_update_freq = target_network_update_freqval, prioritized_replay = prioritized_replayval, prioritized_replay_alpha = prioritized_replay_alphaval, prioritized_replay_beta0 = prioritized_replay_beta0val, prioritized_replay_beta_iters = ast.literal_eval(prioritized_replay_beta_itersval), prioritized_replay_eps = prioritized_replay_epsval, param_noise = ast.literal_eval(param_noiseval) )
            
            #model = DQN(DqnMlpPolicy, env,  gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000, target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None )
            
            print("Custom DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
            
        elif algselected == 'DQN':
            model = DQN(DqnMlpPolicy, env, verbose=1)
            print("DQN training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif (algselected == 'A2C') and (usecustomparams == 'true') :
            policyval = a2cparams["policy"]
            act_funcval = a2cparams["act_func"]
            gammaval = a2cparams["gamma"]
            n_stepsval = a2cparams["n_steps"]
            vf_coefval = a2cparams["vf_coef"]
            ent_coefval = a2cparams["ent_coef"]
            max_grad_normval = a2cparams["max_grad_norm"]
            learning_rateval = a2cparams["learning_rate"]
            alphaval = a2cparams["alpha"]
            epsilonval = a2cparams["epsilon"]
            lr_scheduleval = a2cparams["lr_schedule"]
            verboseval = a2cparams["verbose"]
            tensorboard_logval = a2cparams["tensorboard_log"]
            _init_setup_modelval = a2cparams["_init_setup_model"]
            full_tensorboard_logval = a2cparams["full_tensorboard_log"]
            seedval = a2cparams["seed"]
            n_cpu_tf_sessval = a2cparams["n_cpu_tf_sess"]
            
            network_archval = a2cparams["network_arch"]
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))
            print(policyval)
            model = A2C(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, alpha = alphaval, epsilon = epsilonval, lr_schedule = lr_scheduleval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval))

            print("Custom A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'A2C':
            model = A2C(MlpPolicy, env, verbose=1)
            print("A2C training in process...")
            model.learn(total_timesteps=trainepisodes)
            
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                
        elif (algselected == 'ACER') and (usecustomparams == 'true'):
            gammaval = acerparams["gamma"]
            policyval = acerparams["policy"]
            act_funcval = acerparams["act_func"]
            n_stepsval = acerparams["n_steps"]
            ent_coefval = acerparams["ent_coef"]
            max_grad_normval = acerparams["max_grad_norm"]
            learning_rateval = acerparams["learning_rate"]
            alphaval = acerparams["alpha"]
            lr_scheduleval = acerparams["lr_schedule"]
            verboseval = acerparams["verbose"]
            num_procsval = acerparams["num_procs"]
            tensorboard_logval = acerparams["tensorboard_log"]
            _init_setup_modelval = acerparams["_init_setup_model"]
            full_tensorboard_logval = acerparams["full_tensorboard_log"]
            seedval = acerparams["seed"]
            n_cpu_tf_sessval = acerparams["n_cpu_tf_sess"]
            network_archval = acerparams["network_arch"]
            q_coefval = acerparams["q_coef"]
            rprop_alphaval = acerparams["rprop_alpha"]
            rprop_epsilonval = acerparams["rprop_epsilon"]
            buffer_sizeval = acerparams["buffer_size"]
            replay_ratioval = acerparams["replay_ratio"]
            replay_startval = acerparams["replay_start"]
            correction_termval = acerparams["correction_term"]
            trust_regionval = acerparams["trust_region"]
            trust_regionval = acerparams["trust_region"]
            deltaval = acerparams["delta"]
            
            print("polic val:", policyval)
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))
            
            
            model = ACER(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, alpha = alphaval, lr_schedule = lr_scheduleval, verbose = verboseval, num_procs = ast.literal_eval(num_procsval), tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), q_coef = q_coefval, rprop_alpha = rprop_alphaval, rprop_epsilon = rprop_epsilonval, buffer_size = buffer_sizeval, replay_ratio = replay_ratioval, replay_start = replay_startval, correction_term = float(correction_termval),trust_region = trust_regionval, delta = deltaval )

            print("Custom ACER training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'ACER':
            model = ACER(MlpPolicy, env, verbose=1)
            print("ACER training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif (algselected == 'ACKTR') and (usecustomparams == 'true'):
            gammaval = acktrparams["gamma"]
            policyval = acktrparams["policy"]
            act_funcval = acktrparams["act_func"]
            n_stepsval = acktrparams["n_steps"]
            ent_coefval = acktrparams["ent_coef"]
            max_grad_normval = acktrparams["max_grad_norm"]
            learning_rateval = acktrparams["learning_rate"]
            lr_scheduleval = acktrparams["lr_schedule"]
            verboseval = acktrparams["verbose"]
            tensorboard_logval = acktrparams["tensorboard_log"]
            _init_setup_modelval = acktrparams["_init_setup_model"]
            full_tensorboard_logval = acktrparams["full_tensorboard_log"]
            seedval = acktrparams["seed"]
            n_cpu_tf_sessval = acktrparams["n_cpu_tf_sess"]
            network_archval = acktrparams["network_arch"]          
            nprocsval = acktrparams["nprocs"]
            vf_coefval = acktrparams["vf_coef"]
            vf_fisher_coefval = acktrparams["vf_fisher_coef"]
            kfac_clipval = acktrparams["kfac_clip"]
            async_eigen_decompval = acktrparams["async_eigen_decomp"]
            kfac_updateval = acktrparams["kfac_update"]
            gae_lambdaval = acktrparams["gae_lambda"]
            #policy_kwargsval = dict(net_arch = ast.literal_eval(network_archval))
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, net_arch = ast.literal_eval(network_archval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, net_arch = ast.literal_eval(network_archval))            

            model = ACKTR(policyval, env, gamma = gammaval, n_steps=n_stepsval, ent_coef= ent_coefval, max_grad_norm = max_grad_normval, learning_rate = learning_rateval, lr_schedule = lr_scheduleval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), nprocs = ast.literal_eval(nprocsval), vf_coef = vf_coefval, vf_fisher_coef = vf_fisher_coefval, kfac_clip = kfac_clipval, async_eigen_decomp = async_eigen_decompval, kfac_update = kfac_updateval, gae_lambda = ast.literal_eval(gae_lambdaval) )

            print("Custom ACKTR training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        
        elif algselected == 'ACKTR':
            model = ACKTR(MlpPolicy, env, verbose=1)
            print("ACKTR training in process...")
            model.learn(total_timesteps=trainepisodes)
        elif (algselected == 'DDPG (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            gammaval = ddpgparams["gamma"]
            policyval = ddpgparams["policy"]
            act_funcval = ddpgparams["act_func"]
            #learning_rateval = ddpgparams["learning_rate"]
            verboseval = ddpgparams["verbose"]
            tensorboard_logval = ddpgparams["tensorboard_log"]
            _init_setup_modelval = ddpgparams["_init_setup_model"]
            full_tensorboard_logval = ddpgparams["full_tensorboard_log"]
            seedval = ddpgparams["seed"]
            n_cpu_tf_sessval = ddpgparams["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]         
            buffer_sizeval = ddpgparams["buffer_size"]
            eval_envval = ddpgparams["eval_env"]
            nb_train_stepsval = ddpgparams["nb_train_steps"]
            nb_rollout_stepsval = ddpgparams["nb_rollout_steps"]
            #batch_sizeval = ddpgparams["batch_size"]
            nb_eval_stepsval = ddpgparams["nb_eval_steps"]
            action_noiseval = ddpgparams["action_noise"]
            param_noise_adaption_intervalval = ddpgparams["param_noise_adaption_interval"]
            tauval = ddpgparams["tau"]
            normalize_returnsval = ddpgparams["normalize_returns"]
            critic_l2_regval = ddpgparams["critic_l2_reg"]
            enable_popartval = ddpgparams["enable_popart"]
            normalize_observationsval = ddpgparams["normalize_observations"]
            observation_rangeval = ddpgparams["observation_range"]
            #return_rangeval = ddpgparams["return_range"]
            param_noiseval = ddpgparams["param_noise"]
            actor_lrval = ddpgparams["actor_lr"]    
            critic_lrval = ddpgparams["critic_lr"]    
            clip_normval = ddpgparams["clip_norm"]    
            reward_scaleval = ddpgparams["reward_scale"]
            renderval = ddpgparams["render"] 
            render_evalval = ddpgparams["render_eval"]       
            memory_limitval = ddpgparams["memory_limit"]   
            memory_policyval = ddpgparams["memory_policy"]   
            random_explorationval = ddpgparams["random_exploration"]   
            
            random_explorationval = float(random_explorationval)
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))         
                
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = DDPG(eval(policyval), env, gamma = gammaval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = ast.literal_eval(buffer_sizeval), eval_env = ast.literal_eval(eval_envval), nb_train_steps = nb_train_stepsval, nb_rollout_steps = nb_rollout_stepsval, nb_eval_steps = nb_eval_stepsval, tau = tauval, param_noise_adaption_interval = param_noise_adaption_intervalval, normalize_returns = normalize_returnsval, critic_l2_reg = ast.literal_eval(critic_l2_regval), enable_popart = enable_popartval, normalize_observations = normalize_observationsval, observation_range = ast.literal_eval(observation_rangeval), return_range = (-inf, inf), param_noise = ast.literal_eval(param_noiseval), actor_lr = actor_lrval, critic_lr = critic_lrval, clip_norm = ast.literal_eval(clip_normval), reward_scale = reward_scaleval, render = ast.literal_eval(renderval), render_eval = ast.literal_eval(render_evalval), memory_limit = ast.literal_eval(memory_limitval), memory_policy = ast.literal_eval(memory_policyval), random_exploration = random_explorationval, action_noise = ast.literal_eval(action_noiseval) )
            
            #model = DDPG(DdpgMlpPolicy,env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50, nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None, normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False, observation_range=(-5.0, 5.0), critic_l2_reg=0.0, return_range=(-inf, inf), actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1) 

            print("Custom DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        elif algselected == 'DDPG (Requires Microsoft OpenMPI)':
            from stable_baselines.ddpg.policies import MlpPolicy as DdpgMlpPolicy
            from stable_baselines.ddpg.policies import LnMlpPolicy as DdpgLnMlpPolicy
            from stable_baselines.ddpg.policies import CnnPolicy as DdpgCnnPolicy
            from stable_baselines.ddpg.policies import LnCnnPolicy as DdpgLnCnnPolicy
            from stable_baselines import DDPG
            print("DDPG requires Microsoft Open MPI be installed on your system")
            # the noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = DDPG(DdpgMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
            print("DDPG training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("DDPG training complete")
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
        
        elif (algselected == 'PPO1 (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines.ppo1 import PPO1
            from stable_baselines import PPO1
            gammaval = ppo1params["gamma"]
            act_funcval = ppo1params["act_func"]
            policyval = ppo1params["policy"]


            timesteps_per_actorbatchval = ppo1params["timesteps_per_actorbatch"]

            verboseval = ppo1params["verbose"]
            tensorboard_logval = ppo1params["tensorboard_log"]
            _init_setup_modelval = ppo1params["_init_setup_model"]
            full_tensorboard_logval = ppo1params["full_tensorboard_log"]
            seedval = ppo1params["seed"]
            n_cpu_tf_sessval = ppo1params["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]         
            
            clip_paramval = ppo1params["clip_param"]
            
            entcoeffval = ppo1params["entcoeff"]
            optim_epochsval = ppo1params["optim_epochs"]
            optim_stepsizeval = ppo1params["optim_stepsize"]
            optim_batchsizeval = ppo1params["optim_batchsize"]
            lamval = ppo1params["lam"]
            adam_epsilonval = ppo1params["adam_epsilon"]
            
            scheduleval = ppo1params["schedule"]
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = PPO1(policyval, env, gamma = gammaval, timesteps_per_actorbatch = timesteps_per_actorbatchval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), clip_param = clip_paramval, entcoeff = entcoeffval, optim_epochs = optim_epochsval, optim_stepsize = optim_stepsizeval, optim_batchsize = optim_batchsizeval, lam = lamval, schedule = scheduleval, adam_epsilon = adam_epsilonval )
            

            print("Custom PPO1 training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO1 training complete")
            if savemodel == 'true':
                # Save the agent
                path = results + "\\" + modelname
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
               
        
        elif algselected == 'PPO1 (Requires Microsoft OpenMPI)':
            from stable_baselines.ppo1 import PPO1
            from stable_baselines import PPO1
            model = PPO1(MlpPolicy, env, verbose=1)
            print("PPO1 training in process...")
            model.learn(total_timesteps=trainepisodes)
            print("PPO1 training complete")  
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)

        elif (algselected == 'PPO2') and (usecustomparams == 'true'):
            gammaval = ppo2params["gamma"]
            policyval = ppo2params["policy"]
            act_funcval = ppo2params["act_func"]
            n_stepsval = ppo2params["n_steps"]
            verboseval = ppo2params["verbose"]
            tensorboard_logval = ppo2params["tensorboard_log"]
            _init_setup_modelval = ppo2params["_init_setup_model"]
            full_tensorboard_logval = ppo2params["full_tensorboard_log"]
            seedval = ppo2params["seed"]
            n_cpu_tf_sessval = ppo2params["n_cpu_tf_sess"]
            layersval = dqnparams["layers"]         
            ent_coefval = ppo2params["ent_coef"]
            learning_rateval = ppo2params["learning_rate"]
            vf_coefval = ppo2params["vf_coef"]
            nminibatchesval = ppo2params["nminibatches"]
            noptepochsval = ppo2params["noptepochs"]
            lamval = ppo2params["lam"]
            cliprangeval = ppo2params["cliprange"]
  
            cliprange_vfval = ppo2params["cliprange_vf"]
  
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = PPO2(policyval, env, gamma = gammaval, n_steps = n_stepsval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = ast.literal_eval(full_tensorboard_logval), seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), ent_coef = ent_coefval, learning_rate = learning_rateval, vf_coef = vf_coefval,  nminibatches =  nminibatchesval, noptepochs = noptepochsval, lam = lamval, cliprange_vf = ast.literal_eval(cliprange_vfval), cliprange = cliprangeval )

            #model = PPO2(MlpPolicy, env, gamma = gammaval, n_steps=128, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
            print("Custom PPO2 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
        elif algselected == 'PPO2':
            model = PPO2(MlpPolicy, env, verbose=1)
            print("PPO2 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
        elif (algselected == 'SAC') and (usecustomparams == 'true'):
            gammaval = sacparams["gamma"]
            policyval = sacparams["policy"]
            act_funcval = sacparams["act_func"]
            learning_rateval = sacparams["learning_rate"]
            verboseval = sacparams["verbose"]
            tensorboard_logval = sacparams["tensorboard_log"]
            _init_setup_modelval = sacparams["_init_setup_model"]
            full_tensorboard_logval = sacparams["full_tensorboard_log"]
            seedval = sacparams["seed"]
            n_cpu_tf_sessval = sacparams["n_cpu_tf_sess"]
            layersval = sacparams["layers"]                   
            buffer_sizeval = sacparams["buffer_size"]   
            batch_sizeval = sacparams["batch_size"]
            train_freqval = sacparams["train_freq"]
            learning_startsval = sacparams["learning_starts"]  
            tauval = sacparams["tau"]
            ent_coefval = sacparams["ent_coef"]
            target_update_intervalval = sacparams["target_update_interval"]
            gradient_stepsval = sacparams["gradient_steps"]           
            target_entropyval = sacparams["target_entropy"]
            action_noiseval = sacparams["action_noise"]
            random_explorationval = sacparams["random_exploration"]
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            #model = SAC(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = buffer_sizeval, tau = tauval, ent_coef = ent_coefval, target_update_interval = target_update_intervalval, batch_size = batch_sizeval, train_freq = train_freqval, gradient_steps = gradient_stepsval, learning_starts = learning_startsval,  action_noise = ast.literal_eval(action_noiseval), random_exploration = random_explorationval, target_entropy = target_entropyval)
            
            model = SAC(eval(policyval), env, gamma=0.99, learning_rate=0.0003, buffer_size=50000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto', action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

            print("Custom SAC training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif algselected == 'SAC':
            model = SAC(SacMlpPolicy, env, verbose=1)
            print("SAC training in process...") 
            model.learn(total_timesteps=trainepisodes, log_interval=10)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)                
                
        elif (algselected == 'TD3') and (usecustomparams == 'true'):
            gammaval = td3params["gamma"]
            policyval = td3params["policy"]
            act_funcval = td3params["act_func"]
            learning_rateval = td3params["learning_rate"]
            verboseval = td3params["verbose"]
            tensorboard_logval = td3params["tensorboard_log"]
            _init_setup_modelval = td3params["_init_setup_model"]
            full_tensorboard_logval = td3params["full_tensorboard_log"]
            seedval = td3params["seed"]
            n_cpu_tf_sessval = td3params["n_cpu_tf_sess"]
            layersval = td3params["layers"]                   
            buffer_sizeval = td3params["buffer_size"]   
            batch_sizeval = td3params["batch_size"]
            
            learning_startsval = td3params["learning_starts"]  
            tauval = td3params["tau"]
            
            policy_delayval = td3params["policy_delay"]

            action_noiseval = td3params["action_noise"]
            random_explorationval = td3params["random_exploration"]

            train_freqval = td3params["train_freq"]
            target_noise_clipval = td3params["target_noise_clip"]
            gradient_stepsval = td3params["gradient_steps"]           
            
            target_policy_noiseval = td3params["target_policy_noise"]
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))
            
            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            
            
            model = TD3(eval(policyval), env, gamma = gammaval, learning_rate = learning_rateval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), buffer_size = buffer_sizeval, tau = tauval, target_noise_clip = target_noise_clipval, policy_delay = policy_delayval, batch_size = batch_sizeval, train_freq = train_freqval, gradient_steps = gradient_stepsval, learning_starts = learning_startsval,  action_noise = ast.literal_eval(action_noiseval), random_exploration = random_explorationval, target_policy_noise = target_policy_noiseval  )

            print("Custom TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")            
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)        
        
        elif algselected == 'TD3':
            # The noise objects for TD3
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3(Td3MlpPolicy, env, action_noise=action_noise, verbose=1)
            print("TD3 training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif (algselected == 'TRPO (Requires Microsoft OpenMPI)') and (usecustomparams == 'true'):
            from stable_baselines import TRPO
            gammaval = trpoparams["gamma"]
            policyval = trpoparams["policy"]
            act_funcval = trpoparams["act_func"]
            timesteps_per_batchval = trpoparams["timesteps_per_batch"]
            verboseval = trpoparams["verbose"]
            tensorboard_logval = trpoparams["tensorboard_log"]
            _init_setup_modelval = trpoparams["_init_setup_model"]
            full_tensorboard_logval = trpoparams["full_tensorboard_log"]
            seedval = trpoparams["seed"]
            n_cpu_tf_sessval = trpoparams["n_cpu_tf_sess"]
            layersval = trpoparams["layers"]                   
            max_klval = trpoparams["max_kl"]   
            cg_itersval = trpoparams["cg_iters"]
            
            lamval = trpoparams["lam"]  
            entcoeffval = trpoparams["entcoeff"]
            
            cg_dampingval = trpoparams["cg_damping"]

            vf_stepsizeval = trpoparams["vf_stepsize"]
            vf_itersval = trpoparams["vf_iters"]

            if act_funcval == 'tf.nn.tanh' :
                policy_kwargsval = dict(act_fun = tf.nn.tanh, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.relu' :
                policy_kwargsval = dict(act_fun = tf.nn.relu, layers = ast.literal_eval(layersval))
            elif act_funcval == 'tf.nn.leaky_relu' :
                policy_kwargsval = dict(act_fun = tf.nn.leaky_relu, layers = ast.literal_eval(layersval))
            #policy_kwargsval = dict(layers = ast.literal_eval(layersval))

            model = TRPO(policyval, env, gamma = gammaval, timesteps_per_batch = timesteps_per_batchval, verbose = verboseval, tensorboard_log = tensorboard_logval, _init_setup_model = _init_setup_modelval, policy_kwargs = policy_kwargsval, full_tensorboard_log = full_tensorboard_logval, seed = ast.literal_eval(seedval), n_cpu_tf_sess = ast.literal_eval(n_cpu_tf_sessval), max_kl = max_klval, entcoeff = entcoeffval, cg_damping = cg_dampingval, cg_iters = cg_itersval,  lam = lamval,  vf_stepsize = vf_stepsizeval, vf_iters = vf_itersval,  )

            print("Custom TRPO training in process...")
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        elif algselected == 'TRPO (Requires Microsoft OpenMPI)':
            from stable_baselines import TRPO
            model = TRPO(MlpPolicy, env, verbose=1)
            print("TRPO training in process...") 
            model.learn(total_timesteps=trainepisodes)
            if savemodel == 'true':
                # Save the agent
                model.save(path)
                print("Saving the Trained Agent")
                logmessages = "The trained model was saved"
                sio.emit('messages', logmessages)
                
        #model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./A2C_newobservations/")
        #model = A2C(MlpPolicy, env, verbose=1)
        else:
            print("No learning algorithm selected for training with")
            logmessages = "No learning algorithm selected for training with"
            sio.emit('messages', logmessages)
            sio.disconnect(sid)
    
        # Test the trained agent, (currently not needed, all testing occurs in Unreal itself)
        

        
        env.render(mode='console')
        #env.render()

        obs = env.reset()
        print("Training complete")
        logmessages = "Training complete"
        sio.emit('messages', logmessages)
        intaction = 0
        #Begin strategic behvaior
        evalcomplete = evalepisodes + 2
        print(evalcomplete)
        for step in range(evalcomplete):
                action, _  = model.predict(obs, deterministic=True)
                intaction = action[0]
                print("Action: ", intaction)
                obs, reward, done, info = env.step(action)
                print('obs=', obs, 'reward=', reward, 'done=', done)
                if step == evalepisodes:
                    print(step)
                    logmessages = "Evaluation Complete"
                    sio.emit('messages', logmessages)


                    
        
    print("Disconnecting")    
    sio.disconnect()




#recieves observations and reward from Unreal Engine    
@sio.on('passobs')
def sendobs(obsdata):
    global obsflag
    global observations
    global UEreward
    global UEdone
    print('obs recieved')
    obsflag = 1
    jsonInput = json.loads(obsdata);
    observations = jsonInput['observations']     
    UEreward = jsonInput['reward'];
    UEdone = jsonInput['done'];
    

def start_client():
    sio.connect('http://localhost:3000')

#This sets up the connection, with UE Mindmaker plugin acting as the server  
if __name__ == '__main__':
   start_client()
        
    

