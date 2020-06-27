# MindMaker GymWrapper
MindMaker UE4 Machine Learning Toolkit 

![alt text](http://www.autonomousduck.com/images/banner.png)

The UE4 Machine Learning Toolkit (MindMaker) is an open-source project that enables games and simulations within UE4 to function as environments for training autonomous learning agents. Agents can be trained using deep reinforcement learning, genetic algorithms and a variety of other machine learning methods made accessible through an easy-to-use Python API.  With this tool game developers and researchers can easily train machine learning agents for 2D, 3D and VR projects. 
Possible applications extend beyond game design to a variety of scientific and technical endeavors. These include robotic simulation, autonomous driving, generative architecture, procedural graphics and much more. This API provides a central platform from which advances in machine learning can reach many of these fields.
For game developers, the use cases for self-optimizing agents include controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), prototyping game design decisions, and automated testing of game builds.
The API provides a wrapper for OPEN AI Gym Environments, effectively porting any Unreal Engine 4 environment into a format compatible with a large variety of Python Machine Learning libraries including Stable Baselines  and Uber’s Neuroevolution.  With the API, algorithms can quickly be deployed in UE4 without the user needing to implement them from scratch. The plugin is functionally similar to Unity’s ML agents, with some advantages - rather than needing to create custom OpenAI Gym environment for every application, one uses a single environment and simply choose which of the Agent’s observations and actions to expose to the ML algorithm. Voila, let the learning begin!
Features
•	Ready to use example project demonstrating how to use the API
•	Modify the full range of learning parameters including the number of networks layers, batch size, learning rate, gamma, exploration/exploitation trade off, etc
•	Self-play mechanism for training agents in adversarial scenarios
•	Supported RL Algorithms include:
o	A2C
o	ACER
o	ACKTR
o	DDPG
o	DQN
o	GAIL
o	HER
o	PPO1
o	PPO2
o	SAC 
o	TD3
o	TRPO

Components 
There are two primary components you will use while working with the MindMaker toolkit, an Unreal Project containing your learning environment, and a standalone python application(MindMaker.py) comprising the OpenAI GymWrapper and the associated learning algorithm(s) being used by the agent to optimize whatever it is attempting to learn.
MindMaker.py is essentially agnostic to the problem described in the UE environment and can therefore be used to solve/play any type of game created in Unreal Engine, provided it follows the specified format for passing environmental variables to the learning algorithm.
One may think of MindMaker.py as the “brains” of the AI, which communicates with the unreal engine environment through a socketIO connection.
Installation
You will need to install all dependencies associated with MindMaker.py prior to running the standalone application. 
Python Dependencies 
 
Flask==1.0.2
eventlet==0.20.1
gym==0.17.1
stable_baselines==2.10.0  (or other any other ML library you wish to use in conjunction with mindmaker.py)
numpy==1.18.2
gevent==20.6.2
gevent_socketio==0.3.6
gevent_websocket==0.10.1
python_engineio==3.13.0
Once these have been successfully installed using pip, you will want to configure your unreal environment for communicating with MindMaker.py

SocketIO Plugin – You will be communicating with MindMaker.py through a socketIO connection and therefore must have a socketIO plugin installed for unreal engine. You can install this from Github or the UE4 Marketplace
GitHub Release
https://github.com/getnamo/socketio-client-ue4
UE4 Marketplace
https://www.unrealengine.com/marketplace/en-US/product/socket-io-client

After this you will need to configure your Unreal Engine environment for passing data back and forth between MindMaker.py and Unreal Engine
Steps
•	Add SocketIO to whatever UE4 game asset blueprint you have chosen to function as your Learning Agent.
•	Ensure you have selected Port 3000 within Unreal Engine Socket IO to communicate with MindMaker.py. This can be done within Blueprints.
•	Configure the the SocketIO blueprint function to receive actions from MindMaker.py and send back observations and reward data. Examples of how these can be setup are found in the sample project. 


Understand what problem your agent is trying to solve:
This is a three step process, you need to decide what actions the agent can take, what its reward criteria will be, and what observations the agent will need to make about its environment to successfully learn to receive a reward.
Diagram of the Learning Processfor use with MindMaker

Launch MindMaker.py   ---------Receive Action --------Make Obs-----Check Rewards 							|						|							|                    					|
 ---------------------Send Obs to MindMaker.py 
In the learning process, MindMaker.py must first be configured with the observation space the agent is using and the total number of actions available to the agent. You don’t need to provide any reward information when it is initialized, this will only be encountered during training. 
The overall process for learning is that once launched and connected to connected to Unreal Engine, MyMaker.py will begin supplying random actions for the Unreal Engine agent to take, and in response, the agent with UE will send back a list of observations it made once the action was taken, in addition to any reward it received in process. See above diagram. Over many episodes, the algorithm being employed by MindMaker.py will optimize the agents actions in response to the observations and rewards received from UE.
 This process is the same regardless of what machine learning algorithm one chooses to employ with MindMaker.py.  With this information the learning algorithm being used MindMaker.py will begin to optimize the Agents action decisions, ideally discovering the sequence necessary to consistently receive rewards. The tradeoff between random actions and intentional ones is controlled in the exploration/exploitation parameters of ML library you have selected for use with MindMaker, for example Stable Baselines. This process repeats for each episode of training. 
After a fixed number of training episodes you can switch entirely to using the algorithm to predict “best” actions instead of taking random ones.
MindMakey.py and the Environment Wrapper
MindMaker.py functions by wrapping an unreal environment in a OpenAI Gym compatible format so that any ML library that has been designed to work with OpenAI Gym can be deployed  on your Unreal Engine environment. The purpose of using Open AI Gym is to standardize the relevant factors for learning, namely, the format for receiving the agents observations, rewards and actions, so that any ML algorityhm can have access to the relevant variables for learning without needing to be retrofitted for each specific task. Algorithms that work with OpenAI Gym can than work with any environment and agent which is using the standardized OpenAI protocol.
Configuring MindMaker.py
At the outset you will need to configure class UnrealEnvWrap for your Unreal Engine learning agent. This is done by setting the self.action_space variable within MindMaker.py to equal the total number of actions available to your agent. This can be passed in from unreal engine via the socketIO connection, or changed directly in MindMaker.py
You will also need to configure self.observation_space variable to match the number and type of observations your agent will be using in regard to the reward it is attempting to receive. By default, observations are passed in from Unreal as an array, see the example project.  Depending on the number of observations your agent will find necessary to use, the size of self.observation_space will change. It needs to match the array size you are passing in from Unreal Engine, so this must be set accordingly in MindMaker.py at the outset. 

Key Variables to add in Unreal Engine
Reward – A reward is a variable that is set according to the specific criterion you have chosen for the agent to learn or optimize around. In the UE4 blueprint you will use a branch node to determine what environmental conditions and agent action must be fulfilled for the reward to be activated. This is than passed to MindMaker.py by the socketIO connection and used in the MindMaker.py step function. See Project example. 
Action – This is a variable that contains an integer value representing whatever action the agent has taken. You will also need to decide the total number of actions available to the agent and set the maxctions in MindMaker.py  to equal this number.
Observations – Perhapse the trickiest variables you will be dealing with. The key to setting this correctly is to understand that the agents actions themselves must be included in the observations variable, in addition to any other environmental criterion referenced in the reward function. The agent needs to know what action or actions it took that influenced the reward and any environment variables that changed as well. These are passed to MindMakey.Py as an array and updated in the observations variable therein.
Key Functions to add in Unreal Engine
A sample list of functions from the example project are presented below to understand how information is passed between MindMaker.py and Unreal Engine
All of the UE assets relevant to the toy problem are contained in the Assets/DeeplearningNPC folder. Of particular importance is the blueprint called AI_Character_Controler_BP
In the AI_Character_Controler_BP blueprint, all of the environment variables are configured for passing to the MindMaker standalone application. 
These include the following essential functions
Load Sensory Input function - Imports the objects to which the AI will have access to for sensing or manipulation of its environment
Environmental Controls function - This controls the logic for parts of the environment that change such switching  lights on and off etc
Define Action Space function - Encode all possible agent actions into a single numeric value that can be passed to the standalone application for evaluation by the RL algorithm
LaunchLearningEngine function – this calls the standalone application at the commencing of play so that it can begin evaluation data from the UE environment. After this is initiated, the RL application begins probing the environment with random actions it generates itself, like a blind person searching in the dark for a light. The light is the reward,which is specified in UE function Check Reward function. LaunchLearningEngine also passes in some basic UE environment information to the standalone application, like the number of actions the agent can take, the total number of episodes to train for, and the number of episodes to display the agents acquired strategy after training. Displaying all the agents random training would take far too long.
ReceiveAction function – after the launch learning engine function has begun, the next function to fire is recieveaction. This receives the action that is chosen by the standalone application, and does a number of follow up procedures with it, such as updating the agents location in the environment, checking if the new action satisfies the reward condition, displaying the agents actions if we are through with training, and updated the agents observations about its environment so that they can be passed back to the standalone application in the next episode. 
Make Observations function – The purpose of this is to update the agents observations about its environment following the action it has just taken. These will include, for instance, the agents location with the environment and any other environmental data that has changed since it last took an action. These are stored in a custom structure variable.
CheckReward – this specifies the reward condition for the agent in the environment. If this reward condition is met following the agent taking an action, this information is passed to the standalone application in the send observations function that follows.
Send Observations Function – takes the new observations made by the agent as well as any reward information and passes them to the standalone application. This is how the RL algorithm will be able to evaluate whether the action it has just taken was a good one, and update its strategy accordingly. After this function fires, the one iteration or episode of the game is complete, and the process repeats ad infinitum.
