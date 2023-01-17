# MindMaker AI Plugin for Unreal Engine 4 & 5
Create Machine Learning AI Agents in Unreal Engine 4 & 5

![Banner](https://github.com/krumiaa/MindMaker/blob/master/banner2.png)

Intro. Video: [https://www.youtube.com/watch?v=ERm_pZhAPIA](https://www.youtube.com/watch?v=ERm_pZhAPIA)

**[The MindMaker AI Plugin](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)** is an open-source plugin that enables games and simulations within UE4 and UE5 to function as OpenAI Gym environments for training autonomous machine learning agents. The plugin facilitates a network connection between an Unreal Engine Project containing the learning environment, and a python ML library that receives data from Unreal Engine and parses into a custom OpenAI Gym environment for training the agent. The standalone machine learning library can either be a custom python script in the event you are creating your own ML tool using MindMaker’s Remote ML Server, or it could be a precompiled learning engine such as [MindMaker’s DRL Engine(Stable-Baselines3 Algorithms)](https://unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai). Regardless of which option you choose, using the MindMaker AI Plugin developers and researchers can easily train machine learning agents for 2D, 3D and VR projects.

Possible applications extend beyond game design to a variety of scientific and technical endeavors. These include robotic simulation, autonomous driving, generative architecture, procedural graphics and much more. This API provides a central platform from which advances in machine learning can reach many of these fields.
For game developers, the use cases for self-optimizing agents include controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), prototyping game design decisions, and automated testing of game builds.

The DRL Learning Engine is included in the link to the example project as well as the python source code for modifying it, found in the Content\MindMaker\Source directory. Algorithms presently supported by the DRL Learning Engine include : Actor Critic ( A2C ), Sample Efficient Actor-Critic with Experience Replay (ACER), Actor Critic using Kronecker-Factored Trust Region ( ACKTR ), Deep Q Network ( DQN ), Proximal Policy Optimization ( PPO ), Soft Actor Critic ( SAC ), Twin Delayed DDPG ( TD3 ), Trust Region Policy Optimization ( TRPO ), Deep Deterministic Policy Gradient ( DDPG ). The plugin is functionally similar to Unity’s ML agents, with some advantages - rather than needing to create custom OpenAI Gym environment for every application, one uses a single environment and simply choose which of the Agent’s observations and actions to expose to the ML algorithm. Voila, let the learning begin!

## Examples & Tutorials
### UE4 Releases
- [Download](https://unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai) Complete Examples Project File and Python Source Code
  - [CartPole Task](https://towardsdatascience.com/create-a-custom-deep-reinforcement-learning-environment-in-ue4-cf7055aebb3e): Creating A Custom Reinforcement Learning Environment  [[YouTube Demonstration Video]](https://www.youtube.com/watch?v=kNmNKnDQvgo)
  - [Automated Content Creation & A/B Testing](https://towardsdatascience.com/patterns-of-desire-deep-reinforcement-learning-for-automated-content-creation-and-a-b-testing-967fdc2b9c0d): Automated content creation with a player generated reinforcement learning signal  [[YouTube Demonstration Video]](https://youtu.be/dmxrN7uUUhs)
  - [Automated Stock Trading](https://medium.com/datadriveninvestor/using-the-game-engine-for-automated-stock-trading-without-a-single-line-of-code-31d46f548ab2): Build a Bitcoin Bot In Unreal Engine 4 using Deep Reinforcement Learning
  - Match To Sample: Solving A Memory Puzzle with a Non Player Character
### UE5.1 Stable-Baselines3 Algs
- [Plugin Download](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)
- [Examples Download](https://unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai) (includes python source code for DRL algorithms)
### Videos
- [Intro. To The MindMaker Plugin](https://www.youtube.com/watch?v=ERm_pZhAPIA) 
- [Automated Content Creation / AB Testing](https://youtu.be/dmxrN7uUUhs)
- [Creating a Custom Environment with Cart Pole Example](https://www.youtube.com/watch?v=kNmNKnDQvgo) 
### Documentation
- [Overview Of MindMaker Blueprint Functions](https://aaron-krumins.medium.com/mindmaker-ai-plugin-for-unreal-engine-4-5-blueprint-functions-9ac8d31c1df5)

## Features
- Implement multiple python based ML libraries directly in Unreal Engine with the MindMaker Client Server files – See RemoteML Example and Documentation
- Precompiled Deep Reinforcement Learning Package for production use cases – auto-launches at start of game / simulation
- Ready to use example project demonstrating how to use the API
- Modify the full range of learning parameters including the number of networks layers, batch size, learning rate, gamma, exploration/exploitation trade off, etc
- Self-play mechanism for training agents in adversarial scenarios
- Supported off the shelf Deep RL Algorithms include*:
  -	A2C
  -	ACER
   -	ACKTR
   -	DDPG
   -	DQN
   -	PPO
   -	SAC 
   -	TD3
   -	TRPO
*UE4 and UE5 versions of the DRL Learning engine contain different subset of above algorithms

## Components 
There are two primary components you will use while working with the MindMaker Plugin, an Unreal Enginge Project containing the learning environment, and a standalone machine learning library used by the agent to optimize whatever it is attempting to learn. The standalone machine learning library can either be a custom python script in the event you are creating your own ML tool using MindMaker’s Remote ML Server, or it could be a precompiled learning engine such as the [MindMaker DRL Engine(Stable Baselines Algorithms)](https://unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai). 

## MindMaker Client Server Remote ML
To experiment using different ML libraries in conjunction with MindMaker, use the MindMaker Remote ML Server. With it you can customize your own python learning engine, rather than use the pre-compiled MindMaker DRL Engine. To use the Remote ML Server, follow these steps:
-	[Download](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin) and install the free MindMaker AI Plugin for UE. This provides the Socket IO connections you will need to connect with the remote python learning engine. With the MindMaker Plugin enabled in your project, you will want to add a SocketIO component to the game object that will function as your AI. You will then need to add the relevant LaunchMinder blueprint functions for sending and receiving data from the Learning Engine and your AI game component. Examples of these can be found in DRL UE project in the next step.
-	[Download](https://www.unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai) the DRL Example project for UE – this includes the MindMaker Remote Server Application in a Windows .exe format along with the python source code for this remote server located in the Content\MindMaker\Source directory. You can place this remote server executable(or the python script) on any machine which you would like to serve as your remote server where the actual machine learning will take place. This separation of client and server allows you to have a designated cloud based server for housing and training your models. This is convenient when you have large models with potentially many clients connecting, or if the client computer is not as well equipped for machine learning.
-	Start the UE example project which contains the MindMaker plugin. Now locate and start the mindmaker executable or python source file on your remote server. Configure the SocketIO component in your UE project to connect with the remote host. The Mindmaker server you have loaded on your remote machine will default to port 3000, you will also need to know that computers remote IP address as well and configure your SocketIO component in UE accordingly.  Once you launch the UE project, it will automatically attempt to connect to the remote server and begin training, assuming you have all necessary functions given within the examples files.
-	Modify or replace the python client to use machine learning library of your choice. The python source included with the DRL examples uses an OpenAI Gym wrapper for UE that allows any OpenAI compatible machine learning library to interface with Unreal Engine.

## MARL (Multi Agent Reinforcement Learning & Working with Multiple Clients)

If you would like to employ multiple ML clients connected to a single learning environment, for instance in a multi agent scenario, this can be done using MindMaker server and plugin.

To create multiple learning agents, first setup your learning agents as shown in one of the example blueprints. 
For each new learning agent, you will need to increment the socketio port settings in the new AI controller by 1. At time of launching the server, new server port numbers are automatically created for each new instance of mindmaker.exe that you launch, starting with 3000 and going up from there for a total of 100. If you require more than 100 learning agents, request this in the github repo.

For example, If you add a second learning agent to your map, you will need all the same functions that are in the first learning agent, the launch mindmaker blueprint node etc, but instead of assigning this one to port 3000 you will assign it port 3001 in blueprints. Besides changing the socketio port setting in blueprints, you will also need to also change to change the Connect SocketIO blueprint function, modifying the In Address and Port to the new number you have created “http://localhost:3001” for instance.

Once this is done, you will just need to create a second instance of your mindmaker_client.py file that will connect to your second learning agent. Training can be done simultaneously, in parallel. The only modification that you need to make to mindmaker_client.py is changing  sio.connect('http://localhost:3000') at the bottom of the file to sio.connect('http://localhost:3001') or whatever is the number of new learning agents you are working with. If you have five learning agents, than you will have five instances of the client running and each will have a new port number all the way up to 3005



## Quick Install & Setup Using The MindMaker DRL Engine Starter Content
1. Download Latest Plugin Release from GitHub or [UE Marketplace](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)
2. [Download a compatible MindMaker Learning Engine](https://www.autonomousduck.com/mindmaker.html)
or use the one included with the example project. 
3. Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning engine should be "Content\MindMaker\dist\mindmaker\mindmaker.exe" if the location isnt as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training.
4. Place the MindMaker AI Plugin in the Plugins directory of your UE Project.
5. If you have downloaded the MindMaker DRL Starter Content & Example Project than simply open MindMakerActorBP blueprint or MindMakerAIControlerBP blueprint from the Content\MindMakerStarterContent\Assets\MindMakerStarterContent\MindMakerActorBP directory and begin creating your custom learning agent using the functions supplied. Be sure that the Socket IO address and port your are using is set to http://localhost:3000 .

## Quick Install & Setup For Creating a Custom Learning AI from Scratch
1. [Download Latest Plugin Release](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)
2. [Download a compatible MindMaker Learning Engine](https://www.autonomousduck.com/mindmaker.html)
or use the one included with the example project. 
3. Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning engine should be "Content\MindMaker\dist\mindmaker\mindmaker.exe" if the location isnt as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training.
4.Place the MindMaker AI Plugin in the Plugins directory of your UE Project.
5.Add a socket IO component to the blueprint you have chosen to work with. A socketIO client component is included with Mindmaker AI plugin. Ensure that the Socket IO address and port your are using is set to http://localhost:3000
6. Connect an Event begin play node to a MindMaker Windows node (One of the Plugins Assets) within your blueprint. The MindMaker Windows node can be found under the MindMaker AI blueprints class once the plugin is installed. Currently only MS windows is supported. Once you have MindMaker Windows node connected to an event begin play node, the MindMaker AI Learning Engine will automatically launch at the beginning of play assuming you have placed it in the correct location of your Projects Content Directory.
7. Create the Reward, Action, Obersvation and Launch MindMaker Functions to use with the learning engine. For examples of how to create these, see the /examples directory which includes two maps CartPole and MatchToSample, which can be downloaded with the starter content.

## Saving and Loading Models:
To save a trained model, set the “Save Model after Training” checkbox in the Launch MindMaker Function to True.  You will need to ensure your number of training episodes is a non zero number. The model will save after training completes.
To load the trained models, uncheck the “Save Model after Training” checkbox and instead set the “Load Pre Trained Model” checkbox in the Launch MindMaker Function to True. You will also need to set the number of training episodes to zero, since no training is to occur. Ensure the number of evaluations episodes is a non-zero integer, since this will be how the pre-trained model demonstrates learning.
Models are saved locally in the “Appdata roaming” folder of your computer, for instance c:\Users\LeoN\Appdata\Roaming

## Logging with Tensorboard:
By default MindMaker only saves to the AppData/Roaming directory on windows machines. To enable Tensorboard logging, follow these steps. 
1. Make sure the Use Custom Parameters Boolean checkbox is set to true within the Launch MindMaker Blueprint function. 2. Open the custom parameters for the algorithm you have selected to work with. Ensure the value for the full_tensorboard_log is set to "True". Next go to the tensorboard_log parameter and you will need to specify the FULL path to the directory within app data roaming where you want to save the Tensorboard log file to, for example “C:/Users/aaron/Appdata/Roaming/MindMaker/a2c_cartpole_tensorboard/”
Substitute the name of the user you are working under for aaron within the path. Do not use quotation marks. After training, go to the directory you specified and your log files will be there.


## Understand what problem your agent is trying to solve:
This is a three step process, you need to decide what actions the agent can take, what its reward criteria will be, and what observations the agent will need to make about its environment to successfully learn to receive a reward.

### Diagram of the Learning Processfor use with MindMaker

Launch MindMaker ---------> Receive Action --------> Make Obs -----> Check Rewards --------> Send Obs and Rwrd to MindMaker ------ Return To Recieve Action

In the learning process, MindMaker Learning Engine must first be configured with the observation space the agent is using and the total number of actions available to the agent. You don’t need to provide any reward information when it is initialized, this will only be encountered during training. 

The overall process for learning is that once launched and connected to connected to Unreal Engine, the MindMaker Learning Engine will begin supplying random actions for the Unreal Engine agent to take, and in response, the agent with UE will send back a list of observations it made once the action was taken, in addition to any reward it received in process. See above diagram. Over many episodes, the algorithm being employed by MindMaker will optimize the agents actions in response to the observations and rewards received from UE.
This process is the same regardless of what machine learning algorithm one chooses to employ with MindMaker.  With this information the learning algorithm being used MindMaker will begin to optimize the Agents action decisions, ideally discovering the sequence necessary to consistently receive rewards. The tradeoff between random actions and intentional ones is controlled in the exploration/exploitation parameters of ML library you have selected for use with MindMaker, for example Stable Baselines. This process repeats for each episode of training. After a fixed number of training episodes you can switch entirely to using the algorithm to predict “best” actions instead of taking random ones.

## MindMaker and the Environment Wrapper
MindMaker functions by wrapping an unreal environment in a OpenAI Gym compatible format so that any ML library that has been designed to work with OpenAI Gym can be deployed  on your Unreal Engine environment. The purpose of using Open AI Gym is to standardize the relevant factors for learning, namely, the format for receiving the agents observations, rewards and actions, so that any ML algorityhm can have access to the relevant variables for learning without needing to be retrofitted for each specific task. Algorithms that work with OpenAI Gym can than work with any environment and agent which is using the standardized OpenAI protocol.

Configuring MindMaker Learning Engine
At the outset you will need to configure the Launch Mindmaker function within Unreal Engine for your learning agent. This is done by setting the action_space variable within MindMaker to equal the total number of actions available to your agent. 
You will also need to configure the observation_space variable to match the number and type of observations your agent will be using in regard to the reward it is attempting to receive. By default, observations are passed in from Unreal as an array, see the example project.  Depending on the number of observations your agent will find necessary to use, the size of observation_space will change.



## Key Variables to add in Unreal Engine
Reward – A reward is a variable that is set according to the specific criterion you have chosen for the agent to learn or optimize around. In the UE4 blueprint you will use a branch node to determine what environmental conditions and agent action must be fulfilled for the reward to be activated. This is than passed to MindMaker by the socketIO connection. See Project example. 
Action – This is a variable that contains an integer value representing whatever action the agent has taken. You will also need to decide the total number of actions available to the agent and set the maxctions in MindMaker to equal this number.
Observations – Perhapse the trickiest variables you will be dealing with. The key to setting this correctly is to understand that the agents actions themselves must be included in the observations variable, in addition to any other environmental criterion referenced in the reward function. The agent needs to know what action or actions it took that influenced the reward and any environment variables that changed as well. These are passed to the MindMaker learning engine as an array and updated in the observations variable therein.

## MindMaker Blueprint Functions
### LaunchMindMaker Blueprint Function
Here we will discuss the individual parameters of the LaunchMindMaker blueprint node, which is the main component of the MindMaker Blueprints Functions.

RL Algorithm — This is where one can select the flavor of RL algorithm one wants to train the agent with. There are ten options in the drop down menu, with each algorithm having its own pros and cons. A detailed discussion of the available of the relevant algorithms and their use cases can be found here. https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html

Num Train EP –this is an integer input representing the number of training episodes one wishes the agent to undertake. The larger the number of training episodes, the more exploration the agent does before transitioning to the strategic behavior it acquires during training. The complexity of the actions the agent is attempting to learn typically determines the number of training episodes required — more complex strategies and behaviors require more training episodes.

Num Eval EP — This is also an integer input and represents the number of evaluation episodes the agent will undergo after training. These are the episodes in which the agent demonstrates its learned behavior.

Continuous Action Space — This is a Boolean input which determines if the agent is using a continuous action space. A continuous action space is one in which there are an infinite number of actions the agent can take, for example if it is learning to steer a car, and range of angles over which the steering column can change is a decimal value between 0 and 180, than there is an infinite number of values within that range such as .12 and 145.774454. You will want to identify at the outset of using if your agent has an infinite number of actions or finite number actions they can take. The action space must either be continuous or discrete, it cannot be both.

Discrete Action Space — This is a Boolean input which determines if the agent is using a discrete action space. A discrete action space is one in which there are a finite number of actions the agent can take, such as if the AI can only move right one space or left one space. In which case it only has two actions available to it and the action space is discrete. The user determines which kind of action space the agent will be using before using MindMaker and set these values accordingly.

Action Space Shape — This defines the lower and upper boundaries of the actions available to the agent. If you are using a discrete action space, than this is simply the total number of actions available to the agent, for instance 2 or 8. If you are using a continuous action space, things are more complicated and you must define the low and high boundaries of the action space seperatly. The format for doing so is as follows: low= lowboundary, high= highboundary,shape=(1,) In this case, lowboundary is an value such as -100.4 and highboundary is a values such as 298.46. All decimal values between these bounds will then represent actions available to the agent. If you had an array of such actions, you could change the shape portion to reflect this.

Observation Space Shape — Properly speaking this input is a python derivative of the OPEN AI custom environment class and defines the lower and upper boundaries of observations available to the agent after it takes an action. The format for doing so is as follows: low=np.array([lowboundary]), high=np.array([highboundary]),dtype=np.float32. Imagine an agent that needed to take three specific action in a row to receive a reward, then its observation space would need to include access to those three actions, which would each be represented by a unique observation. Therefore the array of observations would have to include three different values, each one with own unique boundaries. For example, such an action space might be defined as such: low=np.array([0,0,0]), high=np.array([100,100,100]),dtype=np.float32 if each of its own actions that agent needed to observe was a value between 0 and 100. A rule of thumb is that if a value is part of the reward function for the agent, ie their behavior is only rewarded if some condition being met, than the observation space must include a reference to that value. If five conditions must be met for an agent to rewarded, than each of these five conditions must be part of the agents observation space.

Load Pre Trained Model — This is a Boolean value that determines if you want to the agent to load some pre trained behavior that was previously saved. If you set this to true, you will want to specify the name of the file in the Save /Load Model name input box. All models are saved by default to the app data roaming directory of the computer for instance C:\Users\username\AppData\Roaming

Save Model After Training — This is a Boolean value that determines if you want to the agent to save the behavior it has learned after training. If you set this to true, you will want to specify the name of the file in the Save/Load Model name input box. All models are saved by default to the app data roaming directory of the computer for instance C:\Users\username\AppData\Roaming

Save/Load Model Name — This is a string representing the name of the model you wish to save or load. Files are saved to the app data roaming directory of the computer for instance C:\Users\username\AppData\Roaming

Use Custom Params — This is Boolean value that determines if you want to use the stock version of the algorithm you have selected or wish to modify its parameters. If you wish to use custom parameters these can be accessed via the custom parameters structure variables. If you click on the them, for instance A2Cparams, you will see all the values that can be set within these structures. A detailed breakdown of the parameters for each algorithm can be found here: https://stable-baselines.readthedocs.io/en/master/

### Other Blueprint Functions
A sample list of functions from the example project are presented below to understand how information is passed between MindMaker and Unreal Engine
All of the UE assets relevant to the toy problem are contained in the Assets/DeeplearningNPC folder. Of particular importance is the blueprint called AI_Character_Controler_BP
In the AI_Character_Controler_BP blueprint, all of the environment variables are configured for passing to the MindMaker standalone application. 
These include the following essential functions

Load Sensory Input function - Imports the objects to which the AI will have access to for sensing or manipulation of its environment
Environmental Controls function - This controls the logic for parts of the environment that change such switching  lights on and off etc

Define Action Space function - Encode all possible agent actions into a single numeric value that can be passed to the standalone application for evaluation by the RL algorithm

LaunchMindMaker function – this calls the standalone application at the commencing of play so that it can begin evaluation data from the UE environment. After this is initiated, the RL application begins probing the environment with random actions it generates itself, like a blind person searching in the dark for a light. The light is the reward,which is specified in UE function Check Reward function. LaunchLearningEngine also passes in some basic UE environment information to the standalone application, like the number of actions the agent can take, the total number of episodes to train for, and the number of episodes to display the agents acquired strategy after training. Displaying all the agents random training would take far too long.

ReceiveAction function – after the launch learning engine function has begun, the next function to fire is recieveaction. This receives the action that is chosen by the standalone application, and does a number of follow up procedures with it, such as updating the agents location in the environment, checking if the new action satisfies the reward condition, displaying the agents actions if we are through with training, and updated the agents observations about its environment so that they can be passed back to the standalone application in the next episode. 

Make Observations function – The purpose of this is to update the agents observations about its environment following the action it has just taken. These will include, for instance, the agents location with the environment and any other environmental data that has changed since it last took an action. These are stored in a custom structure variable.

CheckReward – this specifies the reward condition for the agent in the environment. If this reward condition is met following the agent taking an action, this information is passed to the standalone application in the send observations function that follows.
Send Observations Function – takes the new observations made by the agent as well as any reward information and passes them to the standalone application. This is how the RL algorithm will be able to evaluate whether the action it has just taken was a good one, and update its strategy accordingly. After this function fires, the one iteration or episode of the game is complete, and the process repeats ad infinitum.

## FAQ
### Q: In the exploration stage it seems that the agent does not move, only stands still.

Certain tasks may require extended periods of training where visualizing the agent movements would prove prohibitively time consuming. As such in certain examples visualizing the agent's movements has been disabled, but training is happening in the background once the example is run and upon completion, the agent will demonstrate the acquired strategy.

### Q: How is exploration used, what strategy is used to explore? It is not clear to me what this "random" means, can you give an example?

Random in this case means that the agent is using a random number generator to choose between the actions available to it during training. The RL Algorithm then observes the results of these random actions as well as any rewards received and uses this information to choose better actions during the “exploitation” phase. This is how a learned strategy is developed.

### Q: What is the information gathered in the learning stage, what does it look like?

The information gathering during learning takes the form of an array of observations that are generated after each of the agent’s random actions. If using the MindMaker plugin, the exact shape of the array is defined in the Observation Size property of the Launch Mindmaker Blueprint function and will depend on what variables are necessary for the agent to observe in that particular game or learning task. It will change depending on the learning task or game.

### Q: Can the agent perceive the whole environment, or just a small area around him?

The agent perceives only the part of the environment that is exposed to them by the game designer. When using the Mindmaker plugin, these observations are populated in the Make Observations blueprint function call within Unreal Engine. This will generate an array of numbers in the shape defined by Observation Size property of the Launch Mindmaker Blueprint function. Observations should be selected so that they only comprise the data that is necessary for the agent to learn from, otherwise training could become prohibitively time consuming.

### Q: What neural network is used? ANN/CNN/RNN?

In vanilla Q Learning – no neural network is required and learning is stored in a tabular format. When using the MindMaker Deep Reinforcement Learning one can choose between a variety of neural network architectures including RNN, CNN etc. One can set these within each algorithm's custom properties of the Launch Mindmaker Blueprint function call. 

## Further Resources
[Creating a Custom Deep Reinforcement Learning Environment](https://towardsdatascience.com/create-a-custom-deep-reinforcement-learning-environment-in-ue4-cf7055aebb3e)

[Intro. to Reinforcement Learning for Video Game AI](https://towardsdatascience.com/creating-next-gen-video-game-ai-with-reinforcement-learning-3a3ab5595d01)

[Reinforcement Learning – It’s Promise and Peril](https://www.amazon.com/Outsmarted-Reinforcement-Learning-Promise-Peril-ebook/dp/B08BG9FDC2/)

[Stable Baselines Documentation](https://stable-baselines.readthedocs.io/en/master/)

