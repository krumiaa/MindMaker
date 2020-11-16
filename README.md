# MindMaker AI Plugin for UE4
Create Self Learning AI Agents in Unreal Engine 4

![alt text](http://www.autonomousduck.com/images/banner2.png)

**[The MindMaker AI Plugin](http://www.autonomousduck.com/mindmaker.html) (aka Neurostudio Gymwrapper) ** is an open-source plugin that enables games and simulations within UE4 to function as environments for training autonomous learning agents. The plugin facilitates a network connection and subsequent launch of a compatible MindMaker Learning Engine, for instance the [Deep Reinforcement Learning(DRL) Engine](https://unrealengine.com/marketplace/en-US/product/neurostudio-self-learning-ai). Agents can currently be trained using deep reinforcement learning, a machine learning approach that combines neural networks with a learning model to sculpt agent behavior. With MindMaker, developers and researchers can easily train machine learning agents for 2D, 3D and VR projects.

Possible applications extend beyond game design to a variety of scientific and technical endeavors. These include robotic simulation, autonomous driving, generative architecture, procedural graphics and much more. This API provides a central platform from which advances in machine learning can reach many of these fields.
For game developers, the use cases for self-optimizing agents include controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), prototyping game design decisions, and automated testing of game builds.

A functioning version of the DRL Learning Engine is included in the link to the example project. Algorithms presently supported by the DRL Learning Engine include : Actor Critic ( A2C ), Sample Efficient Actor-Critic with Experience Replay (ACER), Actor Critic using Kronecker-Factored Trust Region ( ACKTR ), Deep Q Network ( DQN ), Proximal Policy Optimization ( PPO ), Soft Actor Critic ( SAC ), Twin Delayed DDPG ( TD3 ), Trust Region Policy Optimization ( TRPO ), Deep Deterministic Policy Gradient ( DDPG ). The plugin is functionally similar to Unity’s ML agents, with some advantages - rather than needing to create custom OpenAI Gym environment for every application, one uses a single environment and simply choose which of the Agent’s observations and actions to expose to the ML algorithm. Voila, let the learning begin!



## Features
- Ready to use example project demonstrating how to use the API
- Modify the full range of learning parameters including the number of networks layers, batch size, learning rate, gamma, exploration/exploitation trade off, etc
- Self-play mechanism for training agents in adversarial scenarios
- Supported RL Algorithms include:
  -	A2C
  -	ACER
   -	ACKTR
   -	DDPG
   -	DQN
   -	GAIL
   -	HER
   -	PPO1
   -	PPO2
   -	SAC 
   -	TD3
   -	TRPO


## Components 
There are two primary components you will use while working with the MindMaker Plugin, an Unreal Project containing your learning environment, and a standalone application with the associated learning algorithm(s) being used by the agent to optimize whatever it is attempting to learn.


## Quick Install & Setup Using The MindMaker DRL Engine Starter Content
1. [Download Latest Plugin Release](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)
2. [Download a compatible MindMaker Learning Engine](http://www.autonomousduck.com/mindmaker.html)
or use the one included with the example project. 
3. Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning engine should be "Content\MindMaker\dist\mindmaker\mindmaker.exe" if the location isnt as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training.
4. Place the MindMaker AI Plugin in the Plugins directory of your UE Project.
5. If you have downloaded the MindMaker DRL Starter Content & Example Project than simply open MindMakerActorBP blueprint or MindMakerAIControlerBP blueprint from the Content\MindMakerStarterContent\Assets\MindMakerStarterContent\MindMakerActorBP directory and begin creating your custom learning agent using the functions supplied. Be sure that the Socket IO address and port your are using is set to http://localhost:3000 .

## Quick Install & Setup For Creating a Custom Learning AI from Scratch
1. [Download Latest Plugin Release](https://www.unrealengine.com/marketplace/en-US/product/mindmaker-ai-plugin)
2. [Download a compatible MindMaker Learning Engine](http://www.autonomousduck.com/mindmaker.html)
or use the one included with the example project. 
3. Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning engine should be "Content\MindMaker\dist\mindmaker\mindmaker.exe" if the location isnt as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training.
4.Place the MindMaker AI Plugin in the Plugins directory of your UE Project.
5.Add a socket IO component to the blueprint you have chosen to work with. A socketIO client component is included with Mindmaker AI plugin. Ensure that the Socket IO address and port your are using is set to http://localhost:3000
6. Connect an Event begin play node to a MindMaker Windows node (One of the Plugins Assets) within your blueprint. The MindMaker Windows node can be found under the MindMaker AI blueprints class once the plugin is installed. Currently only MS windows is supported. Once you have MindMaker Windows node connected to an event begin play node, the MindMaker AI Learning Engine will automatically launch at the beginning of play assuming you have placed it in the correct location of your Projects Content Directory.
7. Create the Reward, Action, Obersvation and Launch MindMaker Functions to use with the learning engine. For examples of how to create these, see the /examples directory which includes two maps CartPole and MatchToSample, which can be downloaded with the starter content.


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

## Key Functions to add in Unreal Engine
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

## Further Reading
[Creating Next-Gen Video Game AI With Reinforcement Learning](https://towardsdatascience.com/creating-next-gen-video-game-ai-with-reinforcement-learning-3a3ab5595d01)
