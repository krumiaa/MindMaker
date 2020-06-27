# MindMaker Example Project

**Download the UE4 Example Project at the link below:**

https://drive.google.com/file/d/1x_8n1qb_KMrfkAHTd3Hjwjh0u679qmdB/view?usp=sharing

To run the MindMaker example first one must launch MindMakey.py. This is a standalone python application located at MindMakerExample\Content\NeuroStudio\Assets\LearningEngine\MindMaker.py
It contains the learning algorithm the agent will use to develop a strategic behavior. Once the MindMaker.py is launched, it must be kept up and running for the duration of the demonstration. 

Next one can open the Unreal Engine Project and hit the play button. The agent pictured in the environment will not initially move but it will begin simulating actions, as seen in the log data streaming to the screen.  During this time it is mostly taking random actions, traveling to one of 4 locations in the UE environment represented by the balls and cone.  

## Example Provided: The Extended Match to Sample Task

In the example provided we use deep reinforcement learning to solve a match to sample puzzle in which the NPC learns that it must go to the cone and then the gold bowl to recieve a reward. Similar puzzles have been used in a wide variety of animal learning experiments that explore instrumental and associative learning abilities. 

The key point in such a learning task is that the agent must learn to predict that it can take an action to receive a reward only during specific circumstance. In this case, when  touches the cone and then proceeds to the gold food bowl.

The setup for the match to sample task is that NPC begins with a training phase in which it randomly travels from one of four locations, 3 bowls and one switch, represented by balls and a cone respectively. During training period, it learns associations about the values of each of these state action pairs.

If successful, after training the agent displays intentional behavior by first going to the switch(cone) and then going to the food reward bowl. This method can be used to provide a wide variety of intentional behaviors, including avoiding enemy players, collecting health points, etc.

One could have the agent actually move to these locations during training but it would be  time consuming, so instead it is merely simulated. The number of episodes to simulate vs the number of episodes to actually demonstrate the acquired strategy is specified  in the AI_Controller . In this case it is 2000 simulated actions and 10 actions that are visualized to demonstrate learning.  The agent will often learn the correct strategy in as little 1,000 training episodes, but 2000 ensures convergence. 
The agent is only rewarded if it first travels to the cone and then to the gold food bowl. This reward behavior is specified in the CheckReward function. All of the agents learning however takes place in the MindMaker.py using the RL python library.  This allows one to switch out the “brains of the AI” quickly by just 
