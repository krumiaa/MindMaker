The MindMaker AI Plugin enables a networking connection and automatic launch of a compatible MindMaker Learning Engine, for instance the Deep Reinforcement Learning(DRL) Engine available for download from http://www.autonomousduck.com/mindmaker.html.

Setup Instructions 

For use with starter content from http://www.autonomousduck.com/mindmaker.html
1.	Download a compatible learning engine from www.autonomusduck.com/mindmaker or use the one included with the example project.
2.	Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning enginge should be "Content\MindMaker\dist\\mindmaker\mindmaker.exe" if the location isnt as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training
3.	Place the MindMaker AI Plugin in the Plugins directory of your UE Project
4. 	If you have downloaded the MindMaker DRL Starter Content & Example Project from http://www.autonomousduck.com/mindmaker.html than open MindMakerActorBP blueprint or MindMakerAIControlerBP from the Content\MindMakerStarterContent\Assets\MindMakerStarterContent\MindMakerActorBP directory and get started creating your custom learning agent using the functions supplied.

For creating a custom learning AI from scratch
1.	Download a compatible learning engine from http://www.autonomousduck.com/mindmaker.html or use the one included with the example project
2.	Move the learning engine and its accompanying files into the Content directory of your UE Project. The exact location of the learning enginge should be "Content\MindMaker\dist\\mindmaker\mindmaker.exe" if the location isnt exactly as specified the plugin will not work to automaticaly launch the learning engine at the start of play and you will have to manually launch mindmaker.exe before begining training
3.	Place the MindMaker AI Plugin in the Plugins directory of your UE Project
4. 	Add a socket IO component to the blueprint you have chosen to work with. A socketIO client is included with Mindmaker AI plugin
5.	Connect an Event begin play node to a MindMaker Windows node (One of the Plugins Assets). The MindMaker Windows node can be found under the MindMaker AI blueprints class once the plugin is installed. Currently only MS windows is supported. Once you have MindMaker Windows node connected to an event begin play node, the MindMaker AI Learning Engine will automatically launch at the beginning of play assuming you have placed it correctly in the Projects Content Directory.
6.	Create the Reward, Action, Obersvation and Launch MindMaker Functions to use with the learning engine. For examples of how to create these, see the /examples directory which includes two maps CartPole and MatchToSample, which can be downloaded witht the the starter content here http://www.autonomousduck.com/mindmaker.html
