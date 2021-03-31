# Animate and Semantic Role Label Classification

In this project we present a novel approach towards animate and semantic role label classification by combining state of the art neural network approaches
with semantic information of the data. Semantic role label classification is a neural linguistics programming task that consists of the automatic assignment of semantic roles to each argument of the main predicate in a sentence. Animacy is the property for a referent of a noun to be an agent based on how sentient or alive the referent is.

# Motivation

The main drive behind this project is to study the impact of animacy information on semantic role labeling and vice versa. While this project is not an exercise in accuracy boosting, different machine learning approaches towards animacy detection and neural network configurations towards semantic role labeling are evaluated based on their peformance to determine the best setting.

# Framework

In general the framework of the codebase follows in parts two publications:

Shi, Peng, and Jimmy Lin: *Simple bert models for relation extraction and semantic role labeling.*\
Jahan, Labiba, Geeticka Chauhan, and Mark A. Finlayson. *"A new approach to animacy detection."*

Because the codebases for both pipelines animacy detection and semantic role classification are very large we split the jupyter notebooks into 
two parts. Both notebooks can be excuted individually from each other.


Due to the high preprocessing overhead we created three python scripts which peform the preprocessing task. The preprocessed files are already included in the git repository.
If the user wants to peform the preprocessing a global boolean value can be set within the jupyter notebooks. 

The jupyter notebooks both need sufficient GPU processing power thus we only tested it on google collab. 
Google collab allows the user to set a GPU environment which greatly increases runtime. 

All packages are either preinstalled in collab or will be installed into the environment. 


# Installation



To install the jupyter notebook we recommend following these steps:

	1. Download the entire compressed git repository. 
	2. Decompress and upload it to your preferred location on your *google drive*. 
	3. Access the notebooks by going to : https://colab.research.google.com



# How to use?

**IMPORTANT**

One parameter must be set in the beginning called `working_dir`. This parameter is the path to the saved root directory of the unzipped git repository.
For example: "/content/drive/My Drive/Colab Notebooks/" if the user has uploaded it there.

	working_dir = '/content/drive/My Drive/Colab Notebooks/'
	working_dir_extern = r'/content/drive/My\ Drive/Colab\ Notebooks/' 

**Rough outline**:

All global parameters are clearly identifiable into their own menu settings within the notebooks. There the user can specify which analysis pipeline should be peformed.


Animacy:

	1. Initilization
		Set the Working Directory
		...
		...
	2. Animacy
		Global Parameter Settings
		...
		...
		

For the semantic role classification we also included a neural network parameter settings list if the user wishes to further tweak the model. 

Semantic Role Labeling: 


	1. Initilization
		Set the Working Directory
		...
		...
	2. Semantic Role Labeling
		Global Parameter Settings
		...
		...
		Neural Network Application
		...	
		...

