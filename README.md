# Animate and Semantic Role Label Classification

In this project we present a novel approach towards animate and semantic role label classification by combining state of the art neural network approaches
with semantic information of the data. Semantic role label classification is a neural linguistics programming task that consists of the automatic assignment of semantic roles to each argument of the main predicate in a sentence. Animacy is the property for a referent of a noun to be an agent based on how sentient or alive the referent is.

# Motivation

The main drive behind this project is to study the impact of animacy information on semantic role labeling and vice versa. While this project is not an exercise in accuracy boosting, different machine learning approaches towards animacy detection and neural network configurations towards semantic role labeling are evaluated best on their peformance to determine the best setting.

# Framework

In general the framework of the codebase follows in parts two publications:

Shi, Peng, and Jimmy Lin: *Simple bert models for relation extraction and semantic role labeling.*
Jahan, Labiba, Geeticka Chauhan, and Mark A. Finlayson. *"A new approach to animacy detection."*

Due to the high preprocessing overhead we created three python scripts which peform the preprocessing task. The preprocessed files are already included in the git repository.
If the user wants to peform the preprocessing a global boolean value can be set within the jupyter notebook. 

The jupyter notebook itself needs sufficient GPU processing power thus we only tested it on google collab. 
Google collab allows the use of a GPU environment and therefore greatly increases the runtime. 

All needed packages are either pre installed in collab or will be installed into environment when in use. 



# Installation



To install the jupyter notebook we recommend following these steps:

	1. Download the entire compressed git repository. 
	2. Decompress and upload it to your preferred location on your *google drive*. 
	3. Access the notebook by going to : https://colab.research.google.com



# How to use?

**IMPORTANT**

One parameter must be set in the beginning called `working_dir`. This parameter is the path to the saved root directory of the unzipped git repository.
For example: "/content/drive/My Drive/Colab Notebooks/" if the user has uploaded it there.

	working_dir = '/content/drive/My Drive/Colab Notebooks/NLP/'
	working_dir_extern = r'/content/drive/My\ Drive/Colab\ Notebooks/NLP/' 

**Rough outline**:

All global parameters are clearly identifiable into their own menu settings within the notebook. There the user can specify which analysis pipeline should be peformed.
For the srl classification we also included a neural network parameter settings list if the user whiches to further tweak the model. 

	
	1. Initilization
		Set the Working Directory
	2. Animacy
		Global parameter settings
		...
		...
	3. SRL
		Global parameter settings
		...
		...
		Neural network settings
		...	
		...

