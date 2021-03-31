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
If the user wants to peform the preprocessing a global boolean value can be set within the jupyter notebook. The jupyter notebook itself needs sufficient GPU processing power thus we only tested it on google collab. 
Google collab allows the use of a GPU environment and therefore greatly increases the runtime. The notebook itself follows the general outline of:

	1. Initilization
	2. Animacy Detection
		* Setting global variables
		* Peform Preprocessing and Training
		* Results
	3. SRL Detection
		* Setting global variables
		* Peform Preprocessing and Training
		* Results


# Installation/How to use?

To use the jupyter notebook we recommend to download the entire compressed git repository. Next, it should be decompressed and uploaded to your preferred location on your _google drive_. Then the notebook should be accessible by going to : . All other usage steps are explained within the notebook itself.
