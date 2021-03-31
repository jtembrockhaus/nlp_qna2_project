# Import required packages
import xml.dom.minidom
import os
import sys
import pandas as pd
import numpy as np
from progressbar import ProgressBar


"""
To perform animacy detection, a data set is needed containing each single sentence once. 
It differs from the semantic role labeling, where each sentence need to be contained in 
the dataset as often as the number of predicates in the sentence. 
The animacy vector uses a 1 to mark an animate word in a sentence, while inanimate word
are marked by 0. 
"""


##### Set the input and output directories #####
input_xml_file_dir = sys.argv[1]  # parsed as an argument from the pipeline
# directory, where dataframes of the parsed xml files are saved
output_dir = 'data/russian_fairytales/parsed_pickles'
# a list of all files in the "input_xml_file_dir"
input_files = [f for f in os.listdir(input_xml_file_dir)]


##### Function definition to parse the single sentences of the Russian Fairytales Dataset #####
def parse_story(input_filename, output_filename_df, output_filename_text):

    doc = xml.dom.minidom.parse(input_filename)  # parse xml file

    # subfunction to calculate <rep> size
    def get_number_of_elements(list):
        count = 0
        for element in list:
            count += 1
        return count

    reps = doc.getElementsByTagName('rep')  # get all <rep> tags in xml
    for rep in reps:  # iterate over <rep>'s
        rep_id = rep.getAttribute('id')  # get id attribute of respective <rep>

        # get raw text of the story
        if rep_id == 'edu.mit.story.char':
            descs = rep.getElementsByTagName('desc')
            for desc in descs:
                text = desc.firstChild.data

        # get ids, tokens and animacy
        if rep_id == 'edu.mit.parsing.token':  # get into <rep> containing tokens
            descs = rep.getElementsByTagName('desc')
            ids = []  # token id's
            tokens = []  # tokens
            lens = []  # length's of tokens
            offs = []  # position of token in the raw text
            animacy_list = []  # animacy of tokens
            for desc in descs:  # iterate over tokens
                id = desc.getAttribute('id')
                token = desc.firstChild.data
                len = desc.getAttribute('len')
                off = desc.getAttribute('off')
                ids.append(id)
                tokens.append(token)
                lens.append(len)
                offs.append(off)
                # check if animacy information is available (it is only there if the token is indeed animate)
                if desc.hasAttribute('ani'):
                    animacy_list.append(True)
                else:
                    animacy_list.append(None)

        # get pos tags
        if rep_id == 'edu.mit.parsing.pos':  # get into <rep> containing pos tags
            descs = rep.getElementsByTagName('desc')
            pos_list = []  # pos tags
            count = 0
            for desc in descs:
                # e.g. "6 NN", "15 NNP", "29 NN"
                raw = desc.firstChild.data.split(' ')
                id = raw[0]  # token id to which the pos tag is related
                pos_tag = raw[1]  # pos tag
                # only get pos-tags for our tokens (there are more in the list)
                if id == ids[count]:
                    pos_list.append(pos_tag)
                    count += 1

    data = {'id': ids, 'token': tokens, 'len': lens, 'off': offs, 'animacy': animacy_list,
            'pos': pos_list}  # setup data for the DataFrame
    df = pd.DataFrame(data)  # create DataFrame

    df.to_pickle(output_filename_df)  # save the DataFrame as .pickle file

    # save the raw text of the story as .txt file
    with open(output_filename_text, "w") as text_file:
        text_file.write(text)


##### Parse each input xml file and save it as a pandas data frame to drive #####
pbar = ProgressBar()  # initialize a progress bar to visualize the current working state
for filename in pbar(input_files):  # iterate over input files
    input_filename = input_dir + filename
    output_filename_df = output_dir + filename[:-4] + '_df.pickle'
    output_filename_text = output_dir + filename[:-4] + '_text.txt'
    # run the xml parser function from above with the respective filenames for input and output files
    parse_story(input_filename, output_filename_df, output_filename_text)
