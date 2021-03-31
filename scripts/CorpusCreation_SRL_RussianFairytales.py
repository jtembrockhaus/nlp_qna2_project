# import required packages
import xml.dom.minidom
import xml.etree.ElementTree as ET
from progressbar import ProgressBar
import pickle
import os
import nltk


"""
This script creates a data dictionary for the Russian Fairytales data set. 
To perform semantic role labeling an instance for each predicate in a sentence 
need to be created. 
"""

# function to parse xml file of fairytales
def parse_story(input_filename):
    doc = xml.dom.minidom.parse(input_filename)  # parse xml file
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
            descs_total = len(descs)

            ids = []  # token id's
            tokens = []
            offs = []
            animacy = []

            for desc in descs:  # iterate over tokens

                token = desc.firstChild.data
                tokens.append(token)

                off = desc.getAttribute('off')
                offs.append(off)

                id = desc.getAttribute('id')
                ids.append(id)

                if desc.hasAttribute(
                        'ani'):  # check if animacy information id available (is only there if the token is indeed animate)
                    animacy.append('A')
                else:
                    animacy.append('O')

        # get sentence structure
        if rep_id == 'edu.mit.parsing.sentence':  # get into <rep> containing sentence structures
            descs = rep.getElementsByTagName('desc')
            sentences_length = []
            for desc in descs:
                raw = desc.firstChild.data
                sent_length = len(raw.split('~'))
                sentences_length.append(sent_length)

        # get pos tags
        if rep_id == 'edu.mit.parsing.pos':  # get into <rep> containing pos tags
            descs = rep.getElementsByTagName('desc')
            pos = []
            count = 0
            for desc in descs:
                raw = desc.firstChild.data.split(' ')  # e.g. "6 NN", "15 NNP", "29 NN"
                id = raw[0]  # token id to which the pos tag is related
                pos_tag = raw[1]  # pos tag
                if id == ids[count]:  # only get pos-tags for our tokens (there are more in the list)
                    pos.append(pos_tag)
                    count += 1

        # get semantic roles
        if rep_id == 'edu.mit.semantics.semroles':  # get into <rep> containing semantic roles
            descs = rep.getElementsByTagName('desc')
            semroles_strings = []
            semroles_offs = []
            for desc in descs:
                semroles_strings.append(desc.firstChild.data)
                semroles_offs.append(desc.getAttribute('off'))

    sentences_list, sentences_off_list, animacies_list, pos_list = [], [], [], []

    for sl in sentences_length:
        sentence = tokens[0:sl]
        sentences_list.append(sentence)
        sentence_off = offs[0]
        sentences_off_list.append(sentence_off)
        sentence_animacy = animacy[0:sl]
        animacies_list.append(sentence_animacy)
        sentence_pos = pos[0:sl]
        pos_list.append(sentence_pos)
        del tokens[0:sl]
        del offs[0:sl]
        del animacy[0:sl]
        del pos[0:sl]

    return sentences_list, sentences_off_list, animacies_list, pos_list, semroles_strings, semroles_offs


# function acces the sentence tree structure from xml file
def get_sentence_tree_from_story(input_filename):
    tree = ET.parse(input_filename)
    root = tree.getroot()

    sentence_trees = []
    sentence_trees_offs = []

    for child in root:
        if child.attrib['id'] == 'edu.mit.parsing.parse':
            for desc in child.iter('desc'):
                sentence_tree_string = desc.text[6:-1]
                sentence_tree = nltk.Tree.fromstring(sentence_tree_string)
                sentence_trees.append(sentence_tree)
                sentence_trees_offs.append(desc.attrib['off'])

    return sentence_trees, sentence_trees_offs


# function to derive tree positions of arguments from TreePointers
def get_treepos_of_TreePointer(argloc, tree):
    arg_wordnumbers = [argloc.wordnum]
    arg_positions = []
    treepos = argloc.treepos(tree)
    arg_positions.append(treepos)
    arg_tree = tree[treepos]

    return arg_wordnumbers, arg_positions, arg_tree


# function to derive tree positions of arguments from SplitTreePointer
def get_treepos_of_SplitTreePointer(argloc, tree):
    argloc_pieces = argloc.pieces
    arg_wordnumbers = []
    arg_positions = []
    arg_tree = argloc.select(tree)
    for piece in argloc_pieces:
        treepos = piece.treepos(tree)
        arg_positions.append(treepos)
        arg_wordnumbers.append(piece.wordnum)

    return arg_wordnumbers, arg_positions, arg_tree


# function to derive tree positions of arguments from ChainTreePointer
def get_treepos_of_ChainTreePointer(argloc, tree):
    argloc_pieces = argloc.pieces
    arg_wordnumbers = []
    arg_positions = []
    arg_tree = argloc.select(tree)
    for piece in argloc_pieces:
        if isinstance(piece, nltk.corpus.reader.propbank.PropbankTreePointer):
            wn, pos, t = get_treepos_of_TreePointer(piece, tree)
            arg_positions.extend(pos)
            arg_wordnumbers.extend(wn)
        elif isinstance(piece, nltk.corpus.reader.propbank.PropbankSplitTreePointer):
            wn, pos, t = get_treepos_of_SplitTreePointer(piece, tree)
            arg_positions.extend(pos)
            arg_wordnumbers.extend(wn)

    return arg_wordnumbers, arg_positions, arg_tree


# function to derive token indices from tree positions
def get_indices_of_arguments(args, tree):
    roles, indices = [], []
    for arg in args:
        argloc, argid = arg
        if isinstance(argloc, nltk.corpus.reader.propbank.PropbankTreePointer):
            arg_wordnumbers, arg_positions, arg_tree = get_treepos_of_TreePointer(
                argloc, tree)
            span = len(tree[arg_positions[0]].leaves())
            word_indices = []
            for i in range(0, span):
                word_indices.append(arg_wordnumbers[0]+i)
            roles.append(argid)
            indices.append(word_indices)

        elif isinstance(argloc, nltk.corpus.reader.propbank.PropbankSplitTreePointer):
            arg_wordnumbers, arg_positions, arg_tree = get_treepos_of_SplitTreePointer(
                argloc, tree)
            word_indices = []
            for i in range(len(arg_positions)):
                span = len(tree[arg_positions[i]].leaves())
                for j in range(0, span):
                    word_indices.append(arg_wordnumbers[i]+j)
            roles.append(argid)
            indices.append(word_indices)

        elif isinstance(argloc, nltk.corpus.reader.propbank.PropbankChainTreePointer):
            arg_wordnumbers, arg_positions, arg_tree = get_treepos_of_ChainTreePointer(
                argloc, tree)
            word_indices = []
            for i in range(len(arg_positions)):
                span = len(tree[arg_positions[i]].leaves())
                for j in range(0, span):
                    word_indices.append(arg_wordnumbers[i]+j)
            roles.append(argid)
            indices.append(word_indices)

    return roles, indices


# function to create a data dictionary for the corpus
def create_dataset(data_dict, sentences_list, sentences_off_list, animacies_list, pos_list, semroles_strings,
                   semroles_offs, sentence_trees, sentence_trees_offs):
    number_of_sentences = len(sentences_list)
    number_of_instances = len(semroles_strings)
    current_sentence = 0

    pbar = ProgressBar()  # initialize a progress bar

    for i in pbar(range(0, number_of_instances)):
        inst = semroles_strings[i]
        split_inst = inst.split(' ')
        inst_off = int(semroles_offs[i])

        if current_sentence < number_of_sentences - 1:  # check if its already the last sentence
            if i != 0:
                next_sentence_off = int(sentences_off_list[current_sentence + 1])
                if inst_off < next_sentence_off:  # check if the propbank intance corresponds to the current of the next sentence in the list
                    pass
                else:
                    current_sentence += 1
        else:
            current_sentence = number_of_sentences - 1

        # get tokens and pos tags
        tokens = sentences_list[current_sentence]
        pos = pos_list[current_sentence]
        sentence_length = len(tokens)

        # get animacy
        animacy = animacies_list[current_sentence]

        # get predicate, sense labels and the tree structure
        pred = ['O'] * sentence_length
        predicate_position = int(split_inst[0])
        predicate_identifier = split_inst[2]
        pred[predicate_position] = predicate_identifier
        sense_list = ['O'] * sentence_length
        sense_list[predicate_position] = predicate_identifier.split(".")[1]
        tree = sentence_trees[current_sentence]

        # get arguments
        arguments = []
        argument_strings = split_inst[4:len(split_inst)]
        for arg_str in argument_strings:
            arg_str_split = arg_str.split('-')
            if arg_str_split[2] != '':
                arg_role = arg_str_split[1] + '-' + arg_str_split[2]
            else:
                arg_role = arg_str_split[1]
            arg_position_string = arg_str_split[0]

            if '*' in arg_position_string:  # PropbankChainTreePointer needed
                chain_pieces = arg_position_string.split('*')
                chain_input = []
                for chain_piece in chain_pieces:
                    if ',' in chain_piece:  # the chain contains a split
                        split_pieces = chain_piece.split(',')
                        split_input = []
                        for split_piece in split_pieces:
                            pieces = split_piece.split(':')
                            wordnum = int(pieces[0])
                            height = int(pieces[1])
                            tree_pointer = nltk.corpus.reader.propbank.PropbankTreePointer(wordnum, height)
                            split_input.append(tree_pointer)
                        split_tree_pointer = nltk.corpus.reader.propbank.PropbankSplitTreePointer(split_input)
                        chain_input.append(split_tree_pointer)
                    else:  # the chain contains NO split
                        pieces = chain_piece.split(':')
                        wordnum = int(pieces[0])
                        height = int(pieces[1])
                        tree_pointer = nltk.corpus.reader.propbank.PropbankTreePointer(wordnum, height)
                        chain_input.append(tree_pointer)
                arg_pointer = nltk.corpus.reader.propbank.PropbankChainTreePointer(chain_input)

            elif ',' in arg_position_string:  # PropbankSplitTreePointer needed
                split_pieces = arg_position_string.split(',')
                split_input = []
                for split_piece in split_pieces:
                    pieces = split_piece.split(':')
                    wordnum = int(pieces[0])
                    height = int(pieces[1])
                    tree_pointer = nltk.corpus.reader.propbank.PropbankTreePointer(wordnum, height)
                    split_input.append(tree_pointer)
                arg_pointer = nltk.corpus.reader.propbank.PropbankSplitTreePointer(split_input)

            else:  # PropbankTreePointer needed
                pieces = arg_position_string.split(':')
                wordnum = int(pieces[0])
                height = int(pieces[1])
                arg_pointer = nltk.corpus.reader.propbank.PropbankTreePointer(wordnum, height)
            arguments.append((arg_pointer, arg_role))

        roles, indices = get_indices_of_arguments(arguments, tree)
        apred1 = ['O'] * sentence_length
        for k in range(0, len(roles)):
            role = roles[k]
            role_indices = indices[k]
            for j in range(0, len(role_indices)):
                if role_indices[j] < sentence_length:
                    splitvals = role.split("-")
                    if len(splitvals) > 1:
                        arg1, arg2 = splitvals
                        if arg1 in ["ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARG0"]:
                            apred1[role_indices[j]] = arg1
                        else:
                            apred1[role_indices[j]] = role
                    else:
                        apred1[role_indices[j]] = role

        # append sentence with all information to the dictionary
        data_dict.append({
            'tokens': tokens,
            'pos': pos,
            'pred_list': pred,
            'apred1': apred1,
            'sense_list': sense_list,
            'animacy': animacy
        })

    return data_dict


# function to clean wrong-formatted sentences from the dictionary
def exclude_wrong_sentences(sentences_to_exclude, sentences_list, sentences_off_list, animacies_list, pos_list,
                            sentence_trees, sentence_trees_offs):
    for sentence_number in sentences_to_exclude:
        del sentences_list[sentence_number]
        del sentences_off_list[sentence_number]
        del animacies_list[sentence_number]
        del pos_list[sentence_number]
        del sentence_trees[sentence_number]
        del sentence_trees_offs[sentence_number]

    return sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs


def execute_creation(input_files):
    pbar = ProgressBar()  # initialize a progress bar
    data_dict = []

    for filename in pbar(input_files):  # iterate over input files
        input_filename = input_dir + filename
        sentences_list, sentences_off_list, animacies_list, pos_list, semroles_strings, semroles_offs = parse_story(
            input_filename)  # run the xml parser function from above with the respective filename
        sentence_trees, sentence_trees_offs = get_sentence_tree_from_story(input_filename)

        if filename == 'story5.sty':
            sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs = exclude_wrong_sentences(
                [16, 60], sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs)

        if filename == 'story6.sty':
            sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs = exclude_wrong_sentences(
                [65], sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs)

        if filename == 'story8.sty':
            sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs = exclude_wrong_sentences(
                [46], sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs)

        if filename == 'story13.sty':
            sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs = exclude_wrong_sentences(
                [25], sentences_list, sentences_off_list, animacies_list, pos_list, sentence_trees, sentence_trees_offs)

        data_dict = create_dataset(data_dict, sentences_list, sentences_off_list, animacies_list, pos_list,
                                   semroles_strings, semroles_offs, sentence_trees, sentence_trees_offs)

        out_file = '../data/srl_detection/input/data_dict_fairytale.pickle'
        pickle.dump(data_dict, open(out_file, "wb"))


input_dir = '../data/russian_fairytales/xmls/'
input_files = [f for f in os.listdir(input_dir)]

execute_creation(input_files)
