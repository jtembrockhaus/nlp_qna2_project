import nltk
from progressbar import ProgressBar
import pickle

nltk.download('treebank')  # load nltk treebank corpus
nltk.download('propbank')  # load nltk propbank corpus
from nltk.corpus import propbank
from nltk.corpus import treebank


"""
The NLTK library provides the PropBank corpus with predicate-argument annotation for the entire Penn Treebank.
However, the Penn Treebank dataset is behind a pay wall, but 10% of the entire Penn Treebank can be imported via
the NLTK treebank corpus with the required syntactic tree structure needed for the SRL annotations. These 10% correspond
to 199 excerpts from Wall Street Journal articles with a total of 3914 sentences.
In the following script the data dictionary for the Wall Street Journal Corpus is created, that is used for the SRL pipeline.
"""

tb_fileids = treebank.fileids()  # save file id's

# determine the last usable propbank instance, for which treebank entries are present
last_usable_pb_instance = 0  # initialize variable
for index, instance in enumerate(propbank.instances()): # iterate over propbank instances
    if instance.fileid not in tb_fileids:
        last_usable_pb_instance = index
        break
# cur propbank instances
pb_instances = propbank.instances()[0:last_usable_pb_instance]


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
def create_dataset(pb_instances):
    data_dict = []  # initialize data dictionary
    number_of_sentences = len(treebank.sents())
    number_of_instances = len(pb_instances)
    pbar = ProgressBar()  # initialize a progress bar

    for i in pbar(range(0, number_of_instances)):
        inst = pb_instances[i]
        current_fileid = inst.fileid
        current_sentence_number = inst.sentnum
        sentence = treebank.tagged_sents(current_fileid)[
            current_sentence_number]
        sentence_length = len(sentence)

        # get tokens and pos tags
        tokens = []
        pos = []
        for i in range(0, sentence_length):
            tokens.append(sentence[i][0])
            pos.append(sentence[i][1])

        # get predicate and sense labels
        pred = ['O'] * sentence_length
        pred[inst.wordnum] = inst.roleset
        sense_list = ['O'] * sentence_length
        sense_list[inst.wordnum] = inst.roleset.split(".")[1]
        tree = inst.tree
        assert tree == treebank.parsed_sents(
            current_fileid)[current_sentence_number]
        arguments = inst.arguments
        roles, indices = get_indices_of_arguments(arguments, tree)

        # get semantic roles
        apred1 = ['O'] * sentence_length
        for i in range(0, len(roles)):
            role = roles[i]
            role_indices = indices[i]
            for j in range(0, len(role_indices)):
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
            'sense_list': sense_list
        })

    return data_dict


# function removing instances with more than 215 tokens (memory conflict on google colab)
def remove_large_instances(data_dict):
    data_dict_new = []
    for i, cur_dict in enumerate(data_dict):
        if len(cur_dict["tokens"]) > 215:
            pass
        else:
            data_dict_new.append(cur_dict)

    return data_dict_new


# main function executing the data dictionary creation
def main_function():
    data_dict = create_dataset(pb_instances)
    final_data_dict = remove_large_instances(data_dict)
    with open('../data/srl_detection/input/data_dict_brown.pickle', 'wb') as f:
        pickle.dump(final_data_dict, f)
    return 


# final execution
main_function()
