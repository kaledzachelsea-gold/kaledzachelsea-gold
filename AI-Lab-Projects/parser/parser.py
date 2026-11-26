
import nltk

import sys
import string


TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""


NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | Det N | Adj N | Det Adj N | Det Adj Adj N | NP PP
VP -> V | V NP | V PP | V Adv | VP Adv | VP PP
PP -> P NP
"""


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return

    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")

        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert sentence to a list of lowercase words.
    Remove tokens that contain no alphabetic characters.
    """

    words = nltk.tokenize.word_tokenize(sentence)

    return [
        word.lower()
        for word in words
        if any(character.isalpha() for character in word)
    ]


def np_chunk(tree):
    """
    Return noun phrase chunks.

    A noun phrase chunk is an NP subtree
    that does not contain another NP subtree.
    """

    chunks = []

    for subtree in tree.subtrees():

        if subtree.label() == "NP":

            # Get all subtrees inside this NP
            descendants = list(subtree.subtrees())[1:]

            # Only include NP if it contains no other NP
            if not any(
                descendant.label() == "NP"
                for descendant in descendants
            ):
                chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
