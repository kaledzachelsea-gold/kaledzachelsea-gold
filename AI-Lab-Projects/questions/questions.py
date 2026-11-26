
import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(
        query, file_words, file_idfs, n=FILE_MATCHES
    )

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(
        query, sentences, idfs, n=SENTENCE_MATCHES
    )

    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename
    of each .txt file inside that directory to its contents.
    """
    files = {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                with open(path, encoding="utf-8") as file:
                    files[filename] = file.read()

    return files


def tokenize(document):
    """
    Return lowercase words, excluding punctuation and English stopwords.
    """
    words = nltk.tokenize.word_tokenize(document.lower())
    stopwords = nltk.corpus.stopwords.words("english")

    return [
        word for word in words
        if not any(char in string.punctuation for char in word)
        and word not in stopwords
    ]


def compute_idfs(documents):
    """
    Calculate inverse document frequency for every word.
    """
    idfs = {}
    total_documents = len(documents)

    all_words = set()

    for words in documents.values():
        all_words.update(words)

    for word in all_words:
        documents_containing_word = sum(
            1 for words in documents.values()
            if word in words
        )

        idfs[word] = math.log(
            total_documents / documents_containing_word
        )

    return idfs


def top_files(query, files, idfs, n):
    """
    Return the n filenames that best match the query using TF-IDF.
    """
    scores = {}

    for filename, words in files.items():
        score = 0

        for word in query:
            if word in idfs:
                score += words.count(word) * idfs[word]

        scores[filename] = score

    ranked_files = sorted(
        scores,
        key=lambda filename: scores[filename],
        reverse=True
    )

    return ranked_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Rank sentences by matching word measure and query term density.
    """
    scores = {}

    for sentence, words in sentences.items():

        matching_word_measure = sum(
            idfs.get(word, 0)
            for word in query
            if word in words
        )

        if words:
            query_term_density = sum(
                1 for word in words if word in query
            ) / len(words)
        else:
            query_term_density = 0

        scores[sentence] = (
            matching_word_measure,
            query_term_density
        )

    ranked_sentences = sorted(
        scores,
        key=lambda sentence: scores[sentence],
        reverse=True
    )

    return ranked_sentences[:n]


if __name__ == "__main__":
    main()
