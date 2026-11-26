import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):

    num_pages = len(corpus)

    # Give every page its random jump probability
    probabilities = {}

    for p in corpus:
        probabilities[p] = (1 - damping_factor) / num_pages

    # Check whether the current page has outgoing links
    if len(corpus[page]) == 0:

        # Treat a page with no links as linking to every page
        for p in corpus:
            probabilities[p] += damping_factor / num_pages

    else:

        # Calculate probability of following each link
        num_links = len(corpus[page])

        for linked_page in corpus[page]:
            probabilities[linked_page] += damping_factor / num_links

    return probabilities


def sample_pagerank(corpus, damping_factor, n):

    # Create a dictionary to count page visits
    visits = {}

    for page in corpus:
        visits[page] = 0

    # Choose the first page randomly
    current_page = random.choice(list(corpus.keys()))

    visits[current_page] += 1

    # Generate the remaining samples
    for _ in range(n - 1):

        # Get transition probabilities
        probabilities = transition_model(
            corpus,
            current_page,
            damping_factor
        )

        # Separate pages and their probabilities
        pages = list(probabilities.keys())
        weights = list(probabilities.values())

        # Randomly choose the next page
        current_page = random.choices(
            pages,
            weights=weights,
            k=1
        )[0]

        # Record the visit
        visits[current_page] += 1

    # Calculate PageRank from the sample counts
    pagerank = {}

    for page in corpus:
        pagerank[page] = visits[page] / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):

    num_pages = len(corpus)

    # Initially assign equal PageRank to every page
    pagerank = {}

    for page in corpus:
        pagerank[page] = 1 / num_pages

    # Repeat until PageRank values converge
    while True:

        new_pagerank = {}

        # Calculate the new PageRank for every page
        for page in corpus:

            rank = (1 - damping_factor) / num_pages

            # Consider every possible linking page
            for linking_page in corpus:

                links = corpus[linking_page]

                # If a page has no outgoing links,
                # treat it as linking to every page
                if len(links) == 0:

                    rank += damping_factor * (
                        pagerank[linking_page] / num_pages
                    )

                # Otherwise, check whether it links to this page
                elif page in links:

                    rank += damping_factor * (
                        pagerank[linking_page] / len(links)
                    )

            new_pagerank[page] = rank

        # Check whether the rankings have converged
        converged = True

        for page in corpus:

            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                converged = False
                break

        # Update the rankings
        pagerank = new_pagerank

        if converged:
            break

    return pagerank


if __name__ == "__main__":
    main()
