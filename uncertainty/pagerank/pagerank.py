import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 1000000


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
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dist = dict()
    links = list(corpus[page])
    if (len(links) == 0):
        links = list(corpus)
    for curpage in corpus.keys():
        # random out of N
        base_prob = (1 / len(corpus)) * (1 - damping_factor)
        # random out of links
        spec_prob = 0
        if curpage in links: 
            spec_prob = (1 / len(links)) * damping_factor
        dist[curpage] = base_prob + spec_prob
    
    return dist


def choose_random_page(corpus):
    pages = list(corpus.keys())
    r = random.randrange(0, len(pages) - 1)
    return pages[r]


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    scores = dict()
    for key in corpus.keys():
        scores[key] = 0
    total = 0

    current = choose_random_page(corpus)
    for i in range(1, n):
        dist = transition_model(corpus, current, damping_factor)
        r = random.random()
        for page, prob in dist.items():
            r -= prob
            if r <= 0:
                current = page
                break
        
        scores[current] += 1
        total += 1
    
    # normalize scores
    for key in scores.keys():
        scores[key] /= total
        
    return scores


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = dict()
    for page in corpus:
        ranks[page] = 1 / len(corpus)
    
    bored = False
    while not bored:
        bored = True
        for page in ranks:
            # find incoming pages (i)
            incoming = set()
            for inc, links in corpus.items():
                if page in links or len(links) == 0:
                    incoming.add(inc)
            sum = 0
            for ipage in incoming:
                numlinks = len(corpus[ipage])
                if numlinks == 0:
                    numlinks = len(corpus)
                calc = ranks[ipage] / numlinks
                sum += calc
            
            newrank = ((1 - damping_factor) / len(corpus)) + (damping_factor * sum)
            if abs(newrank - ranks[page]) > 0.001:
                bored = False
            ranks[page] = newrank
    return ranks


if __name__ == "__main__":
    main()
