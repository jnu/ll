from bs4 import BeautifulSoup


def load_soup(path):
    with open(path) as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    return soup
