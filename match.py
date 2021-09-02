import common


def load(path):
    soup = common.load_soup(path)
    qs = soup.select('.qbox span')
    return [q.text for q in qs]

