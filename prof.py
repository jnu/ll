from bs4 import BeautifulSoup


def parse_row(row):
    cells = row.find_all('td')
    q = cells[1]
    return [q.text, 'g' in q['class']]


def is_header(row):
    cells = row.find_all('td')
    qh = cells[1]
    return qh.text.strip() == 'Question'


def load_soup(path):
    with open(path) as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    return soup


def load(path):
    soup = load_soup(path)
    rows = soup.select('table.qh tr')
    return [parse_row(row) for row in rows if not is_header(row)]

