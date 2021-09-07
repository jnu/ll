from bs4 import BeautifulSoup


def load_soup(path):
    """Parse HTML from the given pass as a BeautifulSoup object.

    Returns:
        BeautifulSoup object
    """
    with open(path) as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    return soup


def filter_args(allowed, kwargs):
    """Filter a given kwarg dictionary to the allowed set.

    Args:
        allowed - Set of strings that are allowed
        kwargs - Dictionary of arguments

    Returns:
        Filtered kwargs
    """
    return {k: v for k, v in kwargs.items() if k in allowed}
