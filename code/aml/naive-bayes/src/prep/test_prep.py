from utils import removeHtml, removePunctuations, removeNumbers


def test_removeHTML():
    data = "<p>Hello World</p>"
    assert removeHtml(data) == " Hello World "

def test_removePunctuations():
    data = "Hello World!"
    assert removePunctuations(data) == "Hello World "

def test_removeNumbers():
    data = "Hello World 123"
    assert removeNumbers(data) == "Hello World"
