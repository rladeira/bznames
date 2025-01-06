from bznames.ngrams import get_ngrams


def test_get_ngrams():
    ngrams = get_ngrams("test", n=2)

    assert list(ngrams) == [
        (".", "t"),
        ("t", "e"),
        ("e", "s"),
        ("s", "t"),
        ("t", "."),
    ]
