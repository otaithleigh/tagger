import numpy as np
from pytest import raises, warns
from tagger import Tagger


def test_contiguous_fields():
    """Non-contiguous fields are not allowed."""
    with raises(ValueError):
        tagger = Tagger(["type", "story", "num", "story", "num"])


def test_process_tag():
    tagger = Tagger(["type"])
    with warns(DeprecationWarning):
        tag = tagger.process_tag(1)
    assert tag.type == 1


def test_basics():
    tagger = Tagger(["kind", "alpha", "beta"])
    tag = tagger.parse(103)
    assert tag.kind == 1
    assert tag.alpha == 0
    assert tag.beta == 3


def test_spec_dict():
    tagger = Tagger({"kind": 2, "spud": 4})
    assert tagger.spec == {"kind": 2, "spud": 4}
    assert tagger.max_length == 6


def test_spec_dict_bad_key_type():
    with raises(TypeError):
        _ = Tagger({1: 2})
    with raises(TypeError):
        _ = Tagger({b"a": 2})


def test_spec_dict_bad_value_type():
    with raises(TypeError):
        _ = Tagger({"aaah": 3.0})


def test_empty_leading_field():
    tagger = Tagger(["kind", "alpha", "beta"])
    tag = tagger.parse(33)
    assert tag.kind == 0
    assert tag.alpha == 3
    assert tag.beta == 3


def test_parse_multiple_tags():
    tagger = Tagger(["type", "spud"])
    tags = tagger.parse([12, 24, 26])
    assert tags[0] == (1, 2)
    assert tags[1] == (2, 4)
    assert tags[2] == (2, 6)


def test_single():
    tagger = Tagger("knc")
    tag = tagger.tag(1, 1, 1)
    assert tag == 111 and isinstance(tag, int)


def test_multidimensional_arrays():
    tagger = Tagger("knncc")
    tags = tagger.tag(3, [4, 5], [6, 7, 8])
    assert np.all(tags == [[30406, 30407, 30408], [30506, 30507, 30508]])


def test_max():
    tagger = Tagger("kkknnnnnncc")
    assert tagger._max["k"] == 999
    assert tagger._max["n"] == 999999
    assert tagger._max["c"] == 99
