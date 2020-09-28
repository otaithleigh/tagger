tagger
++++++

Provides ``Tagger``, a Python class for handling structured integer tags.
Requires NumPy.

Installation
============

Conda::

    conda install otaithleigh::tagger

Usage
=====

A ``Tagger`` operates according to a specification. This specification is a
description of a fixed-length integer, whose digits correspond to different
"fields". For example, when tagging nodes in OpenSees, we might have a field for
the type of member (brace, column, beam, etc.), a field for the story of the
structure, and an incremental number. We'll use one digit for the type, two for
the story, and two for the number:

>>> tagger = Tagger(['type', 'story', 'story', 'num', 'num'])

Examples of valid tags under this specification:

- 10302 (type 1, story 3, num 2)
- 104 (type 0, story 1, num 4)

The tagger can process existing tags into named tuples, or generate new tags in
NumPy arrays.

>>> tagger.process_tag(10302)
Tag(type=1, story=3, num=2)
>>> tagger.tag(type=2, story=3, num=12)
array(20312)
>>> tagger.tag(type=2, story=[3, 4], num=12)
array([20312, 20412])
>>> tagger.tag(type=2, story=[3, 4], num=[12, 15, 13])
array([[20312, 20315, 20313],
       [20412, 20415, 20413]])

An optional argument ``mapping`` can be provided to the tagger to post-process
the field values. A good example is using an ``Enum`` for the member type:

>>> class MemberType(Enum):
...     BRACE = 1
...     BEAM = 2
...     COLUMN = 3
>>> tagger = Tagger(['type', 'story', 'num'], mapping={'type': MemberType})
>>> tagger.process_tag(102)
Tag(type=<MemberType.BRACE: 1>, story=0, num=2)

But any callable can be used:

>>> tagger = Tagger(['type', 'story', 'num'], mapping={'type': lambda x: x + 1})
>>> tagger.process_tag(102)
Tag(type=2, story=0, num=2)
