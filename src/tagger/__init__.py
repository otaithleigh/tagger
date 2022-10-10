import typing as t
import warnings
from collections import namedtuple, OrderedDict

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

__all__ = [
    'Tagger',
]

Spec = t.Union[t.List[str], t.Dict[str, int]]


def _parse_spec(spec: Spec) -> t.OrderedDict[str, int]:
    """Parse a spec, transforming it into an OrderedDict."""
    if isinstance(spec, dict):
        _check_all_keys_are_str(spec)
        _check_all_values_are_int(spec)
        return OrderedDict(spec)

    spec_dict = OrderedDict()

    # Iterate over the spec, creating a new entry
    # in spec_dict for each new field name.
    prev_field = None
    for field in spec:
        # Convert and check that it's a valid identifier.
        field = str(field)
        if not field.isidentifier():
            raise ValueError(
                f'field names must be valid identifiers: {field!r}')

        # Check for new field.
        if field != prev_field:
            # Check for non-contiguous fields.
            if field in spec_dict:
                raise ValueError(f'field {field!r} is non-contiguous')
            spec_dict[field] = 0

        spec_dict[field] += 1
        prev_field = field

    return spec_dict


def _check_all_keys_are_str(spec: dict):
    for key in spec.keys():
        if not isinstance(key, str):
            raise TypeError('spec keys must be str')


def _check_all_values_are_int(spec: dict):
    for value in spec.values():
        if not isinstance(value, int):
            raise TypeError('spec values must be int')



# object dtype to prevent overflow
_NBITS = np.array([8, 16, 32, 64], dtype=object)
_UINT_MAX = (2 << _NBITS - 1) - 1
"""Maximum values for 8-, 16-, 32-, and 64-bit unsigned integers."""


class Tagger():
    """Generate and parse structured integer tags.

    Tagger takes a list or dict of field names, specifying each digit of a tag,
    and processes tags into named tuples. Tags that have fewer digits than the
    specifier are zero-filled to the specifier's length -- so the tag 104 is
    equivalent to the tag 00104 for a specifier that has five digits.
    """
    def __init__(self, spec: Spec, mapping: dict = None):
        """
        Parameters
        ----------
        spec : list[str], dict[str, int]
            Tag specification. May be either a list where each element
            corresponds to a digit, or a dict mapping field names to number of
            digits in that field. If a list, the fields must be contiguous --
            ['kind', 'kind', 'num'] is acceptable, but ['kind', 'num', 'kind']
            is not.

        mapping : dict, optional
            Dict of callables that post-process the evaluated integers. Does not
            need to be defined for every field; if not present, the integer is
            returned unchanged for that field.
        """
        self.spec = _parse_spec(spec)
        self.num_fields = len(self.spec)
        self.max_length = sum(self.spec.values())
        self._max = dict(zip(self.spec, map(max_field_size, self.spec.values())))
        self.mapping = {} if mapping is None else mapping
        self._tagfactory = namedtuple('Tag', self.spec.keys())

        # Determine how many places each field needs to be shifted by to create
        # the tag. The first field doesn't need to be shifted at all, the second
        # needs to be shifted by the number of digits allotted for the first,
        # the third by the number of digits for the first AND second, etc. Done
        # in reverse order so that left-to-right order is maintained.
        fields = reversed(self.spec.keys())
        digits = list(reversed(self.spec.values()))
        num_places_to_shift = np.cumsum(digits)
        num_places_to_shift = [0, *num_places_to_shift[:-1]]
        self._num_places_to_shift = OrderedDict(zip(fields,
                                                    num_places_to_shift))

        # Determine ranges for slicing into str representations of tags.
        self._field_slices = self._calculate_field_slices()

        # Generate ufunc for parsing.
        self._parse = np.frompyfunc(self._parse_single, nin=1, nout=1)

        # Generate dtype for parse_record.
        self.dtype = self._record_dtype()

    def _calculate_field_slices(self) -> t.List[slice]:
        """Determine ranges for slicing into str representations of tags."""
        slices = []
        start = 0
        for ndigits in self.spec.values():
            stop = start + ndigits
            slices.append(slice(start, stop))
            start = stop
        return slices

    #===========================================================================
    # Parse to named tuples
    #===========================================================================
    def process_tag(self, tag: int):
        """Process a single tag.

        Parameters
        ----------
        tag : int
            Integer tag to process. Must have n or fewer digits, where n is the
            length of the specifier used to construct this object.

        Returns
        -------
        Tag
            Tag processed into descriptive fields.
        """
        warnings.warn(
            '`process_tag` has been deprecated, and will be removed '
            'in version 1.0.0. Use `parse` instead.',
            category=DeprecationWarning,
            stacklevel=2)
        return self._parse_single(tag)

    def _parse_single(self, tag: int):
        tagstr = f'{tag:0{self.max_length}d}'
        if len(tagstr) > self.max_length:
            raise ValueError(f'tag {tagstr!r} exceeds specified length')

        field_values = {}
        for field, slice in zip(self.spec.keys(), self._field_slices):
            map_field = self.mapping.get(field, lambda x: x)
            field_values[field] = map_field(int(tagstr[slice]))

        return self._tagfactory(**field_values)

    def parse(self, tags: t.Union[int, np.ndarray]):
        """Parse tags.

        Parameters
        ----------
        tags : array_like
            Integer tag(s) to parse. Each tag must have n or fewer digits, where
            n is the length of the specifier used to construct this object.

        Returns
        -------
        array[Tag]
            Tags parsed into descriptive fields.

        Example
        -------
        >>> tagger = Tagger(['kind', 'story', 'story', 'num', 'num'])
        >>> tagger.parse(10101)
        Tag(kind=1, story=1, num=1)
        >>> tagger.parse([10101, 10102])
        array([Tag(kind=1, story=1, num=1), Tag(kind=1, story=1, num=2)],
              dtype=object)
        """
        parsed: t.Union[t.NamedTuple, np.ndarray] = self._parse(tags)
        return parsed

    #===========================================================================
    # Parse to record arrays
    #===========================================================================
    def _smallest_integer_type(self, field):
        nbits = _NBITS[self._max[field] <= _UINT_MAX][0]
        return np.dtype(f'uint{nbits}')

    def _record_dtype(self):
        fields = self.spec.keys()
        dtypes = [self._smallest_integer_type(field) for field in fields]
        return np.dtype([*zip(fields, dtypes)])

    def parse_record(self, tags: t.Union[int, np.ndarray]):
        """Parse tags into a NumPy record array.

        Parameters
        ----------
        tags : array_like
            Integer tag(s) to parse. Each tag must have n or fewer digits, where
            n is the length of the specifier used to construct this object.

        Returns
        -------
        recarray
            Tags parsed into record array.

        Example
        -------
        >>> tagger = Tagger(['kind', 'story', 'story', 'num', 'num'])
        >>> tagger.parse_record(10101)
        rec.array((1, 1, 1),
                  dtype=[('kind', 'u1'), ('story', 'u1'), ('num', 'u1')])
        >>> tagger.parse_record([10101, 10102])
        rec.array([(1, 1, 1), (1, 1, 2)],
                  dtype=[('kind', 'u1'), ('story', 'u1'), ('num', 'u1')])

        Record arrays can be accessed in multiple ways:

        >>> tags = tagger.parse_record([10101, 10102, 10203])
        >>> tags.kind
        array([1, 1, 1], dtype=uint8)
        >>> tags[1]
        (1, 1, 2)
        >>> type(tags[1])
        numpy.record
        """
        tags: np.ndarray = np.asarray(tags)

        if np.any(tags < 0):
            raise ValueError('Tags must be non-negative')

        #--------------------------
        # Check length of tags.
        #--------------------------
        # Ignore divide by zero warning that occurs with log10(0) -> -inf.
        with np.errstate(divide='ignore'):
            nd = np.log10(tags)

        nd = np.maximum(nd, 0)  # Catch edge case where log10(0) -> -inf.
        nd = np.max(nd)  # Get the maximum length out of all the tags.
        nd = np.ceil(nd).astype('uint')  # Get a usable integer

        if nd > self.max_length:
            raise ValueError(f'tags have at most {nd} digits, exceeds maximum '
                             f'digits for the spec ({self.max_length})')

        #-------------
        # Parse
        #-------------
        # Construct "index" to pull appropriate pieces out of each integer.
        digit_index: np.ndarray = np.array(
            [*reversed(self._num_places_to_shift.values())])
        digit_index.shape += (1, ) * tags.ndim
        indexed = tags // 10**digit_index

        # Tags are spread out along axis 0, apply modulo along it to get values.
        exponents = np.array([*self.spec.values()])
        parsed = np.apply_along_axis(np.mod, 0, indexed, 10**exponents)

        # Convert to structured array. Transpose once to match with dtype shape,
        # and again to match shape of `tags`.
        struct = unstructured_to_structured(parsed.T, dtype=self.dtype).T

        return np.rec.array(struct)

    #===========================================================================
    # Generate tags
    #===========================================================================
    def tag(self, *values, **kwvalues) -> t.Union[int, np.ndarray]:
        """Create tags from the spec.

        Values can be specified either using positional or keyword arguments,
        but not a mix of both. All fields must have a value specified. Values
        must be scalars or castable to a NumPy array. Non-integer values must be
        castable to int.

        If arrays are specified for the values, they must be vectors (i.e. only
        one non-singular dimension). The tags are returned as an array of
        integers.

        Parameters
        ----------
        *values : array_like
            The values of the fields in the tags, specified as positional
            arguments. Must be specified in the same order as the spec. Cannot
            be mixed with **kwvalues.

        **kwvalues : array_like
            The values of the fields in the tags, specified as keyword
            arguments. Must be specified by field name. Cannot be mixed with
            *values.

        Returns
        -------
        int | np.ndarray
            Array of the created tag(s). For N inputs of length A, B, C, ..., an
            array of size A-by-B-by-C-by-... is created. Dimensions with length
            1 are squeezed out. If only a single tag was requested, an int is
            returned instead of a scalar array.

        Raises
        ------
        TypeError
            - If values could not be cast to int
        ValueError
            - If positional and keyword arguments are mixed
            - If values were not provided for all fields
            - If values are neither scalars nor vectors (i.e., ndim > 1)
            - If any values are negative
            - If any value exceeds the available digits for its field (e.g.,
              requesting the value 10 for a field with only one digit specified
              in the tag spec.)

        Examples
        --------
        >>> tagger = Tagger(['type', 'story', 'story', 'num', 'num'])

        Specify using positional arguments:

        >>> tagger.tag(2, 3, 12)
        20312

        Specify using keyword arguments:

        >>> tagger.tag(type=2, story=3, num=12)
        20312

        Generate arrays of tags from multiple inputs:

        >>> tagger.tag(type=2, story=[3, 4], num=12)
        array([20312, 20412])
        >>> tagger.tag(type=2, story=[3, 4], num=[12, 15, 13])
        array([[20312, 20315, 20313],
               [20412, 20415, 20413]])
        """
        if values and kwvalues:
            raise ValueError('Cannot mix positional and keyword arguments')

        # If values specified by position, transform to dict
        if values:
            name_value_generator = zip(self.spec.keys(), values)
        else:
            name_value_generator = kwvalues.items()

        # Transform inputs to NumPy arrays of integers in a dict
        try:
            values = {
                name: np.asarray(value).astype(int, copy=False)
                for name, value in name_value_generator
            }
        except Exception as exc:
            raise TypeError('Could not convert values to int') from exc

        # Values must be provided for all fields
        if len(values) != self.num_fields:
            raise ValueError('Insufficient number of values '
                             f'(expected {self.num_fields}, got {len(values)})')

        # Check inputs.
        for field, value in values.items():
            # Only scalars and vectors allowed
            if value.squeeze().ndim > 1:
                raise ValueError('Specified values must be scalars or vectors')

            # Negative values don't make any sense here.
            if np.any(value < 0):
                raise ValueError('All values must be nonnegative integers')

            # Check bounds.
            if np.any(value > self._max[field]):
                raise ValueError(f'{value} exceeds the available digits '
                                 f'for field {field!r}')

        # Create the tags. The passed values are reshaped into vectors with
        # ndim == num_fields. The non-singular dimension is different for each
        # vector, so for inputs of length M, N, and P, an array of size
        # M-by-N-by-P is created.
        tags = np.zeros([v.size for v in values.values()], dtype=int)
        for i, field in enumerate(self.spec.keys()):
            field_value = values[field].reshape(
                [-1 if j == i else 1 for j in range(self.num_fields)])
            tags += field_value * 10**self._num_places_to_shift[field]

        # Squeeze the generated array to remove any singular dimensions. If only
        # generating a single tag, extract from the array object.
        if tags.size == 1:
            return tags.item()
        else:
            return tags.squeeze()


def max_field_size(nd: int) -> int:
    """Return the maximum possible value for a field.

    Parameters
    ----------
    nd : int
        Number of digits assigned to the field.
    """
    # for a field that supports the range 000-999, there are three digits,
    # so the maximum value is 10**3 - 1 = 999.
    if nd < 1:
        raise ValueError('Field should have at least 1 digit')
    return 10**nd - 1
