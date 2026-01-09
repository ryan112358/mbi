"""Defines the Domain class for representing attribute domains.

This module provides the `Domain` class, which encapsulates a set of named
attributes and their corresponding discrete sizes (shapes). It facilitates
representing the structure of datasets and graphical models and supports
various operations like projection, marginalization, and merging of domains.
"""
import functools
from collections.abc import Iterator, Sequence
from typing import Any

import attr


@attr.dataclass(frozen=True)
class Domain:
    """Represents a discrete domain defined by attributes and their sizes.

    This class encapsulates a set of named attributes and their corresponding
    discrete sizes (shapes). It provides methods for common domain operations.

    Attributes:
        attributes (tuple[str, ...]): A tuple containing the names of the
            attributes in the domain.
        shape (tuple[int, ...]): A tuple containing the integer sizes
            (number of discrete values) for each corresponding attribute in
            the `attributes` tuple.
        labels (tuple[tuple[Any, ...], ...] | None): An optional tuple of tuples
            containing semantic information (labels) for each attribute's values.
            Must be the same length as attributes, and each inner tuple must have
            length corresponding to the attribute's size.

    Supported Operations:
        - Projection (`project`): Creates a new domain with a subset of attributes.
        - Marginalization (`marginalize`): Creates a new domain excluding specified attributes.
        - Intersection (`intersect`): Creates a new domain containing only common attributes.
        - Merging (`merge`): Combines two domains into a larger one.
        - Size Calculation (`size`): Computes the total number of configurations in the domain or a subset.

    Example Usage (using fromdict):
        >>> domain = Domain.fromdict({'a': 2, 'b': 3})
        >>> print(domain)
        Domain(a: 2, b: 3)
    """
    attributes: tuple[str, ...] = attr.field(converter=tuple)
    shape: tuple[int, ...] = attr.field(converter=lambda sh: tuple(int(n) for n in sh))
    labels: tuple[tuple[Any, ...], ...] | None = attr.field(
        default=None,
        converter=lambda l: tuple(tuple(x) for x in l) if l is not None else None,
    )

    def __attrs_post_init__(self):
        if len(self.attributes) != len(self.shape):
            raise ValueError("Dimensions must be equal.")
        if len(self.attributes) != len(set(self.attributes)):
            raise ValueError("Attributes must be unique.")
        if self.labels is not None:
            if len(self.labels) != len(self.attributes):
                raise ValueError("Labels must be same length as attributes.")
            for i, l in enumerate(self.labels):
                if len(l) != self.shape[i]:
                    raise ValueError(
                        f"Labels for {self.attributes[i]} must have length {self.shape[i]}."
                    )

    @functools.cached_property
    def config(self) -> dict[str, int]:
        """Returns a dictionary of { attr : size } values."""
        return dict(zip(self.attributes, self.shape))

    @functools.cached_property
    def labels_config(self) -> dict[str, tuple[Any, ...]] | None:
        """Returns a dictionary of { attr : labels } values."""
        if self.labels is None:
            return None
        return dict(zip(self.attributes, self.labels))

    @staticmethod
    def fromdict(config: dict[str, int]) -> "Domain":
        """Construct a Domain object from a dictionary of { attr : size } values.

        Example Usage:
            >>> print(Domain.fromdict({'a': 10, 'b': 20}))
            Domain(a: 10, b: 20)

        Args:
          config: a dictionary of { attr : size } values
        Returns:
          the Domain object
        """
        return Domain(config.keys(), config.values())

    def project(self, attributes: str | Sequence[str]) -> "Domain":
        """Project the domain onto a subset of attributes.

        Args:
          attributes: the attributes to project onto
        Returns:
          the projected Domain object
        """
        if isinstance(attributes, str):
            attributes = [attributes]
        if not set(attributes) <= set(self.attributes):
            raise ValueError(f"Cannot project {self} onto {attributes}.")
        shape = tuple(self.config[a] for a in attributes)
        labels = None
        if self.labels is not None:
            labels = tuple(self.labels_config[a] for a in attributes)
        return Domain(attributes, shape, labels=labels)

    def marginalize(self, attrs: Sequence[str]) -> "Domain":
        """Marginalize out some attributes from the domain (opposite of project).

        Example Usage:
            >>> D1 = Domain(['a','b'], [10,20])
            >>> print(D1.marginalize(['a']))
            Domain(b: 20)

        Args:
          attrs: the attributes to marginalize out.
        Returns:
          the marginalized Domain object
        """
        proj = [a for a in self.attributes if a not in attrs]
        return self.project(proj)

    def contains(self, other: "Domain") -> bool:
        """Checks if this domain contains all attributes present in another domain."""
        return set(other.attributes) <= set(self.attributes)

    def canonical(self, attrs):
        """Returns attributes common to the domain and input, maintaining the domain's order."""
        return tuple(a for a in self.attributes if a in attrs)

    def invert(self, attrs):
        """Returns attributes present in the domain but not in the provided list."""
        return [a for a in self.attributes if a not in attrs]

    def intersect(self, other: "Domain") -> "Domain":
        """Intersect this Domain object with another.

        Example Usage:
            >>> D1 = Domain(['a','b'], [10,20])
            >>> D2 = Domain(['b','c'], [20,30])
            >>> print(D1.intersect(D2))
            Domain(b: 20)

        Args:
          other: another Domain object
        Returns:
          the intersection of the two domains
        """
        return self.project([a for a in self.attributes if a in other.attributes])

    def axes(self, attrs: Sequence[str]) -> tuple[int, ...]:
        """Return the axes tuple for the given attributes.

        Args:
          attrs: the attributes
        Returns:
          a tuple with the corresponding axes
        """
        return tuple(self.attributes.index(a) for a in attrs)

    def merge(self, other: "Domain") -> "Domain":
        """Merge this Domain object with another.

        :param other: another Domain object
        :return: a new domain object covering the full domain

        Example:
            >>> D1 = Domain(['a','b'], [10,20])
            >>> D2 = Domain(['b','c'], [20,30])
            >>> print(D1.merge(D2))
            Domain(a: 10, b: 20, c: 30)

        Args:
          other: another Domain object
        Returns:
          a new domain object covering the combined domain.
        """
        extra = other.marginalize(self.attributes)
        new_labels = None
        if self.labels is not None and other.labels is not None:
            new_labels = self.labels + extra.labels
        return Domain(
            self.attributes + extra.attributes,
            self.shape + extra.shape,
            labels=new_labels,
        )

    def size(self, attributes: Sequence[str] | None = None) -> int:
        """Return the total size of the domain.

        Example:
            >>> D1 = Domain(['a','b'], [10,20])
            >>> D1.size()
            200
            >>> D1.size(['a'])
            10

        Args:
          attributes: A subset of attributes whose total size should be returned.
        Returns:
          the total size of the domain
        """
        if attributes is None:
            return functools.reduce(lambda x, y: x * y, self.shape, 1)
        return self.project(attributes).size()

    @property
    def attrs(self):
        """Alias for the `attributes` tuple."""
        return self.attributes
    
    def supports(self, attrs: str | Sequence[str]) -> bool:
        if isinstance(attrs, str):
            attrs = [attrs]
        return set(attrs) <= set(self.attributes)

    def __contains__(self, name: str) -> bool:
        """Check if the given attribute is in the domain."""
        return name in self.attributes

    def __getitem__(self, a: str) -> int:
        return self.config[a]

    def __iter__(self) -> Iterator[str]:
        return self.attributes.__iter__()

    def __len__(self) -> int:
        return len(self.attributes)

    def __str__(self) -> str:
        inner = ", ".join(["%s: %d" % x for x in zip(self.attributes, self.shape)])
        return "Domain(%s)" % inner
