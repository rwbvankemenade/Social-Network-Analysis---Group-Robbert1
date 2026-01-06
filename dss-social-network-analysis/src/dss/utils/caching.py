"""Caching utilities for expensive computations.

Streamlit provides decorators such as `st.cache_data` and `st.cache_resource`
to memoise the results of expensive functions and make dashboards
responsive.  This module wraps these decorators so that the same code can
be executed in contexts where Streamlit is not available (for example,
during unit tests) by falling back to Python’s `lru_cache`.
"""

from functools import lru_cache, wraps
from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable[..., object])


def cache_data(func: F) -> F:
    """Cache the results of a function using Streamlit or lru_cache.

    When running inside Streamlit, use `st.cache_data` to persist results
    between reruns.  Outside of Streamlit, fall back to `functools.lru_cache`.
    """

    try:
        import streamlit as st  # type: ignore
        return st.cache_data()(func)  # type: ignore
    except Exception:
        return lru_cache(maxsize=None)(func)  # type: ignore


def cache_resource(func: F) -> F:
    """Cache resources (e.g., loaded models) using Streamlit or lru_cache."""
    try:
        import streamlit as st  # type: ignore
        return st.cache_resource()(func)  # type: ignore
    except Exception:
        return lru_cache(maxsize=None)(func)  # type: ignore


if __name__ == "__main__":
    # Demonstrate caching behaviour
    @cache_data
    def expensive(x: int) -> int:
        print(f"Computing square of {x}")
        return x * x

    print(expensive(3))
    print(expensive(3))  # Cached result prints only once