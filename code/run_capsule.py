"""Thin Code Ocean entry point for registered-movie quality control.

All logic lives in the ``aind-ophys-movie-qc-library`` package; this wrapper
only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_movie_qc_library.job import run

if __name__ == "__main__":
    run()
