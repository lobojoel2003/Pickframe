"""Entry point of the frozen command-line tool."""

import multiprocessing
import sys

from pickframe.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
