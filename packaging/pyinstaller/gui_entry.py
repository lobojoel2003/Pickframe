"""Entry point of the frozen desktop app."""

import multiprocessing
import sys

from pickframe.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
