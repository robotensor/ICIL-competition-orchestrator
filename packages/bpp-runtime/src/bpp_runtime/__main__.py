"""`python -m bpp_runtime`, the same as the `bpp-runtime` command."""

import sys

from .cli import main

sys.exit(main())
