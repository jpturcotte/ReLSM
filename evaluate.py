"""Deprecated wrapper forwarding to eval_hub.py.

This script exists for backward compatibility. Please invoke
`python eval_hub.py ...` directly to access the unified evaluation
interface and JSON schema.
"""

import sys

from eval_hub import main


if __name__ == "__main__":
    print("[deprecated] evaluate.py is a thin wrapper. Use eval_hub.py instead.")
    sys.exit(main())
