"""Local Avito prototype."""
from pathlib import Path
import sys

# The checked-out prototype can use its existing local dependencies; pip installs
# use their environment normally when this directory is absent.
_local_dependencies = Path(__file__).resolve().parents[1] / '.tools' / 'python'
if sys.prefix == sys.base_prefix and _local_dependencies.is_dir():
    sys.path.insert(0, str(_local_dependencies))
