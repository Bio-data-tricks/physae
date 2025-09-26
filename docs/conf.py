import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

project = "PhysAE"
author = "PhysAE contributors"
current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autosectionlabel_prefix_document = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

language = "fr"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

primary_domain = "py"
highlight_language = "python"

todo_include_todos = True
