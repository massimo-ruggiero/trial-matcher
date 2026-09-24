"""Entry point for the Streamlit app.

It lives at the project root because Streamlit puts the script's own directory
on sys.path: from src/gui/app.py the `src` package would not be importable.

    uv run streamlit run gui.py
"""

from src.gui.app import main

main()
