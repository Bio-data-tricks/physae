#!/usr/bin/env python3
"""Compatibilité CLI pour l'entraînement PIAE."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve()
    target = script_path.with_name("physae")
    if not target.exists():
        raise FileNotFoundError(
            "Script d'entraînement introuvable. Attendu à l'emplacement suivant : "
            f"{target}"
        )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
