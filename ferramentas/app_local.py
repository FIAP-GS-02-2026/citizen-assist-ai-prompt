"""
Sobe a interface Gradio localmente com o MESMO código do notebook (seções 0 a 6), sem o RAGAS.
Uso: python ferramentas/app_local.py   (porta: GRADIO_SERVER_PORT, padrão 7860)
"""

import json
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
nb = json.loads((RAIZ / "1_Notebook_GS2026_2.ipynb").read_text(encoding="utf-8"))
g = {"display": lambda *a, **k: None, "__name__": "__notebook__"}
for celula in nb["cells"]:
    if celula["cell_type"] != "code":
        continue
    codigo = "".join(celula["source"])
    if "from ragas import" in codigo or "def carregar_ground_truth" in codigo:
        break
    exec(codigo, g)
print("Interface no ar. Ctrl+C para sair.", flush=True)
while True:
    time.sleep(3600)
