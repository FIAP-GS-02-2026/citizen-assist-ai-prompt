"""
Monta dist/GS2026_2_1CCR_GlaucoKelly.zip com exatamente os 7 itens do briefing e roda o checklist e a varredura de
credenciais sobre o ZIP pronto. Sai com código 1 se qualquer verificação falhar.

Uso: python ferramentas/montar_zip.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "dist" / "GS2026_2_1CCR_GlaucoKelly.zip"
ARQUIVOS = ["1_Notebook_GS2026_2.ipynb", "3_avaliacao_ragas.json", "4_README.md", "5_certificacao.txt",
            "6_Chat_Log.txt", "7_demo_video_url.txt"]


def main() -> int:
    faltando = [a for a in ARQUIVOS if not (RAIZ / a).exists()]
    if faltando:
        print("Faltam arquivos:", ", ".join(faltando))
        return 1
    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.unlink(missing_ok=True)
    with zipfile.ZipFile(DESTINO, "w", zipfile.ZIP_DEFLATED) as z:
        for a in ARQUIVOS:
            z.write(RAIZ / a, a)
        for doc in sorted((RAIZ / "2_base_documental").glob("*.md")):
            z.write(doc, f"2_base_documental/{doc.name}")
    print(f"{DESTINO.relative_to(RAIZ)} · {DESTINO.stat().st_size / 1e6:.2f} MB")
    ok = True
    for comando in (["ferramentas/verificar_segredos.py", str(DESTINO)], ["ferramentas/verificar_entrega.py", str(DESTINO)]):
        r = subprocess.run([sys.executable, *comando], cwd=RAIZ, capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
        ok &= r.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
