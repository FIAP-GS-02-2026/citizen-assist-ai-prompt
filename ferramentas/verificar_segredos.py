"""Varredura de credenciais. Falha (código 1) se encontrar algo parecido com chave de API.

Uso:
  python ferramentas/verificar_segredos.py <arquivos ou pastas...>
  python ferramentas/verificar_segredos.py --staged     (hook de pre-commit)

Procura: formato da chave da Ollama Cloud (32 hex + ponto + sufixo), atribuições do tipo
API_KEY = "..." com valor literal, tokens comuns (sk-, ghp_, hf_, AKIA) e, se existir um
.env local, o valor exato da chave (comparado sem imprimir).
"""
import os
import re
import subprocess
import sys
import zipfile

PADROES = {
    "chave Ollama": re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}\b"),
    "atribuição literal de chave": re.compile(
        r"(?i)(api_?key|token|secret|senha|password)\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"),
    "token OpenAI": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "token GitHub": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "token Hugging Face": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "chave AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "chave Groq": re.compile(r"\bgsk_[A-Za-z0-9]{40,}"),
}


def _valores_env():
    valores = []
    for caminho in (".env",):
        if os.path.exists(caminho):
            for linha in open(caminho, encoding="utf-8", errors="ignore"):
                if "=" in linha and not linha.lstrip().startswith("#"):
                    v = linha.split("=", 1)[1].strip().strip("'\"")
                    if len(v) >= 16:
                        valores.append(v)
    return valores


def verificar_texto(nome, texto, valores_env):
    achados = []
    for rotulo, padrao in PADROES.items():
        if padrao.search(texto):
            achados.append(f"{nome}: {rotulo}")
    for v in valores_env:
        if v in texto:
            achados.append(f"{nome}: valor exato de uma chave do .env")
    return achados


def _arquivos(alvos):
    for alvo in alvos:
        if os.path.isdir(alvo):
            for raiz, pastas, arquivos in os.walk(alvo):
                pastas[:] = [p for p in pastas if p not in {".git", ".venv", "__pycache__"}]
                for a in arquivos:
                    yield os.path.join(raiz, a)
        else:
            yield alvo


def main(argv):
    valores_env = _valores_env()
    achados = []
    if argv[:1] == ["--staged"]:
        nomes = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                               capture_output=True, text=True).stdout.split()
        for nome in nomes:
            if os.path.basename(nome).startswith(".env") and nome != ".env.exemplo":
                achados.append(f"{nome}: arquivo .env não pode ser commitado")
                continue
            conteudo = subprocess.run(["git", "show", f":{nome}"], capture_output=True).stdout
            achados += verificar_texto(nome, conteudo.decode("utf-8", "ignore"), valores_env)
    else:
        for caminho in _arquivos(argv or ["."]):
            if caminho.endswith(".zip"):
                with zipfile.ZipFile(caminho) as z:
                    for item in z.namelist():
                        achados += verificar_texto(f"{caminho}!{item}",
                                                   z.read(item).decode("utf-8", "ignore"), valores_env)
                continue
            if os.path.basename(caminho).startswith(".env"):
                continue
            try:
                texto = open(caminho, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            achados += verificar_texto(caminho, texto, valores_env)
    if achados:
        print("BLOQUEADO: possível credencial encontrada")
        for a in sorted(set(achados)):
            print("  -", a)
        return 1
    print("OK: nenhuma credencial encontrada")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
