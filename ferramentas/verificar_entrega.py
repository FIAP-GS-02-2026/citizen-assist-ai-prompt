"""
Checklist automático da GS 2026.2 de Prompt and AI: confere a entrega item a item contra o briefing
e contra o que o prof. Jorge cobrou nas Sprints (guardrails sondados, versões fixadas, diagrama como arquivo).

Uso:
  python ferramentas/verificar_entrega.py                 (confere a pasta do repositório)
  python ferramentas/verificar_entrega.py dist/GS...zip   (confere o ZIP final)

Sai com código 1 se algum item obrigatório falhar. Cada linha diz o item, o resultado e a evidência.
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ITENS_ZIP = ["1_Notebook_GS2026_2.ipynb", "2_base_documental/", "3_avaliacao_ragas.json", "4_README.md",
             "5_certificacao.txt", "6_Chat_Log.txt", "7_demo_video_url.txt"]
TITULOS = ["0. Metadados", "1. Base documental e chunking", "2. Embeddings e índice ChromaDB",
           "3. Pipeline RAG com contexto", "4. Agente com ferramentas", "5. Interface Gradio",
           "6. Segurança e guardrails", "7. Avaliação com RAGAS", "8. Conclusão"]
RMS = ["572840", "569074", "569073"]


class Fonte:
    """Lê arquivos da pasta ou de dentro do ZIP com a mesma interface."""

    def __init__(self, alvo: Path):
        self.zip = zipfile.ZipFile(alvo) if alvo.suffix == ".zip" else None
        self.raiz = alvo
        if self.zip:
            nomes = self.zip.namelist()
            # aceita o conteúdo na raiz do ZIP ou dentro de uma pasta única
            prefixos = {n.split("/")[0] + "/" for n in nomes if "/" in n}
            self.prefixo = ""
            if not any(n in nomes for n in ITENS_ZIP if not n.endswith("/")) and len(prefixos) == 1:
                self.prefixo = prefixos.pop()

    def nomes(self) -> list:
        if self.zip:
            return [n[len(self.prefixo):] for n in self.zip.namelist() if n.startswith(self.prefixo)]
        return [str(p.relative_to(self.raiz)) + ("/" if p.is_dir() else "") for p in self.raiz.rglob("*")
                if ".git" not in p.parts and ".venv" not in p.parts]

    def existe(self, nome: str) -> bool:
        if nome.endswith("/"):
            return any(n.startswith(nome) and n != nome for n in self.nomes())
        return nome in self.nomes()

    def ler(self, nome: str) -> str:
        if self.zip:
            return self.zip.read(self.prefixo + nome).decode("utf-8", "ignore")
        return (self.raiz / nome).read_text(encoding="utf-8", errors="ignore")


def verificar(alvo: Path) -> list:
    f = Fonte(alvo)
    res = []

    def item(criterio, descricao, ok, evidencia=""):
        res.append((criterio, descricao, bool(ok), evidencia))

    # ---- Critério 1: entrega e organização ------------------------------------------------
    for nome in ITENS_ZIP:
        item(1, f"item do ZIP presente: {nome}", f.existe(nome))
    if f.zip:
        extras = sorted({n.split("/")[0] for n in f.nomes() if n and n.split("/")[0] + ("/" if "/" in n else "")
                         not in ITENS_ZIP})
        item(1, "ZIP só com os 7 itens (nada sobrando)", not extras, ", ".join(extras))
        item(1, "nome do ZIP no padrão GS2026_2_<TURMA>_<GRUPO>.zip",
             re.fullmatch(r"GS2026_2_1CCR_[A-Za-z0-9]+\.zip", alvo.name), alvo.name)
        item(1, "ZIP com menos de 50 MB", alvo.stat().st_size < 50 * 1024 * 1024, f"{alvo.stat().st_size/1e6:.1f} MB")

    nb = None
    if f.existe("1_Notebook_GS2026_2.ipynb"):
        nb = json.loads(f.ler("1_Notebook_GS2026_2.ipynb"))
        md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        saidas = []
        for c in nb["cells"]:
            for o in c.get("outputs", []):
                saidas.append(o.get("text", "") if isinstance(o.get("text"), str) else "".join(o.get("text", [])))
                for v in (o.get("data") or {}).values():
                    saidas.append(v if isinstance(v, str) else "".join(v))
                if o.get("output_type") == "error":
                    saidas.append("ERRO_DE_EXECUCAO " + o.get("ename", ""))
        saida = "\n".join(saidas)
        cabecalhos = re.findall(r"^## (.+)$", md, flags=re.M)
        for t in TITULOS:
            item(1, f"título exato da seção: '{t}'", t in cabecalhos)
        item(1, "títulos na ordem do briefing", [h for h in cabecalhos if h in TITULOS] == TITULOS)
        executadas = [c for c in nb["cells"] if c["cell_type"] == "code"]
        item(1, "notebook executado (todas as células de código com execution_count)",
             all(c.get("execution_count") for c in executadas), f"{sum(bool(c.get('execution_count')) for c in executadas)}/{len(executadas)}")
        item(1, "nenhuma célula com erro", "ERRO_DE_EXECUCAO" not in saida)
        item(1, "nenhuma saída com a palavra ERROR/Traceback", not re.search(r"\bERROR\b|Traceback", saida))
        item(1, "chave lida por userdata.get('OLLAMA_API_KEY')", "userdata.get('OLLAMA_API_KEY')" in codigo)
        item(1, "pip install com versões fixas (==) para langchain e chromadb",
             re.search(r"langchain==\d", codigo) and re.search(r"chromadb==\d", codigo))
        item(1, "sem API paga (sem openai/anthropic/gemini no código)",
             not re.search(r"langchain_openai|ChatOpenAI|anthropic|google\.generativeai|langchain_google", codigo))
        item(1, "modelo via Ollama Cloud (ChatOllama + ollama.com)", "ChatOllama" in codigo and "https://ollama.com" in codigo)
        for rm in RMS:
            item(1, f"RM {rm} no notebook", rm in md)
        item(1, "turma, modalidade e frente declaradas", all(x in md for x in ("1CCR", "Trio", "Mobilidade")))

        # ---- Critério 2: base e chunking --------------------------------------------------
        item(2, "metadados titulo, data_publicacao e tema carregados", all(x in codigo for x in ('"titulo"', '"data_publicacao"', '"tema"')))
        item(2, "2 estratégias de chunking comparadas (tabela com nº de chunks)", "nº de chunks" in codigo and "nº de chunks" in saida)
        just = re.search(r"### Escolha do chunking\n+(.+?)(\n## |\Z)", md, re.S)
        linhas_just = [l for l in (just.group(1).strip().splitlines() if just else []) if l.strip() and "<!--" not in l]
        item(2, "justificativa do chunking em 3–5 linhas", 3 <= len(linhas_just) <= 5, f"{len(linhas_just)} linhas")

        # ---- Critério 3: RAG ---------------------------------------------------------------
        item(3, "coleção ChromaDB com doc_id e data_publicacao", "Chroma" in codigo and "doc_id" in saida and "data_publicacao" in saida)
        item(3, "busca semântica com distance na saída", "distance" in saida)
        item(3, "pipeline LCEL (prompt | llm | parser)", re.search(r"prompt_rag \| llm \| StrOutputParser\(\)", codigo))
        item(3, "respostas com citação de doc_id na saída", len(re.findall(r"\[doc_\d{2}", saida)) >= 3)
        item(3, "comportamento de 'não sei' demonstrado", "Não há informação na base documental" in saida)

        # ---- Critério 4: agente ------------------------------------------------------------
        item(4, "≥ 2 tools com @tool e docstring", len(re.findall(r"@tool\s*\ndef \w+\(.*\) -> str:\n\s+\"\"\"", codigo)) >= 2)
        item(4, "1 tool que calcula (calcular_*)", "def calcular_proximo_horario" in codigo)
        item(4, "erro controlado demonstrado", "ERRO CONTROLADO" in saida)
        item(4, "limite de chamadas (max_iterations ≤ 6)", re.search(r"max_iterations=LIMITE_CHAMADAS", codigo) and "LIMITE_CHAMADAS = 6" in codigo)
        item(4, "trace thought → action → observation na saída", all(x in saida for x in ("thought", "action", "observation")))
        item(4, "pergunta composta com 2 tools (bônus)", "Trace 3 (pergunta composta)" in codigo)

        # ---- Critério 5: Gradio ------------------------------------------------------------
        item(5, "Gradio com histórico e painel de fontes", "gr.Chatbot" in codigo and "Fontes usadas" in codigo)
        item(5, "share=True no Colab", "share=EM_COLAB" in codigo)

        # ---- Critério 6: guardrails -------------------------------------------------------
        for g in ("escopo", "injection", "LGPD", "Limite de confiança"):
            item(6, f"guardrail documentado: {g}", g.lower() in md.lower())
        item(6, "bateria automatizada de sondas com assert", "assert acertos == len(SONDAS)" in codigo)
        m = re.search(r"Bateria de guardrails: (\d+)/(\d+)", saida)
        item(6, "bateria de sondas 100% na saída", m and m.group(1) == m.group(2), m.group(0) if m else "sem resultado")
        item(6, "injeção na fonte presente no documento e testada", "Ignore as instruções anteriores" in codigo)

        # ---- Critério 7: RAGAS ------------------------------------------------------------
        item(7, "juiz e embeddings explícitos (llm= e embeddings=)", "llm=juiz" in codigo and "embeddings=embeddings_juiz" in codigo)
        item(7, "4 métricas: faithfulness, answer_relevancy, context_precision, context_recall",
             all(x in codigo for x in ("Faithfulness()", "ResponseRelevancy()", "LLMContextPrecisionWithReference()", "LLMContextRecall()")))
        analise = re.search(r"### Leitura dos números\n+(.+?)(\n## |\Z)", md, re.S)
        item(7, "explicação dos números escrita", analise and "<!--" not in analise.group(1) and len(analise.group(1)) > 300)
        item(7, "melhoria concreta citando número", analise and re.search(r"melhoria", analise.group(1), re.I) and re.search(r"0,\d\d", analise.group(1)))

        # ---- Critério 8 (notebook): conclusão ---------------------------------------------
        conc = re.search(r"## 8\. Conclusão\n+(.+)", md, re.S)
        linhas_conc = [l for l in (conc.group(1).strip().splitlines() if conc else []) if l.strip()]
        item(8, "conclusão com ≤ 15 linhas", conc and 0 < len(linhas_conc) <= 15 and "<!--" not in conc.group(1), f"{len(linhas_conc)} linhas")
        item(8, "conclusão cita número medido", conc and re.search(r"0,\d\d", conc.group(1)))

    # ---- Base documental --------------------------------------------------------------------
    docs = sorted(n for n in f.nomes() if n.startswith("2_base_documental/") and n.endswith((".md", ".txt")))
    item(2, "≥ 2 documentos .md/.txt (trio: doc extra → ≥ 3)", len(docs) >= 3, f"{len(docs)} documentos")
    for d in docs:
        texto = f.ler(d)
        corpo = texto.split("---", 2)[2] if texto.startswith("---") else texto
        palavras = len(re.findall(r"\w+", corpo))
        meta_ok = all(re.search(rf"^{k}: .+", texto, re.M) for k in ("doc_id", "titulo", "data_publicacao", "tema"))
        item(2, f"{d.split('/')[-1]}: ≥ 100 palavras e metadados", palavras >= 100 and meta_ok, f"{palavras} palavras")

    # ---- RAGAS JSON ---------------------------------------------------------------------------
    if f.existe("3_avaliacao_ragas.json"):
        try:
            r = json.loads(f.ler("3_avaliacao_ragas.json"))
            final = r["por_pergunta"][r["configuracao_final"]]
            item(7, "JSON com 8 perguntas (4 nível 1 + 4 nível 2)", len(final) == 8 and sum(p["nivel"] == 1 for p in final) == 4)
            item(7, "JSON com as 4 métricas por pergunta e a média",
                 all(all(m in p for m in r["metricas"]) for p in final) and len(r["media_configuracao_final"]) == 4)
            item(7, "ground truth preenchido nas 8 perguntas", all(p["ground_truth"] for p in final))
            nans = sum(p[m] is None for p in final for m in r["metricas"])
            item(7, "nenhuma métrica NaN na configuração final", nans == 0, f"{nans} NaN")
        except Exception as e:
            item(7, "3_avaliacao_ragas.json válido", False, str(e))

    # ---- README -------------------------------------------------------------------------------
    if f.existe("4_README.md"):
        rd = f.ler("4_README.md")
        for trecho, desc in [("armadilha", "armadilha explicada"), ("diagrama", "diagrama do pipeline"),
                             ("==", "versões travadas"), ("custo", "custo aproximado"), ("limitaç", "limitações"),
                             ("decis", "decisões técnicas"), ("juiz", "juiz do RAGAS declarado"),
                             ("como executar", "instruções de execução")]:
            item(8, f"README: {desc}", trecho.lower() in rd.lower())
        imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", rd)
        item(8, "README com prints (≥ 5 imagens)", len(imgs) >= 5, f"{len(imgs)} imagens")
        for rm in RMS:
            item(8, f"README com RM {rm}", rm in rd)

    # ---- Certificação, chat log, vídeo -----------------------------------------------------------
    if f.existe("5_certificacao.txt"):
        c = f.ler("5_certificacao.txt")
        item(1, "certificação com nomes, RMs e links", all(rm in c for rm in RMS) and "http" in c)
    if f.existe("6_Chat_Log.txt"):
        item(1, "Chat Log com registro do uso de IA", len(f.ler("6_Chat_Log.txt")) > 800)
    if f.existe("7_demo_video_url.txt"):
        v = f.ler("7_demo_video_url.txt")
        item(1, "link do vídeo no YouTube", re.search(r"https://(www\.)?(youtube\.com|youtu\.be)/", v), v.strip()[:80])

    # ---- Segredos -----------------------------------------------------------------------------------
    padrao = re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{16,}\b")
    vazou = [n for n in f.nomes() if not n.endswith("/") and not n.startswith(".env") and padrao.search(f.ler(n))]
    item(1, "nenhuma chave de API em nenhum arquivo", not vazou, ", ".join(vazou))
    item(1, "nenhum .env na entrega", not any(Path(n).name.startswith(".env") for n in f.nomes()))
    return res


def main() -> int:
    alvo = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ
    resultado = verificar(alvo)
    falhas = [r for r in resultado if not r[2]]
    for crit, desc, ok, ev in resultado:
        print(f"{'OK ' if ok else 'FALHA'} [{crit}] {desc}" + (f"  ({ev})" if ev else ""))
    print(f"\n{len(resultado) - len(falhas)}/{len(resultado)} itens ok")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
