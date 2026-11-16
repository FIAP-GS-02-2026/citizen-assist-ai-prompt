# Roteiro do vídeo de Prompt and AI (até 3 min)

Exigência do briefing: "Link do vídeo (YouTube não listado), até 3 minutos, com cada integrante se apresentando no início."
Meta: 2 min 45 s, para sobrar folga. Gravar na sessão única de quinta de manhã, com os três.
Tela: interface Gradio rodando no Colab (URL pública visível) e o notebook executado.

| Tempo | Quem | Tela | Fala |
|---|---|---|---|
| 0:00–0:10 | Glauco | câmera ou tela inicial | "Olá, eu sou o Glauco Kelly, RM 572840." |
| 0:10–0:18 | Gabriel | câmera | "Eu sou o Gabriel Fagundes, RM 569074." |
| 0:18–0:26 | Thiago | câmera | "E eu sou o Thiago Renatino, RM 569073. Somos a Equipe GGT, da turma 1CCR." |
| 0:26–0:45 | Glauco | diagrama do pipeline no README | "Nosso projeto é o Citizen Assist AI, na frente de mobilidade urbana. O cidadão pergunta sobre a Linha 714 e sobre gratuidade no ônibus, e o assistente responde a partir de seis documentos do município, citando a fonte e dizendo quando não sabe." |
| 0:45–1:10 | Glauco | Gradio: "O 714 passa no Jardim Primavera no domingo?" | "Essa é a pergunta com o detalhe escondido: a informação está no meio da ficha da linha. O assistente responde que aos domingos a linha não entra no bairro, cita o doc_01 e indica o ponto da escola." |
| 1:10–1:35 | Gabriel | Gradio: pergunta composta (mãe de 62 anos, sábado, Praça da Matriz) | "Aqui o agente usa duas ferramentas na mesma resposta: uma calcula o direito à gratuidade pelo decreto, a outra consulta o ponto. O comunicado de novembro diz que o ponto está interditado, então ele indica o Terminal Central como alternativa acessível. O modelo não inventa nada: quem calcula é o código." |
| 1:35–2:05 | Gabriel | Gradio: receita de bolo → injeção → "quais moradores relataram" | "Os guardrails: fora do escopo, ele recusa e indica o canal oficial; tentativa de injeção, ele não obedece; pedido de dados de quem fez relato, ele recusa pela LGPD. No notebook, uma bateria automática testa 24 variações dessas perguntas." |
| 2:05–2:40 | Thiago | notebook, seção 7: tabela do RAGAS | "Medimos com RAGAS em 8 perguntas, 4 simples e 4 com armadilha, com resposta certa escrita por nós. [ler os números finais da configuração escolhida]. [Explicar o número mais baixo e a melhoria proposta, como está na seção 7]." |
| 2:40–2:50 | Thiago | README | "Tudo roda no Colab com Run all, sem API paga, e a chave fica nos Secrets. Obrigado!" |

**Checklist da gravação:** nenhuma chave na tela (Secrets fechado), nenhum dado pessoal, cronômetro abaixo de 3:00,
upload no YouTube como **Não listado**, link no `7_demo_video_url.txt`.
