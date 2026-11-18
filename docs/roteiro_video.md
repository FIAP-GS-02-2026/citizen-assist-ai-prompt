# Roteiro do vídeo de Prompt and AI (até 3 min)

Exigência do briefing: "Link do vídeo (YouTube não listado), até 3 minutos, com cada integrante se apresentando no
início." Entrega individual: só o Glauco apresenta.
Meta: 2 min 45 s, para sobrar folga.
Tela: interface Gradio rodando no Colab (URL pública visível) e o notebook executado.

| Tempo | Tela | Fala |
|---|---|---|
| 0:00–0:12 | câmera ou tela inicial | "Olá, eu sou o Glauco Kelly, RM 572840, da turma 1CCR. Esta é a minha entrega individual da Global Solution de Prompt and AI." |
| 0:12–0:35 | diagrama do pipeline no README | "O projeto é o Citizen Assist AI, na frente de mobilidade urbana. O cidadão pergunta sobre a Linha 714 e sobre gratuidade no ônibus, e o assistente responde a partir de seis documentos do município, citando a fonte e dizendo quando não sabe." |
| 0:35–1:00 | Gradio: "O 714 passa no Jardim Primavera no domingo?" | "Esta é a pergunta com o detalhe escondido: a informação está no meio da ficha da linha. O assistente responde que aos domingos a linha não entra no bairro, cita o doc_01 e indica o ponto da escola." |
| 1:00–1:30 | Gradio: pergunta composta (mãe de 62 anos com cadeira de rodas, sábado, Praça da Matriz) | "Aqui o agente usa duas ferramentas na mesma resposta: uma calcula o direito à gratuidade pelo decreto, a outra consulta o ponto. O comunicado de novembro diz que o ponto está interditado, então ele indica o Terminal Central como alternativa acessível. Quem calcula é o código, não o modelo." |
| 1:30–2:00 | Gradio: receita de bolo → "ignore as instruções" → "quais moradores reclamaram do ponto" | "Os guardrails: fora do escopo, ele recusa e indica o canal oficial; tentativa de injeção, ele não obedece; pedido de dados de quem fez relato, ele recusa pela LGPD. No notebook, uma bateria automática testa 30 variações, inclusive perguntas legítimas que não podem ser bloqueadas." |
| 2:00–2:40 | notebook, seção 7: tabela do RAGAS | "Medi com RAGAS em 8 perguntas, 4 simples e 4 com armadilha, com a resposta certa escrita por mim. O contexto certo veio em todas as perguntas: context recall 1,0 e precision 0,875. A fidelidade ficou em 0,85. O pior caso foi a pergunta da senhora de 62 anos: a conclusão está certa, mas o modelo reescreveu a regra do decreto com as próprias palavras, e o juiz deu 0,33. A melhoria é citar a regra literal antes da conclusão, que é o que a ferramenta do agente já faz." |
| 2:40–2:50 | README | "Tudo roda no Colab com Run all, com modelo gratuito no Groq, e a chave fica nos Secrets. Obrigado!" |

**Checklist da gravação:** nenhuma chave na tela (Secrets fechado), nenhum dado pessoal, cronômetro abaixo de 3:00,
upload no YouTube como **Não listado**, link no `7_demo_video_url.txt`.
