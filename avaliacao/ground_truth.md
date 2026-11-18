# Ground truth das 8 perguntas do RAGAS

**Quem preenche:** o grupo (Glauco, Gabriel e Thiago), à mão. O briefing proíbe pedir à IA o ground_truth;
se a verdade for gerada pelo modelo, o critério 7 cai para no máximo 1,0.

## Como preencher

1. Leia os documentos em `2_base_documental/` (doc_01 a doc_06).
2. Para cada pergunta, escreva a resposta correta com as suas palavras, em 1 a 3 frases, logo abaixo de
   "Resposta correta". Cite o que for preciso para a resposta ficar completa (datas, números, ponto alternativo).
3. Preencha "Escrito por" com o nome de quem escreveu.
4. Não apague as perguntas nem mude os títulos: o notebook lê este arquivo.

A IA só confere depois se a resposta de vocês bate com os documentos e aponta divergência, sem reescrever.

---

## P1 · nível 1 (simples)

**Pergunta:** Quais documentos uma pessoa de 70 anos precisa mostrar para andar de graça no ônibus municipal?

**Resposta correta:** Para utilizar a gratuidade, o beneficiário deve apresentar ao motorista ou validar no equipamento embarcado: documento oficial com foto ou o Cartão Aurora Sênior.

**Escrito por:** Glauco Kelly

---

## P2 · nível 1 (simples)

**Pergunta:** Onde e em que horário eu faço o cadastro para tirar o cartão de gratuidade?

**Resposta correta:** O cadastro é feito nos postos de atendimento do Consórcio Aurora Mobilidade, no Terminal Central, de segunda a sexta-feira, das 8h às 17h, sem necessidade de agendamento.

**Escrito por:** Glauco Kelly

---

## P3 · nível 1 (simples)

**Pergunta:** De quanto em quanto tempo passa o ônibus da Linha 714 em dia útil, e qual é a primeira partida do Terminal Central?

**Resposta correta:** Dias úteis (segunda a sexta-feira): primeira partida às 05h00 e última partida às 23h00, com intervalo regular de 20 minutos entre as viagens (05h00, 05h20, 05h40 e assim sucessivamente).

**Escrito por:** Glauco Kelly

---

## P4 · nível 1 (simples)

**Pergunta:** Minha mãe usa cadeira de rodas e não consegue pegar o ônibus comum. Como eu agendo o serviço porta a porta para ela?

**Resposta correta:** Pessoas com deficiência ou mobilidade reduzida severa que não conseguem usar o ônibus regular podem solicitar o serviço Aurora+ Porta a Porta, com veículo adaptado. O agendamento deve ser feito com pelo menos 48 horas de antecedência pela Central de Atendimento 156 ou pelo Portal do Cidadão, e exige cadastro prévio no Cartão Aurora Acesso.

**Escrito por:** Glauco Kelly

---

## P5 · nível 2 (armadilha: faixa condicional)

**Pergunta:** Minha mãe tem 62 anos. Ela pode andar de graça no ônibus numa terça-feira? E no sábado?

**Resposta correta:** Pessoas com idade entre 60 (sessenta) e 64 (sessenta e quatro) anos têm direito à gratuidade somente aos sábados, domingos e feriados, em caráter experimental, até 31 de dezembro de 2026.

**Escrito por:** Glauco Kelly

---

## P6 · nível 2 (armadilha: desatualização por portaria)

**Pergunta:** Ainda tem ônibus da 714 saindo do Terminal Central às 20h no domingo?

**Resposta correta:** Ficam suprimidas, a partir de 4 de outubro de 2026, as partidas das 07h30 e das 20h00 do Terminal Central nos domingos e feriados da Linha 714. A partida das 22h00 dos domingos e feriados passa a ocorrer às 21h30, a partir da mesma data.

**Escrito por:** Glauco Kelly

---

## P7 · nível 2 (armadilha: contradição entre documentos)

**Pergunta:** O ponto da Praça da Matriz está funcionando para quem usa cadeira de rodas? Se não, qual é o ponto acessível mais perto?

**Resposta correta:** O ponto de parada P-703, localizado na Praça da Matriz e utilizado pela Linha 714, está interditado para embarque e desembarque de passageiros de 5 de novembro a 5 de dezembro de 2026. Para quem precisa de ponto acessível na região central, a alternativa mais próxima é o ponto P-701, no Terminal Central (plataforma 3), a cerca de 400 metros da Praça da Matriz. No sentido bairro, também permanece acessível o ponto P-705, na UBS Acácias.

**Escrito por:** Glauco Kelly

---

## P8 · nível 2 (armadilha: detalhe que só existe no meio do documento)

**Pergunta:** O ônibus 714 passa no Jardim Primavera no domingo?

**Resposta correta:** Aos domingos e feriados, por causa da baixa demanda registrada na pesquisa de origem e destino, a linha não entra no Jardim Primavera: o ponto P-707 (Rua das Margaridas) não é atendido nesses dias, e o veículo segue direto da Escola Estadual Monte Verde para o ponto final. Moradores do Jardim Primavera devem utilizar o ponto P-706 aos domingos e feriados.

**Escrito por:** Glauco Kelly
