# Peça pesada 02 — "Qual é o pior horário da minha rua?"

Contagem de veículos com YOLOv8 + ByteTrack. Uma gravação de dia util a noite (18:30).

---

## Etapa 1 — Coleta (o único trabalho de campo)

Gravei por **5min**, mantendo o mesmo ponto:

<!-- `06:00 · 08:00 · 10:00 · 12:00 · 14:00 · 16:00 · 18:00 · 20:00 · 22:00` -->

Regras que decidem se o dado presta:

- **Enquadramento idêntico** em todas as janelas. Marque com fita o lugar do tripé (ou do celular apoiado). Se o ângulo muda, a comparação entre horários morre.
- **Ângulo alto e oblíquo** (janela de andar alto é ideal). Vista de topo evita oclusão — carro atrás de carro é o que mais estraga contagem.
- **A linha virtual deve atravessar a via inteira**, num trecho onde os carros passam sem parar. Se cruzar em cima de um semáforo, o carro parado em cima da linha oscila e é contado duas vezes.
- **Mesmo dia da semana.** Terça e domingo são ruas diferentes. Se quiser comparar dia útil x fim de semana, isso é uma *segunda* peça, não a mesma.
- **Grave em 1080p, 30fps.** 4K só deixa a inferência lenta sem melhorar a contagem.

**Privacidade (leve a sério):** filmar via pública é legal, mas rosto e placa são dado pessoal sob a LGPD. Enquadre de longe e de cima, de forma que placas não sejam legíveis. No vídeo publicado, aplique blur nas placas se alguma aparecer. Isso não é só conformidade — é um detalhe que sinaliza profissionalismo, e você pode transformá-lo em conteúdo ("como eu anonimizei o dataset").

## Etapa 2 — Processamento

```bash
pip install ultralytics opencv-python pandas matplotlib

python contagem_carros.py --video rua_06h.mp4 --janela "06:00"
python contagem_carros.py --video rua_08h.mp4 --janela "08:00" --preview
# ... uma vez por janela; tudo se acumula em contagens.csv

python analise_trafego.py --csv contagens.csv --rua "Rua X"
```

O `--preview` abre a janela anotada com as caixas, os IDs e o contador subindo. **Grave a tela nesse momento** — essa é a imagem que vende o vídeo inteiro.

Antes de gravar de verdade, rode `python analise_trafego.py --exemplo` para ver o layout do gráfico com dados sintéticos.

## Etapa 3 — Validação (o passo que quase todo mundo pula)

Conte **uma janela na mão**, assistindo ao vídeo. Compare com o modelo.

Se o YOLO contou 87 e você contou 92, o modelo tem ~5% de erro — e esse número é ouro. É a diferença entre "fiz um projeto de CV" e "sei avaliar um sistema de CV". Nenhum canal de dica rápida faz isso.

---

## Roteiro do vídeo âncora (75s, vertical)

| Tempo | Fala | Tela |
|---|---|---|
| 0–6s | "Moro numa avenida muito movimentada e bem barulhenta. Uma vez pensando em quantos carros passam aqui durante o dia, resolvi procurar uma maneira de medir." | Vista da janela, rua com movimento |
| 6–15s | "Coloquei o celular na janela, gravei 10 minutos a cada 2 horas e joguei tudo num detector de objetos." | Time-lapse do setup |
| 15–30s | "Detectar carro é a parte fácil. O problema é não contar o mesmo carro 30 vezes — ele aparece em 30 quadros. Por isso não basta detecção: precisa de rastreamento, que dá uma identidade a cada veículo e a mantém enquanto ele atravessa a tela." | Preview rodando: caixas, IDs, contador subindo |
| 30–42s | "Aí eu desenho uma linha virtual. No frame em que o centro do carro troca de lado da linha, ele é contado. Uma vez. E eu ainda sei o sentido." | Zoom na linha, um carro cruzando |
| 42–58s | "Resultado: [X] veículos por minuto no pico contra [Y] no vale. E o pior horário não é [o que todo mundo acha] — é [horário real]." | Gráfico de barras, pico em laranja |
| 58–70s | "Conferi contando uma janela na mão: o modelo errou [Z]%. Sistema de visão computacional que ninguém validou é enfeite." | Comparação manual x modelo |
| 70–78s | "Código completo no GitHub. Roda na sua rua também." | Card do repositório |

**Se o resultado confirmar o senso comum**, não jogue fora — mude o gancho: "Dessa vez o achismo estava certo. Mas eu só sei disso porque medi."

---

## Atomização: 5 peças curtas da mesma gravação

Uma gravação de sábado alimenta a semana inteira.

1. **Segunda — "Detecção não é contagem"** (45s). Mostra o erro de contar 30x o mesmo carro. Conceito puro, alto valor educacional.
2. **Terça — "O que é rastreamento"** (60s). ByteTrack explicado com o ID pulando na tela.
3. **Quarta — O resultado** (75s). O vídeo âncora acima.
4. **Quinta — "Onde o modelo errou"** (60s). Oclusão, moto detectada como bicicleta, caminhão parcialmente fora do quadro. Vídeo de erro engaja mais que vídeo de acerto.
5. **Sexta — "Rode na sua rua"** (45s). Tutorial de 3 comandos, chamada pro GitHub. Peça de conversão.

---

## Textos de distribuição

**Legenda (Reels/TikTok/Shorts):**

> Medi o trânsito da minha rua com visão computacional em vez de achismo.
>
> 10 minutos de gravação a cada 2 horas, YOLOv8 pra detectar e ByteTrack pra rastrear. A parte difícil não é reconhecer o carro — é não contar o mesmo carro 30 vezes.
>
> Resultado: [X] veículos/min no pico contra [Y] no horário mais calmo.
>
> Código no GitHub, roda na sua rua. Link na bio.
>
> #visaocomputacional #python #yolo #machinelearning #dados

**Título do blog (busca + assistentes de IA):**
Como contar carros em um vídeo com YOLOv8 e ByteTrack em Python

**Primeira linha (resposta direta, antes da introdução):**
Para contar veículos em um vídeo sem duplicar contagens, combine um detector (YOLOv8) com um rastreador (ByteTrack) para atribuir um ID persistente a cada veículo, defina uma linha virtual sobre a via e registre o veículo apenas no frame em que o centro de sua caixa troca de lado em relação a essa linha.

---

## Desdobramentos (a série não acaba aqui)

- Dia útil x domingo, mesma rua.
- Chove: o volume muda?
- Proporção de motos por horário — muda muito, e quase ninguém sabe disso.
- Estimativa de velocidade por homografia (calibrando com o comprimento de uma faixa da via). Aqui você entra em território que praticamente nenhum criador brasileiro cobre.
