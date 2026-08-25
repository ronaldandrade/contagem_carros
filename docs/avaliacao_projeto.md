# Avaliação do projeto `contagem_carros`

Revisão técnica do repositório no estado do commit `a13489a` (19/08/2026), cobrindo
código, dados, reprodutibilidade e a coerência entre o que a documentação afirma e o
que o código faz.

**Resumo:** o projeto é honesto e bem escrito onde importa mais — a matemática da
avaliação está correta, isolada e testada, e a análise crítica em `docs/avaliacao.md`
sobre vazamento treino/validação é o ponto alto do trabalho. Os problemas se
concentram em outro lugar: a etapa de **contagem** (o produto final anunciado) está
desconectada da etapa de **avaliação**, o treino não é reproduzível a partir do
repositório, e vários caminhos de saída são inconsistentes.

---

## 1. Pontos fortes

- **Separação da matemática.** `src/nucleo_avaliacao.py` não importa YOLO, não lê
  arquivos e não desenha nada. É só numpy. Isso permite o autoteste de
  `_autoteste()` (`nucleo_avaliacao.py:87`), que confere AP em três casos verificáveis
  à mão. É a decisão de arquitetura mais acertada do projeto.
- **AP implementado corretamente.** `average_precision` (`nucleo_avaliacao.py:53`) faz
  a interpolação all-points com envoltória monótona e integra só nos pontos de
  mudança de recall — é a definição padrão (COCO), não uma aproximação de 11 pontos.
- **Casamento predição↔GT correto.** `casar_por_imagem` (`nucleo_avaliacao.py:18`)
  ordena por confiança e consome o GT casado (`gt_usado`), penalizando detecções
  duplicadas como FP. Comportamento certo, e implementado sem gambiarra.
- **Análise crítica dos próprios resultados.** `docs/avaliacao.md` identifica o
  vazamento entre treino e validação por frames vizinhos, demonstra o efeito medindo
  nos dois conjuntos, e recusa a leitura fácil do AP 1,000 em ônibus. Isso vale mais
  do que o modelo em si.
- **Números conferem.** Verifiquei contra os artefatos: `outputs/contagens.csv` tem
  exatamente 125 linhas (58 em 13:00, 67 em 16:30; 101 carros, 13 motos, 8 caminhões,
  3 ônibus) e os melhores F1 das quatro tabelas em `outputs/` são 0,154 / 0,065 /
  0,943 / 0,921 — todos batem com o README. O mAP@0,5 de 0,660 bate com o
  `results.csv` do treino (0,66082). A documentação não inflaciona resultado.
- **Linha de contagem em coordenada relativa** (`contagem_carros.py:17`), o que torna
  a contagem independente da resolução.

---

## 2. Problemas graves

### 2.1 O contador não consegue usar o modelo treinado

`contagem_carros.py:14` fixa `VEICULOS = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}`,
que são IDs do COCO, e passa `classes=list(VEICULOS)` para o `track` (`contagem_carros.py:52`).
O modelo treinado neste dataset tem `nc: 3` com IDs 0, 1, 2. Não existe flag
`--classes` aqui (só em `avaliar_detector.py:75`). Consequência: rodar
`contagem_carros.py --modelo runs/.../best.pt` filtra por classes que não existem no
modelo — não conta nada útil, e se algo passar, `VEICULOS[cls]` levanta `KeyError`.

Ou seja: o modelo que o projeto treinou, e que é o assunto de metade da documentação,
**não é utilizável na função principal do projeto**. Os 125 veículos contados foram
produzidos pelo `yolov8n.pt` genérico (o default em `contagem_carros.py:29`) — o mesmo
modelo que a própria avaliação mede com melhor F1 de **0,154** nesta cena.

Isso merece estar escrito no README: a contagem publicada vem do detector cuja
qualidade medida é a pior das duas.

### 2.2 A avaliação não calibra o contador, apesar de a documentação dizer que sim

`docs/avaliacao.md` afirma que o limiar de melhor F1 "é o que você deveria passar ao
`contagem_carros.py`" e o README diz que "a avaliação não serve só para dar uma nota,
ela também ajusta o contador". Não há nenhum código ligando as duas coisas:
`avaliar_detector.py` imprime o melhor limiar, e `contagem_carros.py:30` mantém
`--conf 0.35` fixo como default. Os limiares ótimos medidos foram 0,05 (genérico) e
0,40 / 0,50 (treinado) — nenhum deles é 0,35. A afirmação é uma intenção, não um
comportamento do sistema.

### 2.3 O treino não é reproduzível a partir do repositório

- Não existe script de treino versionado. O treino existe apenas como
  `runs/detect/runs/treino_trafego/args.yaml` (yolov8n, 60 epochs, batch 8, imgsz 640,
  cpu, seed 0). Os comandos do README cobrem extração, contagem, análise e avaliação —
  o treino, que é o experimento central, não tem comando documentado.
- `.gitignore` tem a regra `data.yaml`, que casa também com
  `data/yolo_dataset/data.yaml`. O arquivo que define o dataset de treino está fora do
  versionamento; quem clonar não consegue treinar nem sabe a divisão de classes.
- `.gitignore` tem `*.pt`, então `best.pt` não está versionado; e o diretório
  `runs/.../weights/` inteiro aparece como não rastreado (inclui `best.onnx` e
  `best.fp16.onnx`, que **não** são ignorados e simplesmente nunca foram commitados).
  O comando de avaliação do modelo treinado no README não roda em um clone limpo.
- `.gitignore` tem `frames_*/`, então `data/frames_yolo/` (as 60 imagens) não está
  versionado — mas `data/labels/` está. O repositório guarda as anotações sem as
  imagens correspondentes. As mesmas imagens existem versionadas em
  `data/yolo_dataset/images/{train,val}/`, o que torna a situação mais confusa: o
  comando `--imagens data/frames_yolo` do README aponta para o único diretório que o
  clone não terá.
- `*.csv` ignorado exclui `outputs/metricas_por_limiar_*.csv` e `outputs/contagens.csv`
  — os quatro gráficos `.png` estão versionados, as tabelas que os originam não.

### 2.4 Label ausente é indistinguível de frame sem veículo

`ler_gt_yolo` (`avaliar_detector.py:27`) retorna caixas vazias silenciosamente quando o
`.txt` não existe. Um erro de caminho em `--labels`, ou um único arquivo faltando, não
gera aviso: todas as detecções daquele frame viram falsos positivos e a métrica
despenca sem explicação. Como o casamento de nomes depende de `img.stem + ".txt"` e os
nomes carregam hashes do Roboflow, é exatamente o tipo de erro fácil de cometer.
Falta um contador de "N frames sem arquivo de label" no relatório.

---

## 3. Problemas médios

### 3.1 Saídas em lugares inconsistentes

| Script | Escreve onde | Deveria |
|---|---|---|
| `avaliar_detector.py:122` | `metricas_por_limiar.csv` no diretório atual | `outputs/` |
| `avaliar_detector.py:159` | `outputs/grafico_avaliacao_detector.png` | ok |
| `analise_trafego.py:81` | `grafico_trafego.png` no diretório atual | `outputs/` |
| `contagem_carros.py:28` | `contagens.csv` no diretório atual | `outputs/` |

O mesmo script grava metade em `outputs/` e metade na raiz. Além disso, os nomes são
fixos: cada execução sobrescreve a anterior — `docs/avaliacao.md` chega a instruir o
usuário a renomear os arquivos manualmente entre execuções. Os quatro pares de
arquivos em `outputs/` (`_generico`, `_generico_val`, `_treinado`, `_treinado_val`)
foram renomeados à mão. Uma flag `--saida`/`--tag` resolve.

### 3.2 O melhor F1 do modelo genérico está na borda da grade

`avaliar_detector.py:119` varre limiares de 0,05 a 0,95. O melhor F1 do modelo genérico
cai em **0,05**, o primeiro ponto da grade, enquanto a inferência roda com
`--conf-min 0.001`. O ótimo real pode estar abaixo da grade e não foi medido. Vale
estender a varredura para baixo (ou avisar quando o ótimo cair na borda).

### 3.3 Trabalho duplicado no laço de avaliação

Em `avaliar_detector.py:101-107`, `avaliar_classe` já executa `casar_por_imagem` para
cada imagem, e o laço logo abaixo repete exatamente o mesmo casamento só para agregar
`marcas_todas`. É o dobro do trabalho. `avaliar_classe` poderia devolver as marcas que
já calculou.

### 3.4 `analise_trafego.py` assume duração igual para todas as janelas

`--minutos` é um único valor aplicado a todas as gravações (`analise_trafego.py:42`).
Com gravações de durações diferentes, a normalização "veículos/minuto" fica errada
sem nenhum aviso. A duração deveria vir do próprio vídeo (o `contagem_carros.py` já a
conhece) e ser gravada no CSV por janela.

Além disso, com apenas duas janelas medidas, "pior horário" e "melhor horário"
(`analise_trafego.py:45-47`) são o máximo e o mínimo de dois pontos — a conclusão de
pico/vale que o gráfico apresenta não se sustenta com essa amostra, e o README a
apresenta sem essa ressalva.

### 3.5 `contagem_carros.py` acumula sem proteção

`contagem_carros.py:95` abre o CSV em modo append. Rodar duas vezes a mesma janela
duplica silenciosamente as contagens no arquivo — e é exatamente o fluxo esperado
(uma execução por gravação). Não há verificação de janela já processada.

---

## 4. Problemas menores

- **Sentido invertido em relação à documentação.** `contagem_carros.py:78` decide o
  sentido pelo lado **novo** (`lado > 0` → "descendo"), enquanto o README diz que "o
  sentido vem justamente de qual lado ele estava antes". Os rótulos saem coerentes
  entre si, mas a descrição está trocada em relação ao código.
- **Cruzamento exatamente sobre a linha é perdido.** O teste `anterior * lado < 0`
  (`contagem_carros.py:72`) ignora o caso `lado == 0`. Raro em float, mas gratuito de
  corrigir.
- **`lado_anterior` cresce sem limite** (`contagem_carros.py:45`) — um dicionário por
  ID de track, nunca podado. Irrelevante em 5 minutos de vídeo, problema real em
  operação contínua.
- **Linha de contagem fixa em `0.78`** (`contagem_carros.py:17`) sem flag de linha de
  comando. Trocar de cena exige editar o código.
- **Erro de digitação propagado no dataset:** a classe é `motocycle` (`data.yaml`),
  não `motorcycle`. Está gravado nos pesos treinados e nos caches.
- **`__pycache__/` na raiz contém módulos órfãos** (`gerar_video_tiktok`,
  `extrair_frames_exemplo`) de arquivos que não existem mais. Lixo rastreável de
  versões antigas; convém apagar o diretório.
- **`requirements.txt` não fixa `torch`**, que é a dependência mais pesada e a que
  mais quebra entre versões — vem implicitamente via `ultralytics`.
- **Sem LICENSE**, apesar de o `data.yaml` registrar o dataset como CC BY 4.0.
- **Sem `tests/` nem CI.** O autoteste do núcleo é bom, mas só roda se alguém lembrar
  de executar `python src/nucleo_avaliacao.py`. Migrar para `pytest` e rodar no CI
  custa pouco e protege o único código que realmente precisa estar certo.
- **Alterações não commitadas:** `docs/avaliacao.md` tem 4 linhas modificadas (troca de
  travessões por vírgulas) e `runs/.../weights/` está não rastreado.

---

## 5. Recomendações, em ordem de retorno

1. **Ligar avaliação e contagem.** Adicionar `--classes {coco,custom}` ao
   `contagem_carros.py` (extraindo o mapa de classes para um módulo comum com
   `avaliar_detector.py`) e permitir passar o limiar escolhido pela avaliação. Sem
   isso, o modelo treinado é um experimento morto.
2. **Recontar com o modelo treinado** e comparar com os 125 do modelo genérico. Essa
   comparação é o resultado mais interessante que o projeto ainda não tem — e é a
   demonstração prática do que a avaliação vinha dizendo.
3. **Versionar o treino:** um `src/treinar.py` (ou os comandos exatos no README), mais
   `data/yolo_dataset/data.yaml` fora do `.gitignore` (trocar a regra `data.yaml` por
   `/data.yaml`).
4. **Rever o `.gitignore` inteiro.** Hoje ele exclui as imagens de `frames_yolo/`, as
   tabelas de métricas e os pesos, deixando o repositório com comandos de README que
   não rodam num clone limpo. Decidir explicitamente: dados de entrada versionados ou
   não, e artefatos de saída versionados ou não.
5. **Padronizar saídas** com `--saida` e um prefixo por execução, tudo sob `outputs/`.
6. **Avisar sobre labels ausentes** em `ler_gt_yolo`/`main`.
7. **Refazer a divisão treino/validação por tempo** (primeiros 80% / últimos 20% do
   vídeo). `docs/avaliacao.md` já prescreve a correção mas não a aplica — aplicá-la e
   publicar o número honesto, mesmo que pior, fecha o argumento do estudo.
8. **Repetir a avaliação com `--iou 0.75`**, também já prescrito e não executado.

---

## 6. Veredito

Como estudo sobre avaliação de detectores, é um trabalho acima da média: mede certo,
desconfia do próprio resultado, e a investigação do vazamento é genuinamente útil.
Como sistema de contagem de veículos, está incompleto — o pipeline se parte no meio,
entre o modelo que foi treinado e avaliado e o contador que efetivamente produziu os
números publicados. As correções do item 5 são a diferença entre "dois scripts que
compartilham uma pasta" e "um pipeline".
