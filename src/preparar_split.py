"""Gera um split treino/validacao a partir da fonte unica (data/frames + data/labels).

Splits sao DERIVADOS, nao fonte: nada em data/splits/ e versionado, tudo e
reconstruido por este script. A fonte sao os 60 frames e os 60 arquivos de rotulo.

Dois modos:

  temporal  (padrao)  validacao = bloco CONTIGUO no fim do video, separada do
                      treino por uma zona-tampao de frames descartados.
  aleatorio           validacao sorteada, reproduzindo o split original -- serve
                      como braco de controle para MEDIR o vazamento, nao para
                      treinar um modelo que se leve a serio.

Por que o temporal e o correto: frames vizinhos do mesmo video mostram os MESMOS
veiculos quase na mesma posicao (58% das caixas tem IoU >= 0,5 com uma caixa do
frame anterior). Sortear frames coloca o mesmo veiculo nos dois lados da divisao --
no split original, os 12 frames de validacao tinham TODOS um vizinho imediato no
treino. A zona-tampao existe para que nenhum veiculo cruze a fronteira.

Uso:
    python src/preparar_split.py                       # temporal
    python src/preparar_split.py --modo aleatorio      # controle
"""

import argparse
import random
import re
import shutil
from collections import Counter
from pathlib import Path

NOMES = {0: "onibus", 1: "carro", 2: "moto"}


def coletar(frames_dir, labels_dir):
    """Indexa a fonte por numero de frame."""
    itens = {}
    for img in sorted(Path(frames_dir).glob("*.jpg")):
        m = re.match(r"frame_(\d+)", img.name)
        if not m:
            continue
        itens[int(m.group(1))] = (img, Path(labels_dir) / (img.stem + ".txt"))
    if not itens:
        raise SystemExit(f"Nenhum frame em {frames_dir}")
    return itens


def contar(itens, numeros):
    c = Counter()
    for n in numeros:
        txt = itens[n][1]
        if txt.exists():
            for l in txt.read_text().strip().splitlines():
                if l.strip():
                    c[int(l.split()[0])] += 1
    return c


def escrever(itens, train, val, destino):
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    for sub in ["train", "val"]:
        (destino / "images" / sub).mkdir(parents=True)
        (destino / "labels" / sub).mkdir(parents=True)
    for nums, sub in [(train, "train"), (val, "val")]:
        for n in nums:
            img, txt = itens[n]
            shutil.copy2(img, destino / "images" / sub / img.name)
            if txt.exists():
                shutil.copy2(txt, destino / "labels" / sub / txt.name)
    (destino / "data.yaml").write_text(
        f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n\n"
        f"nc: 3\nnames: ['bus', 'car', 'motocycle']\n")
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["temporal", "aleatorio"], default="temporal")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--destino", default=None,
                    help="padrao: data/splits/<modo>")
    ap.add_argument("--val-inicio", type=int, default=52,
                    help="temporal: primeiro frame do bloco de validacao")
    ap.add_argument("--tampao", type=int, default=4,
                    help="temporal: frames descartados antes da validacao")
    ap.add_argument("--fracao-val", type=float, default=0.2,
                    help="aleatorio: fracao dos frames para validacao")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    itens = coletar(args.frames, args.labels)
    ns = sorted(itens)
    destino = Path(args.destino or f"data/splits/{args.modo}")

    if args.modo == "temporal":
        val = [n for n in ns if n >= args.val_inicio]
        tampao = [n for n in ns if args.val_inicio - args.tampao <= n < args.val_inicio]
        train = [n for n in ns if n < args.val_inicio - args.tampao]
    else:
        emb = list(ns)
        random.Random(args.seed).shuffle(emb)
        corte = round(len(emb) * args.fracao_val)
        val, train, tampao = sorted(emb[:corte]), sorted(emb[corte:]), []

    escrever(itens, train, val, destino)
    ct, cv = contar(itens, train), contar(itens, val)

    print(f"Split '{args.modo}' -> {destino}/\n")
    print(f"  treino    : {len(train):2d} frames  " +
          ", ".join(f"{ct[k]:3d} {v}" for k, v in NOMES.items()))
    if tampao:
        print(f"  tampao    : {len(tampao):2d} frames descartados  {tampao}")
    print(f"  validacao : {len(val):2d} frames  " +
          ", ".join(f"{cv[k]:3d} {v}" for k, v in NOMES.items()))

    d = min(abs(v - t) for v in val for t in train)
    print(f"\n  Distancia minima entre validacao e treino: {d} frame(s)")
    if d <= 1:
        print("  ^ VAZAMENTO: frames vizinhos mostram os mesmos veiculos. Este split")
        print("    mede memorizacao, nao generalizacao. Use --modo temporal.")

    ausentes = [v for k, v in NOMES.items() if cv[k] == 0]
    if ausentes:
        print(f"\n  Sem exemplos de {', '.join(ausentes)} na validacao: essas classes")
        print(f"  nao sao avaliaveis. As 28 caixas de onibus sao 1 onibus parado nos")
        print(f"  frames 7-34 mais 1 no frame 47, e ha 1 unica caixa de moto no")
        print(f"  dataset inteiro -- nenhum split resolve isso.")


if __name__ == "__main__":
    main()
