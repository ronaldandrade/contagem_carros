"""Prepara os dados para treino e validacao do detector YOLOv8.

python src/preparar_split_data2.py                    # temporal
    python src/preparar_split_data2.py --modo roboflow    # controle
"""

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NOMES = {0: "onibus", 1: "carro", 2: "moto", 3: "caminhao"}
SPLITS_ORIGEM = ["train", "valid", "test"]


def assinatura(caminho):
    """Miniatura cinza normalizada: robusta a recompressao, sensivel a conteudo."""
    a = np.asarray(Image.open(caminho).convert("L").resize((32, 18), Image.BILINEAR),
                   dtype=np.float32)
    return ((a - a.mean()) / (a.std() + 1e-6)).ravel()


def indexar_origem(frames_dir):
    """Assinatura de cada frame extraido, por (video, numero)."""
    chaves, vetores = [], []
    for vid in sorted(p for p in Path(frames_dir).iterdir() if p.is_dir()):
        for img in sorted(vid.glob("*.jpg")):
            m = re.match(r"frame_(\d+)", img.name)
            if m:
                chaves.append((vid.name, int(m.group(1))))
                vetores.append(assinatura(img))
    if not chaves:
        raise SystemExit(f"Nenhum frame de origem em {frames_dir}")
    return chaves, np.stack(vetores)


def casar(imagens, chaves, matriz):
    """Devolve {caminho_imagem: (video, numero_do_frame)}."""
    origem = {}
    for img in imagens:
        d = np.linalg.norm(matriz - assinatura(img), axis=1)
        origem[img] = chaves[int(d.argmin())]
    return origem


def caixas_do_rotulo(caminho_txt):
    """Le YOLO caixa (5 campos) ou poligono (impar >= 7) e devolve caixas cxcywh."""
    linhas = []
    if not Path(caminho_txt).exists():
        return linhas
    for linha in Path(caminho_txt).read_text().split("\n"):
        p = linha.split()
        if len(p) == 5:
            linhas.append((int(p[0]), *(float(x) for x in p[1:])))
        elif len(p) >= 7 and len(p) % 2 == 1:
            v = np.array([float(x) for x in p[1:]], dtype=float).reshape(-1, 2)
            x0, y0 = v.min(axis=0)
            x1, y1 = v.max(axis=0)
            linhas.append((int(p[0]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    # Descarta caixa degenerada: area zero nao e alvo de deteccao.
    return [l for l in linhas if l[3] > 1e-6 and l[4] > 1e-6]


def escrever(grupos, destino, nc):
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    for sub in ("train", "val"):
        (destino / "images" / sub).mkdir(parents=True)
        (destino / "labels" / sub).mkdir(parents=True)
    contagem = {}
    for sub, itens in grupos.items():
        c = Counter()
        for img, rot in itens:
            shutil.copy2(img, destino / "images" / sub / img.name)
            caixas = caixas_do_rotulo(rot)
            texto = "".join(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            for k, cx, cy, w, h in caixas)
            (destino / "labels" / sub / (img.stem + ".txt")).write_text(texto)
            for k, *_ in caixas:
                c[k] += 1
        contagem[sub] = c
    nomes = [NOMES[i] for i in range(nc)]
    (destino / "data.yaml").write_text(
        f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n\n"
        f"nc: {nc}\nnames: {nomes}\n")
    return contagem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["temporal", "roboflow"], default="temporal")
    ap.add_argument("--raiz", default="data/data2")
    ap.add_argument("--destino", default=None, help="padrao: data/splits2/<modo>")
    ap.add_argument("--fracao-val", type=float, default=0.2,
                    help="temporal: fracao final de cada video que vira validacao")
    ap.add_argument("--tampao", type=int, default=4,
                    help="temporal: frames descartados antes da validacao")
    ap.add_argument("--nc", type=int, default=4)
    args = ap.parse_args()

    raiz = Path(args.raiz)
    destino = Path(args.destino or f"data/splits2/{args.modo}")

    pares = []
    for s in SPLITS_ORIGEM:
        for img in sorted((raiz / "image" / s).glob("*.jpg")):
            pares.append((img, raiz / "labels" / s / (img.stem + ".txt"), s))

    # Os 60 frames do data1 sao reconhecidos pelo nome: o export preservou o
    # hash do Roboflow original. Ver ponto 4 no topo.
    nomes_data1 = set()
    d1 = Path("data/data1/frames")
    if d1.exists():
        nomes_data1 = {t.stem for t in d1.glob("*.jpg")}

    if args.modo == "roboflow":
        grupos = {"train": [(i, r) for i, r, s in pares if s == "train"],
                  "val": [(i, r) for i, r, s in pares if s in ("valid", "test")]}
        origem = None
    else:
        chaves, matriz = indexar_origem(raiz / "frames")
        origem = casar([i for i, _, _ in pares], chaves, matriz)

        # Fronteira por video: os ultimos `fracao_val` viram validacao.
        por_video = defaultdict(list)
        for v, n in chaves:
            por_video[v].append(n)
        corte = {v: max(ns) + 1 - round((max(ns) + 1) * args.fracao_val)
                 for v, ns in por_video.items()}

        grupos = {"train": [], "val": []}
        vistos = set()
        n_d1 = 0
        for img, rot, _ in pares:
            if img.stem in nomes_data1:
                # Numeracao incomparavel, linha do tempo conhecida: sempre treino.
                grupos["train"].append((img, rot))
                n_d1 += 1
                continue
            v, n = origem[img]
            if n >= corte[v]:
                # Validacao guarda UMA copia por frame: medir em copias
                # aumentadas do mesmo frame so repete o mesmo caso.
                if (v, n) not in vistos:
                    vistos.add((v, n))
                    grupos["val"].append((img, rot))
            elif n < corte[v] - args.tampao:
                grupos["train"].append((img, rot))

    contagem = escrever(grupos, destino, args.nc)

    print(f"Split '{args.modo}' -> {destino}/\n")
    for sub in ("train", "val"):
        c = contagem[sub]
        print(f"  {sub:5} : {len(grupos[sub]):3d} imagens  " +
              ", ".join(f"{c[k]:3d} {v}" for k, v in NOMES.items() if k < args.nc))

    if origem is not None:
        print()
        for v in sorted(corte):
            tr = sorted({origem[i][1] for i, _ in grupos["train"]
                         if origem[i][0] == v and i.stem not in nomes_data1})
            va = sorted({origem[i][1] for i, _ in grupos["val"] if origem[i][0] == v})
            if not va:
                continue
            d = min(abs(a - b) for a in va for b in tr) if tr else 0
            print(f"  {v}: treino ate {max(tr) if tr else '-'}, "
                  f"validacao {min(va)}-{max(va)} ({len(va)} frames), "
                  f"distancia minima {d} frame(s)")
        print(f"\n  Imagens do data1 fixadas no treino: {n_d1}")
        cruz = ({origem[i] for i, _ in grupos["train"] if i.stem not in nomes_data1}
                & {origem[i] for i, _ in grupos["val"]})
        print(f"\n  Frames de origem nos dois lados: {len(cruz)}")
    else:
        print("\n  Split do export: sorteado, com o mesmo frame e frames vizinhos")
        print("  dos dois lados da divisao. Braco de controle -- mede vazamento.")

    for k, v in NOMES.items():
        if k < args.nc and contagem["val"][k] == 0:
            print(f"  Sem {v} na validacao: classe nao avaliavel neste split.")


if __name__ == "__main__":
    main()
