"""
    python src/avaliar_detector.py --modelo models/split_temporal.pt --classes custom
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deteccao import detectar, ler_gts, listar_imagens, mapas_de_classe
from nucleo_avaliacao import avaliar_classe, casar_por_imagem, metricas_por_limiar

AZUL, LARANJA, VERDE, CINZA = "#2f6fdd", "#e07b39", "#2e9e6b", "#c9ced6"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="models/split_temporal.pt")
    ap.add_argument("--imagens", default="data/splits/temporal/images/val")
    ap.add_argument("--labels", default="data/splits/temporal/labels/val")
    ap.add_argument("--classes", choices=["coco", "custom", "custom4"], default="custom",
                    help="'coco' para modelo pre-treinado; 'custom' para modelo "
                         "treinado neste dataset")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-min", type=float, default=0.001)
    ap.add_argument("--saida", default="outputs",
                    help="diretorio de saida; o nome dos arquivos vem do modelo")
    args = ap.parse_args()

    from ultralytics import YOLO

    CLASSES, mapa_gt = mapas_de_classe(args.classes)
    imagens = listar_imagens(args.imagens)

    modelo = YOLO(args.modelo)
    preds_por_img, tamanhos = detectar(modelo, imagens, CLASSES, args.conf_min)
    gts_por_img = ler_gts(tamanhos, args.labels, CLASSES, mapa_gt)

    resultados, marcas_todas, n_gt_todas = {}, [], 0
    for c, nome in CLASSES.items():
        resultados[nome] = avaliar_classe(preds_por_img[c], gts_por_img[c], args.iou)
        for preds, gts in zip(preds_por_img[c], gts_por_img[c]):
            marcas, n_gt = casar_por_imagem(preds, gts, args.iou)
            marcas_todas.extend(marcas)
            n_gt_todas += n_gt

    aps = {n: r["ap"] for n, r in resultados.items() if r["n_gt"] > 0}
    mAP = float(np.mean(list(aps.values()))) if aps else 0.0

    print(f"Modelo: {args.modelo}")
    print(f"Avaliacao em {len(imagens)} frames de {args.imagens} | IoU = {args.iou}\n")
    print(f"{'classe':10} {'AP':>7} {'n_GT':>6}")
    for nome, r in resultados.items():
        marca = "" if r["n_gt"] > 0 else "   (sem GT, nao avaliavel)"
        print(f"{nome:10} {r['ap']:7.3f} {r['n_gt']:6d}{marca}")
    print(f"\nmAP@{args.iou} = {mAP:.3f}  (media das {len(aps)} classes com GT)")

    limiares = np.round(np.arange(0.05, 0.96, 0.05), 2)
    df = pd.DataFrame(metricas_por_limiar(marcas_todas, n_gt_todas, limiares))
    melhor = df.loc[df["f1"].idxmax()]
    print(f"Melhor F1 = {melhor['f1']:.3f} no limiar {melhor['limiar']:.2f} "
          f"(precisao {melhor['precisao']:.2f}, recall {melhor['recall']:.2f})")

    # Nome derivado do modelo e do conjunto: rodar duas vezes com modelos
    # diferentes nao sobrescreve o resultado anterior nem exige renomear a mao.
    modelo_nome = Path(args.modelo).stem
    conjunto = Path(args.imagens).parent.parent.name
    # nao repete o nome quando modelo e conjunto coincidem (split_temporal + temporal)
    etiqueta = modelo_nome if conjunto in modelo_nome else f"{modelo_nome}_{conjunto}"
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    csv = saida / f"metricas_{etiqueta}.csv"
    png = saida / f"avaliacao_{etiqueta}.png"
    df.to_csv(csv, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    for (nome, r), cor in zip(resultados.items(), [AZUL, LARANJA, VERDE, "#8e44ad"]):
        if r["n_gt"] > 0 and len(r["recall"]):
            ax1.plot(r["recall"], r["precision"], "-", color=cor, lw=2,
                     label=f"{nome} (AP={r['ap']:.2f})")
    ax1.set_xlabel("Recall", fontsize=11); ax1.set_ylabel("Precisão", fontsize=11)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.set_title(f"Curva precisão-recall  |  mAP@{args.iou} = {mAP:.2f}",
                  fontsize=12, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(df["limiar"], df["precisao"], "-o", color=AZUL, lw=2, markersize=4, label="Precisão")
    ax2.plot(df["limiar"], df["recall"], "-o", color=LARANJA, lw=2, markersize=4, label="Recall")
    ax2.plot(df["limiar"], df["f1"], "-o", color=VERDE, lw=2.4, markersize=4, label="F1")
    ax2.axvline(melhor["limiar"], color=CINZA, ls="--", lw=1.4)
    ax2.set_xlabel("Limiar de confiança", fontsize=11); ax2.set_ylabel("Métrica", fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Melhor F1 em {melhor['limiar']:.2f}", fontsize=12, loc="left", pad=12)
    ax2.legend(frameon=False, fontsize=9, loc="lower center")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Avaliação de {Path(args.modelo).stem}", fontsize=15, weight="bold",
                 x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.015, f"{len(imagens)} frames | casamento por IoU ≥ {args.iou}",
             fontsize=9, color="#6b7280")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(png, dpi=200)
    print(f"\nTabela : {csv}\nGrafico: {png}")


if __name__ == "__main__":
    main()
