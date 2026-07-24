"""
    python analise_trafego.py --csv contagens.csv
    python analise_trafego.py --exemplo  
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, LARANJA, CINZA, TEXTO = "#2f6fdd", "#e07b39", "#c9ced6", "#1f2328"


def dados_exemplo():
    rng = np.random.default_rng(7)
    janelas = ["06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]
    intensidade = [40, 95, 55, 62, 50, 70, 110, 58, 22]
    linhas = []
    for j, base in zip(janelas, intensidade):
        for _ in range(rng.poisson(base)):
            classe = rng.choice(["carro", "moto", "onibus", "caminhao"], p=[0.78, 0.16, 0.04, 0.02])
            linhas.append({"janela": j, "classe": classe,
                           "sentido": rng.choice(["subindo", "descendo"])})
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--exemplo", action="store_true")
    ap.add_argument("--minutos", type=float, default=10.0,
                    help="Duracao de cada gravacao, para normalizar em veiculos/minuto")
    ap.add_argument("--rua", default="minha rua")
    args = ap.parse_args()

    df = dados_exemplo() if args.exemplo else pd.read_csv(args.csv)

    total = df.groupby("janela").size().sort_index()
    taxa = total / args.minutos
    pico, vale = taxa.idxmax(), taxa.idxmin()

    print(f"Pior horario: {pico} ({taxa.max():.1f} veiculos/min)")
    print(f"Melhor horario: {vale} ({taxa.min():.1f} veiculos/min)")
    print(f"O pico tem {taxa.max()/taxa.min():.1f}x o volume do vale.")
    print("\nComposicao da frota (%):")
    print((df["classe"].value_counts(normalize=True) * 100).round(1).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})

    cores = [LARANJA if h == pico else AZUL for h in taxa.index]
    ax1.bar(taxa.index, taxa.values, color=cores, width=0.62)
    for h, v in taxa.items():
        ax1.text(h, v + taxa.max() * 0.02, f"{v:.1f}", ha="center", fontsize=9,
                 color=LARANJA if h == pico else TEXTO,
                 weight="bold" if h == pico else "normal")
    ax1.set_ylabel("Veículos por minuto", fontsize=11)
    ax1.set_title(f"Pico às {pico}: {taxa.max()/taxa.min():.1f}x o movimento do horário mais calmo",
                  fontsize=12, loc="left", pad=12)
    ax1.set_ylim(0, taxa.max() * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)

    comp = df["classe"].value_counts(normalize=True) * 100
    ax2.barh(comp.index[::-1], comp.values[::-1], color=CINZA, height=0.6)
    for i, v in enumerate(comp.values[::-1]):
        ax2.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=10, color=TEXTO)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Participação (%)", fontsize=11)
    ax2.set_title("Do que é feito esse trânsito", fontsize=12, loc="left", pad=12)
    ax2.spines[["top", "right"]].set_visible(False)

    titulo = f"Qual é o pior horário da {args.rua}?"
    if args.exemplo:
        titulo += "   [DADOS ILUSTRATIVOS - NÃO PUBLICAR]"
    fig.suptitle(titulo, fontsize=15, weight="bold", x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.015, "YOLOv8 + ByteTrack | contagem por cruzamento de linha",
             fontsize=9, color="#6b7280")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("grafico_trafego.png", dpi=200)
    print("\nGrafico salvo.")


if __name__ == "__main__":
    main()
