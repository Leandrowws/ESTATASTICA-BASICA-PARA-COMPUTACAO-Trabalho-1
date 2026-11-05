"""
Estatística Básica para Computação (T1) - Trabalho Extraclasse, Parte 1
Aluno: Leandro Castro Delevati

Variáveis:
    i : número do corpo de prova
    A : Teor de fibra (% de massa)            -> 10, 20, 30
    B : Temperatura de extrusão (°C)          -> 210, 230, 250
    C : Velocidade da rosca (RPM)             -> 100, 150
    X : Resistência à tração (MPa)
    Y : Elasticidade (GPa)
"""
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Matriz de dados: cada linha = [i, A, B, C, X, Y]
# ---------------------------------------------------------------------------
dados = [
    # i,  A,   B,   C,    X,    Y
    [1,  10, 210, 100,  45.2,  3.2],
    [2,  10, 210, 100,  44.8,  3.1],
    [3,  10, 210, 150,  47.1,  3.4],
    [4,  10, 230, 100,  49.5,  3.6],
    [5,  10, 230, 100,  50.2,  3.8],
    [6,  10, 230, 150,  52.3,  4.1],
    [7,  10, 250, 100,  41.0,  2.9],
    [8,  10, 250, 100,  40.5,  2.8],
    [9,  10, 250, 150,  42.8,  3.0],
    [10, 20, 210, 100,  65.4,  5.5],
    [11, 20, 210, 100,  66.1,  5.7],
    [12, 20, 210, 150,  68.7,  6.0],
    [13, 20, 230, 100,  72.3,  6.8],
    [14, 20, 230, 100,  73.0,  6.9],
    [15, 20, 230, 150,  75.8,  7.2],
    [16, 20, 250, 100,  58.2,  4.9],
    [17, 20, 250, 100,  57.9,  4.8],
    [18, 20, 250, 150,  60.1,  5.1],
    [19, 30, 210, 100,  88.5,  8.8],
    [20, 30, 210, 100,  89.2,  9.0],
    [21, 30, 210, 150,  91.0,  9.3],
    [22, 30, 230, 100,  95.4, 10.2],
    [23, 30, 230, 100,  96.1, 10.5],
    [24, 30, 230, 150,  98.7, 11.0],
    [25, 30, 250, 100,  78.3,  7.5],
    [26, 30, 250, 100,  77.5,  7.3],
    [27, 30, 250, 150,  80.2,  7.8],
    [28, 10, 210, 150,  46.5,  3.3],
    [29, 10, 230, 150,  51.8,  4.0],
    [30, 10, 250, 150,  43.1,  3.1],
    [31, 20, 210, 150,  67.9,  5.9],
    [32, 20, 230, 150,  76.2,  7.4],
    [33, 20, 250, 150,  59.5,  5.0],
    [34, 30, 210, 150,  90.3,  9.2],
    [35, 30, 230, 150,  99.1, 11.2],
    [36, 30, 250, 150,  79.8,  7.7],
    [37, 10, 230, 100,  49.9,  3.7],
    [38, 20, 230, 100,  72.8,  6.7],
    [39, 30, 230, 100,  95.8, 10.3],
    [40, 10, 210, 100,  45.0,  3.2],
    [41, 20, 210, 100,  65.8,  5.6],
    [42, 30, 210, 100,  88.9,  8.9],
    [43, 10, 250, 100,  41.2,  2.9],
    [44, 20, 250, 100,  58.0,  4.9],
    [45, 30, 250, 100,  78.0,  7.4],
]

matriz = np.array(dados)
A = matriz[:, 1].astype(int)   # Teor de fibra (%)
B = matriz[:, 2].astype(int)   # Temperatura de extrusão (°C)
C = matriz[:, 3].astype(int)   # Velocidade da rosca (RPM)
X = matriz[:, 4]               # Resistência à tração (MPa)
Y = matriz[:, 5]               # Elasticidade (GPa)
df = pd.DataFrame({"A": A, "B": B, "C": C, "X": X, "Y": Y})

NBSP = "\u00a0"                # espaço que não quebra linha (usado em "n = 45")


# ===========================================================================
# FUNÇÕES AUXILIARES
# ===========================================================================
def fmt(v, casas=2):
    """Número no padrão brasileiro (vírgula decimal)."""
    return ("%.*f" % (casas, v)).replace(".", ",")


def banner(texto):
    print("=" * 70)
    print(texto)
    print("=" * 70)


def tabela_como_imagem(arquivo, titulo, cabecalho, linhas, larguras, alinhamento,
                       grupos=None, linha_total=False,
                       fonte="Fonte: Elaborada pelo autor.", notas=None, tam_fonte=10):
    """
    Imprime a tabela no terminal e a salva como imagem, seguindo as normas de
    apresentação tabular (IBGE/ABNT):
      - título acima, numerado ("Tabela N – ...");
      - traço grosso no topo e na base; traço fino sob o cabeçalho e acima do total;
      - sem traços verticais nas laterais (tabela aberta);
      - unidades no cabeçalho ou junto ao valor; decimais com vírgula alinhados;
      - fonte e notas abaixo da tabela.

    alinhamento : lista com 'l' (esquerda), 'c' (centro) ou 'v' (valor, alinhado pela vírgula)
    grupos      : lista de (texto, coluna_inicial, coluna_final) para títulos que abrangem colunas
    linha_total : True se a última linha é de total/marginal (recebe traço fino acima)
    """
    # --- terminal -------------------------------------------------------------
    print("\n" + titulo.replace(NBSP, " "))
    print(pd.DataFrame(linhas, columns=[c.replace("\n", " ") for c in cabecalho])
          .to_string(index=False))

    # --- geometria (polegadas) --------------------------------------------------
    LH, PAD, MARGEM = 0.19, 0.15, 0.35          # altura da linha de texto, folga, margem
    GROSSO, FINO = 1.8, 0.7                     # espessuras dos traços
    grupos = grupos or []
    ncol = len(cabecalho)
    larg_tab = sum(larguras)
    xs = [MARGEM + sum(larguras[:j]) for j in range(ncol + 1)]   # bordas das colunas

    cpl = int(larg_tab / (tam_fonte * 0.0088))            # caracteres por linha (aprox.)
    cpl_rod = int(larg_tab / ((tam_fonte - 1) * 0.0088))
    titulo_q = textwrap.fill(titulo, cpl)
    rodape = "\n".join(textwrap.fill(t, cpl_rod) for t in [fonte] + (notas or []))

    em_grupo = {j for _, a, b in grupos for j in range(a, b + 1)}
    n_lin = lambda cols: max([cabecalho[j].count("\n") + 1 for j in cols] or [1])
    h_tit = (titulo_q.count("\n") + 1) * LH + 0.12
    h1 = (LH + PAD) if grupos else 0                       # faixa dos títulos agrupados
    h2 = max(n_lin(em_grupo) * LH + PAD,
             n_lin(set(range(ncol)) - em_grupo) * LH + PAD - h1)
    h_cab = h1 + h2
    h_lin = LH + PAD
    h_corpo = len(linhas) * h_lin
    h_rod = (rodape.count("\n") + 1) * LH * 0.92 + 0.15
    W = larg_tab + 2 * MARGEM
    H = MARGEM + h_tit + h_cab + h_corpo + h_rod + MARGEM

    y_t = H - MARGEM - h_tit            # topo da tabela
    y_cab = y_t - h_cab                 # fim do cabeçalho
    y_b = y_cab - h_corpo               # base da tabela

    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def traco(x1, x2, y1, y2, esp):
        ax.plot([x1, x2], [y1, y2], color="black", lw=esp, solid_capstyle="butt")

    def texto(x, y, s, ha="center", **kw):
        ax.text(x, y, s, ha=ha, va="center", fontsize=tam_fonte, **kw)

    # --- título e traços horizontais -------------------------------------------
    ax.text(MARGEM, H - MARGEM, titulo_q, ha="left", va="top", fontsize=tam_fonte)
    traco(xs[0], xs[-1], y_t, y_t, GROSSO)              # topo
    traco(xs[0], xs[-1], y_cab, y_cab, FINO)            # sob o cabeçalho
    traco(xs[0], xs[-1], y_b, y_b, GROSSO)              # base
    if linha_total:
        y_tot = y_cab - (len(linhas) - 1) * h_lin
        traco(xs[0], xs[-1], y_tot, y_tot, FINO)

    # --- traços verticais internos ---------------------------------------------
    for j in range(1, ncol):
        dentro = any(a < j <= b for _, a, b in grupos)
        traco(xs[j], xs[j], (y_t - h1) if dentro else y_t, y_b, FINO)

    # --- cabeçalho -------------------------------------------------------------
    cab_kw = dict(fontweight="bold", linespacing=1.15)
    for txt, a, b in grupos:
        texto((xs[a] + xs[b + 1]) / 2, y_t - h1 / 2, txt, fontweight="bold")
        traco(xs[a] + 0.08, xs[b + 1] - 0.08, y_t - h1, y_t - h1, FINO)
    for j in range(ncol):
        yc = (y_t - h1 - h2 / 2) if j in em_grupo else (y_t - h_cab / 2)
        if alinhamento[j] == "l":
            texto(xs[j] + 0.08, yc, cabecalho[j], ha="left", multialignment="left", **cab_kw)
        else:
            texto((xs[j] + xs[j + 1]) / 2, yc, cabecalho[j], multialignment="center", **cab_kw)

    # --- corpo -----------------------------------------------------------------
    rend = fig.canvas.get_renderer()

    def largura(s):
        t = ax.text(0, 0, s, fontsize=tam_fonte)
        w = t.get_window_extent(rend).width / fig.dpi
        t.remove()
        return w

    tem_num = lambda s: any(ch.isdigit() for ch in s)
    parte = lambda s: s.split(",", 1) if "," in s else [s, ""]        # [antes, depois da vírgula]
    esq, dire = {}, {}                  # larguras antes/depois da vírgula (colunas 'v')
    for j in range(ncol):
        if alinhamento[j] == "v":
            esq[j] = max(largura(parte(l[j])[0]) for l in linhas if tem_num(l[j]))
            dire[j] = max(largura("," + parte(l[j])[1]) if "," in l[j] else 0
                          for l in linhas if tem_num(l[j]))

    for r, linha in enumerate(linhas):
        yc = y_cab - (r + 0.5) * h_lin
        for j, s in enumerate(linha):
            xc = (xs[j] + xs[j + 1]) / 2
            if alinhamento[j] == "l":
                texto(xs[j] + 0.08, yc, s, ha="left")
            elif alinhamento[j] == "v" and tem_num(s):
                x_v = xc + (esq[j] - dire[j]) / 2                   # posição da vírgula
                antes, depois = parte(s)
                texto(x_v, yc, antes, ha="right")
                if depois:
                    texto(x_v, yc, "," + depois, ha="left")
            else:
                texto(xc, yc, s)

    # --- fonte e notas ---------------------------------------------------------
    ax.text(MARGEM, y_b - 0.10, rodape, ha="left", va="top", fontsize=tam_fonte - 1,
            linespacing=1.3)

    fig.savefig(arquivo, dpi=200, facecolor="white")
    plt.close(fig)                       # não abre janela extra no plt.show()


def quantil(v, p):
    """Quantil pelo método da posição n*p: posição inteira -> média do dado e do seguinte;
    caso contrário, arredonda a posição para cima."""
    v = np.sort(np.asarray(v))
    pos = len(v) * p
    if abs(pos - round(pos)) < 1e-9:
        pos = int(round(pos))
        return (v[pos - 1] + v[pos]) / 2
    return v[math.ceil(pos) - 1]


# ===========================================================================
# QUESTÃO 2 - Medidas descritivas de X (Resistência à tração) e Y (Elasticidade)
# ===========================================================================
def classifica_assimetria(g1):
    if abs(g1) < 0.5:
        return "Aproximadamente simétrica"
    return "Assimétrica positiva (à direita)" if g1 > 0 else "Assimétrica negativa (à esquerda)"


def classifica_curtose(k):
    if abs(k - 3) < 0.1:
        return "Mesocúrtica"
    return "Leptocúrtica" if k > 3 else "Platicúrtica"


def questao2():
    banner("QUESTÃO 2 - Medidas descritivas (n = %d)" % len(X))

    def descritivas(v):
        media, desvio = v.mean(), v.std(ddof=1)        # variância/desvio amostrais (n-1)
        return [media, v.var(ddof=1), desvio, desvio / media * 100,
                stats.skew(v, bias=True),                         # m3 / m2^(3/2)
                stats.kurtosis(v, fisher=False, bias=True)]       # m4 / m2^2 (normal = 3)

    dx, dy = descritivas(X), descritivas(Y)
    medidas = ["Média aritmética", "Variância", "Desvio padrão", "Coef. de variação (%)",
               "Coef. de assimetria", "Coef. de curtose"]
    un_x = ["MPa", "MPa²", "MPa", "", "", ""]
    un_y = ["GPa", "GPa²", "GPa", "", "", ""]
    linhas = [[m, ("%s %s" % (fmt(x), ux)).strip(), ("%s %s" % (fmt(y), uy)).strip()]
              for m, x, y, ux, uy in zip(medidas, dx, dy, un_x, un_y)]

    tabela_como_imagem(
        "tabela_q2_medidas_descritivas.png",
        "Tabela 1 – Medidas descritivas da Resistência à tração (X) e da Elasticidade (Y) "
        "dos corpos de prova (n%s=%s%d)" % (NBSP, NBSP, len(X)),
        ["Medida", "Resistência à tração (X)", "Elasticidade (Y)"],
        linhas,
        larguras=[2.4, 2.6, 2.6],
        alinhamento=["l", "v", "v"],
        notas=["Nota: Variância e desvio padrão amostrais (divisor n − 1); assimetria e "
               "curtose pelos coeficientes de momentos (curtose da normal = 3)."],
    )

    print("\nClassificação quanto ao formato:")
    print("  X: %s | %s" % (classifica_assimetria(dx[4]), classifica_curtose(dx[5])))
    print("  Y: %s | %s" % (classifica_assimetria(dy[4]), classifica_curtose(dy[5])))


# ===========================================================================
# QUESTÃO 3 - Distribuição de frequências e histograma de X
# ===========================================================================
def questao3():
    n = len(X)
    k_sturges = 1 + 3.322 * np.log10(n)           # regra de Sturges
    k = int(math.ceil(k_sturges))                 # nº de classes (arredondado para cima)
    amplitude_total = X.max() - X.min()
    h_calc = amplitude_total / k
    h = 8.38                                      # amplitude de classe (arredondada para cima)
    limites = [round(X.min() + h * j, 2) for j in range(k + 1)]   # 1ª classe começa no mínimo

    banner("QUESTÃO 3 - Distribuição de frequências de X (Resistência à tração)")
    print("n = %d | mínimo = %.1f | máximo = %.1f | amplitude total = %.1f"
          % (n, X.min(), X.max(), amplitude_total))
    print("Sturges: k = 1 + 3,322*log10(n) = %.2f -> %d classes" % (k_sturges, k))
    print("Amplitude de classe: %.2f -> adotado h = %.2f" % (h_calc, h))

    # classes [inferior, superior); a última inclui o máximo
    fi, _ = np.histogram(X, bins=limites)
    assert fi.sum() == n, "Há observações fora das classes!"
    fr = fi / n
    Fi, Fr = np.cumsum(fi), np.cumsum(fr)
    pontos_medios = [(limites[j] + limites[j + 1]) / 2 for j in range(k)]

    linhas = [["%s |-- %s" % (fmt(limites[j]), fmt(limites[j + 1])), str(fi[j]), str(Fi[j]),
               fmt(fr[j], 4), fmt(Fr[j], 4), fmt(pontos_medios[j])] for j in range(k)]
    linhas.append(["Total", str(fi.sum()), "–", fmt(fr.sum(), 4), "–", "–"])

    tabela_como_imagem(
        "tabela_q3_distribuicao_frequencias.png",
        "Tabela 2 – Distribuição de frequências da Resistência à tração (MPa) "
        "dos corpos de prova (n%s=%s%d)" % (NBSP, NBSP, n),
        ["Classe\n(MPa)", "Frequência\nabsoluta\n(fi)", "Frequência\nabsoluta\nacumulada (Fi)",
         "Frequência\nrelativa\n(fr)", "Frequência\nrelativa\nacumulada (Fr)",
         "Ponto médio\n(MPa)"],
        linhas,
        larguras=[1.9, 1.2, 1.5, 1.2, 1.5, 1.2],
        alinhamento=["c", "v", "v", "v", "v", "v"],
        linha_total=True,
        notas=["Nota: Classes do tipo [inferior, superior), com amplitude h = %s MPa "
               "(regra de Sturges, k = %d). O sinal –%sindica que a medida não se aplica."
               % (fmt(h), k, NBSP)],
    )

    # --- histograma ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(X, bins=limites, edgecolor="black", color="lightblue")
    ax.set_xticks(limites)
    ax.set_xticklabels(["%.2f" % v for v in limites], fontsize=8)
    ax.set_xlabel("Resistência à tração (MPa)")
    ax.set_ylabel("Frequência absoluta")
    ax.set_title("Histograma da Resistência à tração (n = %d)" % n)
    for j in range(k):                                   # frequência sobre cada barra
        ax.text(pontos_medios[j], fi[j] + 0.2, str(fi[j]), ha="center")
    fig.tight_layout()
    fig.savefig("histograma_resistencia.png", dpi=150)


# ===========================================================================
# QUESTÃO 4 - Médias de X por Teor de fibra (A) x Temperatura de extrusão (B)
# ===========================================================================
def questao4():
    banner("QUESTÃO 4 - Médias de Resistência à tração (MPa): Teor de fibra x Temperatura")

    # margins=True acrescenta as médias marginais (linha, coluna e média geral)
    medias = df.pivot_table(values="X", index="A", columns="B", aggfunc="mean",
                            margins=True, margins_name="Média marginal")
    linhas = [[str(idx)] + [fmt(v) for v in medias.loc[idx]] for idx in medias.index]

    notas = []
    contagem = df.groupby(["A", "B"]).size().unique()    # corpos de prova por combinação
    if len(contagem) == 1:
        notas.append("Nota: Cada média refere-se a %d corpos de prova." % contagem[0])

    tabela_como_imagem(
        "tabela_q4_medias_dupla_entrada.png",
        "Tabela 3 – Resistência à tração média (MPa) dos corpos de prova, "
        "segundo o teor de fibra e a temperatura de extrusão",
        ["Teor de fibra\n(%)"] + [str(c) for c in medias.columns[:-1]] + ["Média\nmarginal"],
        linhas,
        larguras=[1.6, 1.3, 1.3, 1.3, 1.5],
        alinhamento=["l", "v", "v", "v", "v"],
        grupos=[("Temperatura de extrusão (°C)", 1, 3)],
        linha_total=True,
        notas=notas,
    )


# ===========================================================================
# QUESTÃO 5 - Elasticidade (Y) por Velocidade da rosca (C): resumo de cinco
#             números, cercas, valores atípicos e box plot
# ===========================================================================
def questao5():
    banner("QUESTÃO 5 - Elasticidade (GPa) por Velocidade da rosca")
    resumo, atipicos, caixas = {}, {}, []

    for nivel in (100, 150):
        y = df.loc[df["C"] == nivel, "Y"].to_numpy()
        q1, med, q3 = quantil(y, 0.25), quantil(y, 0.50), quantil(y, 0.75)
        aiq = q3 - q1                                   # amplitude interquartílica
        c_inf, c_sup = q1 - 1.5 * aiq, q3 + 1.5 * aiq   # cercas
        fora = (y < c_inf) | (y > c_sup)
        atipicos[nivel] = y[fora]
        resumo["%d RPM" % nivel] = {
            "Quantidade": len(y), "Mínimo": y.min(), "Q1": q1, "Mediana": med, "Q3": q3,
            "Máximo": y.max(), "AIQ": aiq, "Cerca inferior": c_inf, "Cerca superior": c_sup}
        # estatísticas do box plot calculadas com os mesmos quartis da tabela
        caixas.append({"label": "%d RPM" % nivel, "q1": q1, "med": med, "q3": q3,
                       "whislo": y[~fora].min(), "whishi": y[~fora].max(), "fliers": y[fora]})

    tabela = pd.DataFrame(resumo)
    tabela.index.name = "Medida"
    print(tabela.round(3).to_string())
    print()
    for nivel, fora in atipicos.items():
        if len(fora) == 0:
            print("%d RPM: nenhum valor atípico" % nivel)
        else:
            print("%d RPM: valores atípicos = %s" % (nivel, list(fora)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bxp(caixas, patch_artist=True, boxprops=dict(facecolor="lightblue"),
           medianprops=dict(color="red"))
    ax.set_xlabel("Velocidade da rosca")
    ax.set_ylabel("Elasticidade (GPa)")
    ax.set_title("Box plot da Elasticidade por Velocidade da rosca")
    fig.tight_layout()
    fig.savefig("boxplot_elasticidade.png", dpi=150)


if __name__ == "__main__":
    questao2()
    print()
    questao3()
    print()
    questao4()
    print()
    questao5()
    plt.show()