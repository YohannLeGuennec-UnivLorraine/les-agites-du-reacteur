# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.11",
# ]
# ///
import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Distillation binaire éthanol / eau
    ## Étude par la méthode de McCabe–Thiele

    ---

    ## 1. Contexte industriel

    La distillation est l’un des procédés de séparation les plus utilisés en industrie chimique, pharmaceutique et agroalimentaire.
    Elle permet de séparer les constituants d’un mélange liquide en exploitant leur différence de volatilité.

    On s’intéresse ici à la séparation d’un mélange binaire **éthanol / eau**, système représentatif :

    - de la production de bioéthanol,
    - de la purification de solvants pharmaceutiques,
    - de procédés de chimie fine.

    La séparation est réalisée dans une **colonne de distillation continue fonctionnant à pression atmosphérique**.

    ---

    ## 2. Description du procédé

    Une alimentation liquide de composition molaire en éthanol $x_F$ est introduite dans la colonne.

    On souhaite obtenir :

    - un distillat enrichi en éthanol, de composition $x_D$,
    - un résidu appauvri, de composition $x_B$.

    La colonne fonctionne avec un **rapport de reflux réglable $R$**, défini par :

    $$
    R = \frac{L}{D}
    $$

    où :

    - $L$ est le débit molaire de liquide recyclé en tête,
    - $D$ est le débit molaire de distillat extrait.

    ---

    ## 3. Hypothèses retenues

    Afin de simplifier l’analyse, on adopte les hypothèses suivantes :

    - mélange binaire idéal,
    - pression constante,
    - équilibre liquide-vapeur atteint sur chaque plateau,
    - débits molaires constants dans chaque section,
    - volatilité relative constante $\alpha$.

    ---

    ## 4. Modélisation de l’équilibre liquide–vapeur

    L’équilibre est décrit par la relation :

    $$
    y = \frac{\alpha x}{1 + (\alpha - 1)x}
    $$

    où :

    - $x$ : fraction molaire d’éthanol dans la phase liquide,
    - $y$ : fraction molaire d’éthanol dans la phase vapeur,
    - $\alpha$ : volatilité relative.

    ---

    ## 5. Objectifs de l’étude

    On se propose de :

    1. Tracer le diagramme d’équilibre $y = f(x)$.
    2. Construire graphiquement les droites d’exploitation.
    3. Appliquer la méthode de McCabe–Thiele.
    4. Déterminer le nombre d’étages théoriques nécessaires.
    5. Étudier l’influence du reflux $R$ sur la séparation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Théorie de la distillation binaire

    ---

    ## 1. Principe général

    La distillation repose sur la différence de volatilité des constituants d’un mélange.

    Si le composé A est plus volatil que B, alors :

    $$
    y_A > x_A
    $$

    Autrement dit, la phase vapeur est enrichie en composé le plus volatil.

    ---

    ## 2. Volatilité relative

    On définit la volatilité relative :

    $$
    \alpha = \frac{(y_A/x_A)}{(y_B/x_B)}
    $$

    Pour un mélange binaire idéal :

    - si $\alpha > 1$ : séparation possible,
    - si $\alpha \rightarrow 1$ : séparation difficile,
    - plus $\alpha$ est grand, plus la séparation est aisée.

    ---

    ## 3. Bilan matière global

    Autour de la colonne, le bilan molaire global donne :

    $$
    F = D + B
    $$

    Bilan en éthanol :

    $$
    F x_F = D x_D + B x_B
    $$

    Ces relations relient les compositions et les débits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Méthode graphique de McCabe–Thiele

    ---

    ## 1. Principe

    La méthode de McCabe–Thiele est une méthode graphique permettant de déterminer :

    - le nombre d’étages théoriques,
    - la position du plateau d’alimentation,
    - l’influence du reflux.

    Elle repose sur une alternance :

    - étape horizontale → équilibre liquide–vapeur,
    - étape verticale → droite d’exploitation.

    ---

    ## 2. Construction graphique

    1. Tracer la courbe d’équilibre $y = f(x)$.
    2. Tracer la droite $y = x$.
    3. Tracer les droites d’exploitation.
    4. Construire les "escaliers" à partir de $x_D$.
    5. Compter le nombre d’étages jusqu’à $x_B$.

    Chaque marche correspond à un plateau théorique.

    ---

    ## 3. Interprétation physique

    - Plus les marches sont nombreuses → plus la séparation est difficile.
    - Augmenter le reflux rapproche la droite d’exploitation de la diagonale.
    - Lorsque le reflux tend vers l’infini, le nombre d’étages diminue.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Droite de rectification (section d’enrichissement)

    ---

    ## 1. Bilan matière sur la section supérieure

    On considère un plateau situé au-dessus de l’alimentation.

    Sous l’hypothèse de débits molaires constants :

    - $L$ : débit liquide descendant
    - $V$ : débit vapeur montant

    Le bilan molaire en éthanol conduit à :

    $$
    V y = L x + D x_D
    $$

    En divisant par $V$ :

    $$
    y = \frac{L}{V} x + \frac{D}{V} x_D
    $$

    ---

    ## 2. Expression en fonction du reflux

    On rappelle :

    $$
    R = \frac{L}{D}
    $$

    et :

    $$
    V = L + D
    $$

    On obtient finalement :

    $$
    y = \frac{R}{R+1} x + \frac{x_D}{R+1}
    $$

    Cette droite est appelée **droite de rectification**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Droite de stripping (section d’appauvrissement)

    ---

    ## 1. Bilan matière sous l’alimentation

    Dans la partie inférieure :

    - $L'$ : débit liquide
    - $V'$ : débit vapeur

    Bilan molaire en éthanol :

    $$
    V' y = L' x - B x_B
    $$

    Sous l’hypothèse de débits constants dans la section :

    la droite prend la forme :

    $$
    y = m x + c
    $$

    ---

    ## 2. Détermination pratique

    En pratique, la droite de stripping :

    - passe par le point $(x_B, x_B)$,
    - intersecte la droite de rectification au niveau de l’alimentation.

    Sa pente dépend des bilans locaux autour du plateau d’alimentation.

    ---

    ## 3. Interprétation physique

    Cette droite traduit l’enrichissement progressif du liquide descendant en composé le moins volatil.

    Plus la pente est forte, plus la séparation est efficace dans la zone basse.
    """)
    return


@app.cell
def _(mo):
    xF = mo.ui.slider(0.05, 0.6, 0.01, value=0.30, label="x_F (alimentation)")
    xD = mo.ui.slider(0.6, 0.95, 0.01, value=0.85, label="x_D (distillat)")
    xB = mo.ui.slider(0.01, 0.2, 0.01, value=0.05, label="x_B (résidu)")
    R = mo.ui.slider(0.5, 5.0, 0.1, value=2.0, label="Reflux R")
    alpha = mo.ui.slider(1.5, 4.0, 0.1, value=2.2, label="Volatilité relative α")

    mo.vstack([xF, xD, xB, R, alpha])
    return R, alpha, xB, xD, xF


@app.function
def equilibrium(x, alpha):
    return (alpha * x) / (1 + (alpha - 1) * x)


@app.cell
def _(np):
    def inverse_equilibrium(y, alpha):
        x_vals = np.linspace(0, 1, 2000)
        y_vals = equilibrium(x_vals, alpha)
        return np.interp(y, y_vals, x_vals)

    return (inverse_equilibrium,)


@app.function
def rectifying_line(x, R, xD):
    return (R / (R + 1)) * x + xD / (R + 1)


@app.function
def stripping_line(x, xB, xF, R, xD):
    yF = rectifying_line(xF, R, xD)
    slope = (yF - xB) / (xF - xB)
    return slope * (x - xB) + xB


@app.cell
def _(inverse_equilibrium):
    def mccabe_thiele(xD, xB, xF, R, alpha):
        x = xD
        y = xD

        x_points = [x]
        y_points = [y]

        stages = 0
        max_stages = 50

        while x > xB and stages < max_stages:

            # 1️⃣ Étape horizontale → équilibre
            x_eq = inverse_equilibrium(y, alpha)

            x_points.append(x_eq)
            y_points.append(y)

            # 2️⃣ Étape verticale → droite d’exploitation
            if x_eq >= xF:
                y_new = rectifying_line(x_eq, R, xD)
            else:
                y_new = stripping_line(x_eq, xB, xF, R, xD)

            x_points.append(x_eq)
            y_points.append(y_new)

            # Mise à jour
            x = x_eq
            y = y_new
            stages += 1

        return x_points, y_points, stages

    return (mccabe_thiele,)


@app.cell
def _(R, alpha, mccabe_thiele, np, plt, xB, xD, xF):
    x = np.linspace(0, 1, 400)
    y_eq = equilibrium(x, alpha.value)
    y_rect = rectifying_line(x, R.value, xD.value)
    y_strip = stripping_line(x, xB.value, xF.value, R.value, xD.value)

    xp, yp, N = mccabe_thiele(
        xD.value, xB.value, xF.value, R.value, alpha.value
    )

    plt.figure(figsize=(8,8))
    plt.plot(x, y_eq, label="Équilibre liquide-vapeur", lw=2)
    plt.plot(x, x, "--", label="y = x")
    plt.plot(x, y_rect, label="Droite d'enrichissement")
    plt.plot(x, y_strip, label="Droite d'appauvrissement")

    plt.plot(xp, yp, "k-", lw=1.5)
    plt.scatter(xp, yp, color="red", zorder=5)

    plt.scatter(xp, yp, color="red", zorder=5)

    plt.xlabel("x (liquide)")
    plt.ylabel("y (vapeur)")
    plt.title(f"Méthode de McCabe–Thiele\nNombre d'étages théoriques ≈ {N}")
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
