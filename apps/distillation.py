# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///
import marimo

__generated_with = "unknown"
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

            x_eq = inverse_equilibrium(y, alpha)

            x_points.append(x_eq)
            y_points.append(y)

            if x_eq >= xF:
                y_new = rectifying_line(x_eq, R, xD)
            else:
                y_new = stripping_line(x_eq, xB, xF, R, xD)

            x_points.append(x_eq)
            y_points.append(y_new)

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
    return (N,)


@app.cell
def _(N, R):
    D = 1.0              # kmol/h
    Hvap = 35000         # kJ/kmol
    t_an = 8000          # h/an
    cout_kWh = 0.05      # €/kWh

    C0 = 50000           # € coût référence
    N0 = 10              # nombre étages référence
    amortissement = 10   # années

    # Débit vapeur interne
    V = (R.value + 1) * D

    # Puissance thermique
    Q_kJh = V * Hvap

    # Conversion kW
    Q_kW = Q_kJh / 3600

    # Énergie annuelle
    E_an = Q_kW * t_an

    # Coût énergétique annuel
    C_energy = E_an * cout_kWh

    C_column = C0 * (N / N0)**0.6

    # Annualisation
    C_column_annuel = C_column / amortissement

    C_total = C_energy + C_column_annuel
    return (
        C0,
        C_column,
        C_column_annuel,
        C_energy,
        C_total,
        D,
        E_an,
        Hvap,
        N0,
        Q_kW,
        V,
        amortissement,
        cout_kWh,
        t_an,
    )


@app.cell
def _(
    C0,
    C_column,
    C_column_annuel,
    C_energy,
    C_total,
    D,
    E_an,
    Hvap,
    N,
    N0,
    Q_kW,
    R,
    V,
    amortissement,
    cout_kWh,
    mo,
    t_an,
):
    mo.md(rf"""
    # Analyse énergétique et économique de la distillation

    ---

    ## 1. Nombre d’étages théoriques

    La méthode de McCabe–Thiele permet de déterminer graphiquement le nombre d’étages nécessaires à la séparation.

    Dans notre cas :

    $$
    N \approx {N:.1f}
    $$

    Plus le nombre d’étages est élevé :

    - plus la séparation est efficace,
    - mais plus la colonne est haute et coûteuse.

    ---

    ## 2. Rapport de reflux

    Le reflux correspond à la fraction du distillat renvoyée dans la colonne.

    $$
    R = \frac{{L}}{{D}}
    $$

    Dans cette simulation :

    $$
    R = {R.value:.2f}
    $$

    Un reflux élevé améliore la séparation mais augmente fortement la consommation énergétique.

    ---

    ## 3. Débit de vapeur interne

    Sous l’hypothèse des débits molaires constants :

    $$
    V = (R+1)D
    $$

    Avec :

    - $D = {D:.1f} \ \mathrm{{kmol/h}}$
    - $R = {R.value:.2f}$

    on obtient :

    $$
    V = {V:.2f} \ \mathrm{{kmol/h}}
    $$

    ---

    ## 4. Puissance thermique du rebouilleur

    La puissance thermique nécessaire à la vaporisation vaut :

    $$
    Q_R = V \Delta H_{{vap}}
    $$

    où :

    - $\Delta H_{{vap}} = {Hvap:.0f} \ \mathrm{{kJ/kmol}}$

    Ainsi :

    $$
    Q_R = {Q_kW:.1f} \ \mathrm{{kW}}
    $$

    Cette puissance représente l’énergie à fournir au rebouilleur de la colonne.

    ---

    ## 5. Consommation énergétique annuelle

    Pour un fonctionnement annuel de :

    $$
    t_{{an}} = {t_an:.0f} \ \mathrm{{h/an}}
    $$

    l’énergie consommée vaut :

    $$
    E_{{annuel}} = Q_R \times t_{{an}}
    $$

    d’où :

    $$
    E_{{annuel}} = {E_an:.0f} \ \mathrm{{kWh/an}}
    $$

    ---

    ## 6. Coût énergétique annuel

    En supposant un coût énergétique de :

    $$
    C_{{energie}} = {cout_kWh:.2f} \ \mathrm{{€/kWh}}
    $$

    le coût annuel de fonctionnement devient :

    $$
    C_{{op}} = {C_energy:.0f} \ \mathrm{{€/an}}
    $$

    ---

    ## 7. Coût d’investissement de la colonne

    Le coût d’investissement est estimé par une loi empirique :

    $$
    C_{{colonne}} =
    C_0
    \left(
    \frac{{N}}{{N_0}}
    \right)^{{0.6}}
    $$

    avec :

    - $C_0 = {C0:.0f} \ \mathrm{{€}}$
    - $N_0 = {N0:.0f}$

    On obtient :

    $$
    C_{{colonne}} = {C_column:.0f} \ \mathrm{{€}}
    $$

    ---

    ## 8. Annualisation du coût d’investissement

    En supposant une durée d’amortissement de :

    $$
    {amortissement} \ \mathrm{{ans}}
    $$

    le coût annuel équivalent devient :

    $$
    C_{{CAPEX}} =
    {C_column_annuel:.0f}
    \ \mathrm{{€/an}}
    $$

    ---

    ## 9. Coût total annuel

    Le coût global du procédé est estimé par :

    $$
    C_{{total}} =
    C_{{op}} + C_{{CAPEX}}
    $$

    soit :

    $$
    C_{{total}}
    =
    {C_total:.0f}
    \ \mathrm{{€/an}}
    $$

    ---

    # Conclusion

    Cette étude montre le compromis classique du génie des procédés :

    - augmenter le reflux diminue le nombre d’étages,
    - mais augmente la consommation énergétique.

    Le dimensionnement optimal d’une colonne résulte donc d’un compromis entre :

    - coût énergétique,
    - coût d’investissement,
    - efficacité de séparation.
    """)
    return


if __name__ == "__main__":
    app.run()
