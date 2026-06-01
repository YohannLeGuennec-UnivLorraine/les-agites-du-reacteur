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

    mo.md("#Modélisation de la croissance bactérienne dynamique au sein d'un bioréacteur en mode discontinu")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie I - Énoncé

    Il a été vu précédemment ( i.e *Notebook : Modélisation de la croissance bactérienne dynamique au sein d'un bioréacteur* ) que la croissance bactérienne dépandait de la concentration en substrat introduit, en réalité, l'évolution de la population microbiologique prend en compte d'autre facteurs tel que les **bilans de matières**.

    En industrie, il existe 3 types de réacteurs : Batch (discontinu), Fed-batch (semi-continu), continu :

    | Type de réacteur | Entrée de matière | Sortie de matière | Volume | Description |
    |------------------|------------------|------------------|----------|-------------|
    | Batch            | ❌               | ❌               | Constant | Système fermé, pas d’échange avec l’extérieur |
    | Fed-batch        | ✅               | ❌               | Variable | Apport de substrat au cours du temps |
    | Continu          | ✅               | ✅               | Constant | Entrée et sortie continues (régime stationnaire possible) |

    Dans notre cas, nous travaillons sur un réacteur discontinu (Batch), on définit le bilan de matière de la manière suivante :

    \[
    Entrée + Réaction = Sortie + Accumulation
    \]

    En batch :
    \[
     Réaction = Accumulation
    \]


    ## Partie II - Modélisation

    ### Contexte : fermentation par *Saccharomyces cerevisiae*

    La levure *Saccharomyces cerevisiae* est un micro-organisme largement utilisé en biotechnologie, notamment dans la production de boissons fermentées et de bioéthanol.

    Dans un bioréacteur en mode batch, cette levure consomme un substrat sucré, généralement le glucose, pour assurer sa croissance. Au cours de ce processus, une partie du glucose est convertie en biomasse (croissance cellulaire), tandis qu’une autre partie est transformée en produits de fermentation, principalement l’éthanol et le dioxyde de carbone.

    Ce système constitue un exemple simple permettant d’illustrer les bilans de matière :
    - le glucose (S) est consommé,
    - la biomasse (X) augmente,
    - le produit (P), ici l’éthanol, est formé.

    L’objectif est de modéliser l’évolution de ces grandeurs au cours du temps, et de comprendre comment elles sont liées entre elles.

    ###Définition :
    On définit la vitesse volumique de réaction $r'''_A =k*[A]$.
    On obtient le système d'équation suivant :
    $$
    \begin{cases}
    \frac{dX}{dt} = r'''_X \\
    \frac{dS}{dt} = -r'''_S \\
    \frac{dP}{dt} = r'''_P
    \end{cases}
    $$

    Avec :

    - $r'''_A$ = la vitesse volumique de réaction en mol/L· s^-1^
    - $X$ = la concentration en biomasse en µmol · L^-1^
    - $S$ = la concentration en substrat (glucose) en µmol · L^-1^
    - $P$ = la concentration de produit (éthanol) en µmol · L^-1^

    On observe que la croissance de la biomasse s’accompagne :
    - d’une consommation de substrat ( signe "-" devant le $r'''$)
    - d’une formation de produit

    Ces phénomènes ne sont pas indépendants : ils sont proportionnels entre eux.

    On introduit alors des coefficients de proportionnalité appelés rendements.

    définition des rendements :

    $$
    Y_{X/S} = \frac{\text{biomasse formée}}{\text{substrat consommé}}
    \quad ; \quad
    Y_{P/X} = \frac{\text{produit formé}}{\text{biomasse formée}}
    $$


    on injecte dans les équations :
    $$
    \begin{cases}
    \frac{dX}{dt} = r'''_X \\
    \frac{dS}{dt} = -\frac{1}{Y_{X/S}} \, r'''_X \\
    \frac{dP}{dt} = Y_{P/X} \, r'''_X
    \end{cases}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En remplaçant les constantes de vitesses :
    $r'''_X =k*[X]$  , $r'''_S =k_S*[X]$, $r'''_P =k''*[X]$


    On obtient les équations modèles suivantes par intégration :

    Pour un bioréacteur en batch avec conditions initiales \(X_0\) et \(S_0\) :

    $$
    \begin{cases}
    X (t) = X_0 \, e^{k t} \\
    S(t) = S_0 - \frac{X (t) - X_0}{Y_{X/S}} \\
    P(t) = Y_{P/X} \, \big(X (t) - X_0\big)
    \end{cases}
    $$

    où :

    - \(X (t)\) : biomasse (levure) à l’instant \(t\)
    - \(S(t)\) : concentration de substrat (glucose)
    - \(P(t)\) : concentration du produit (éthanol)
    - \(k\) : taux de croissance spécifique de X
    - \(k_S\) : taux de croissance spécifique de S
    - \(Y_{X/S}\) : rendement biomasse / substrat
    - \(Y_{P/X}\) : rendement produit / biomasse
    """)
    return


@app.cell
def _(mo):
    X0 = mo.ui.slider(0.01, 5, step=0.1, value=1)
    mo.md(f"$X_0$ (Biomasse initiale) : {X0}")
    return (X0,)


@app.cell
def _(mo):
    S0 = mo.ui.slider(1.0, 25.0, step=0.1, value=20.0)
    mo.md(f"$S_0$ (Substrat initiale) : {S0}")
    return (S0,)


@app.cell
def _(mo):
    k  = mo.ui.slider(0.1, 1.0, step=0.03, value=0.3)
    mo.md(f"$k$ (Croissance) : {k}")
    return (k,)


@app.cell
def _(mo):
    kS  = mo.ui.slider(0.1, 1.0, step=0.06, value=0.3)
    mo.md(f"$k_S$ (Subtrat) : {kS}")
    return (kS,)


@app.cell
def _(mo):
    YPX = mo.ui.slider(0.0, 1.0, step=0.01, value=0.4)
    mo.md(f"$Y_P/_X$ (produit/biomasse, fixé) : {YPX}")
    return (YPX,)


@app.cell
def _(S0, X0, YPX, k, kS, mo):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    def model_batch(t, X0, S0, k, kS, YPX):
        X = X0 * np.exp(k * t)
        # Biomasse maximale autorisée par le substrat disponible
        Xmax = X0 + (S0 * k / kS)
        X = np.minimum(X, Xmax)

        # Substrat limité à S >= 0
        S = np.maximum(S0 - (kS/k)*(X - X0), 0)

        # Produit proportionnel à la biomasse formée
        P = YPX * (X - X0)
        return X, S, P

    # -----------------------------
    # 3. Temps et simulation
    # -----------------------------
    t = np.linspace(0, 20, 500)
    X, S, P = model_batch(t, X0.value, S0.value, k.value, kS.value, YPX.value)

    # -----------------------------
    # 4. Calcul des rendements finaux
    # -----------------------------
    YXS_calc = (X[-1] - X0.value) / (S0.value - S[-1])
    YPX_calc = P[-1] / (X[-1] - X0.value) if (X[-1] - X0.value) > 0 else 0

    mo.md(f"""
    **Rendements calculés à la fin du batch :**  
    - Y_X/S = {YXS_calc:.2f}  
    - Y_P/X (fixé) = {YPX.value:.2f}
    """)

    rendements = pd.DataFrame({
    "Rendement": ["Y_X/S", "Y_P/X"],
    "Valeur": [YXS_calc, YPX_calc]
    })

    print("Rendements finaux calculés selon les valeurs des sliders :")
    print(rendements)

    # -----------------------------
    # 5. Visualisation
    # -----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(t, X, label="X (biomasse)")
    plt.plot(t, S, label="S (substrat)")
    plt.plot(t, P, label="P (produit)")
    plt.xlabel("Temps")
    plt.ylabel("Concentration")
    plt.title("Batch avec substrat limité et rendements cohérents")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie III - Code Python
    Le code utilisé pour modéliser les graphes est le suivant :

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    def model_batch(t, X0, S0, k, kS, YPX):
        X = X0 * np.exp(k * t)
        # Biomasse maximale autorisée par le substrat disponible
        Xmax = X0 + (S0 * k / kS)
        X = np.minimum(X, Xmax)

        # Substrat limité à S >= 0
        S = np.maximum(S0 - (kS/k)*(X - X0), 0)

        # Produit proportionnel à la biomasse formée
        P = YPX * (X - X0)
        return X, S, P

    # -----------------------------
    # 3. Temps et simulation
    # -----------------------------
    t = np.linspace(0, 20, 500)
    X, S, P = model_batch(t, X0.value, S0.value, k.value, kS.value, YPX.value)

    # -----------------------------
    # 4. Calcul des rendements finaux
    # -----------------------------
    YXS_calc = (X[-1] - X0.value) / (S0.value - S[-1])
    YPX_calc = P[-1] / (X[-1] - X0.value) if (X[-1] - X0.value) > 0 else 0

    mo.md(f"\"\"
    **Rendements calculés à la fin du batch :**
    - Y_X/S = {YXS_calc:.2f}
    - Y_P/X (fixé) = {YPX.value:.2f}
    "\"\")

    rendements = pd.DataFrame({
    "Rendement": ["Y_X/S", "Y_P/X"],
    "Valeur": [YXS_calc, YPX_calc]
    })

    print("Rendements finaux calculés selon les valeurs des sliders :")
    print(rendements)

    # -----------------------------
    # 5. Visualisation
    # -----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(t, X, label="X (biomasse)")
    plt.plot(t, S, label="S (substrat)")
    plt.plot(t, P, label="P (produit)")
    plt.xlabel("Temps")
    plt.ylabel("Concentration")
    plt.title("Batch avec substrat limité et rendements cohérents")
    plt.legend()
    plt.gcf()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie IV - Conclusion

    La modélisation de la dynamique de croissance bactérienne au sein d'un réacteur discontinu permet d'interpréter plus facilement l'influence des paramètres :


    - La biomasse initiale ${X_0}$ : Plus elle est élevée, plus la phase de croissance est atteinte rapidement.
    - La concentration initiale en substrat ${S_0}$ : Détermine la concentration en biomasse finale. Elle est régulée par le taux de croissance $k$ et le taux de consommation $k_S$.
    - Le taux de croissance $k$ : Plus il est elevé, plus le microorganisme se reproduit de manière efficace, et inversement.
    - Le taux de consommation du substrat $k_S$ : Indique l'affinité avec le substrat, une faible valeur de $k_s$ indique même si le milieu est pauvre en nutriments, les microorganismes se développent efficacement.
    - Le rendement $Y_P/_X$ : Exprime la quantité de produit formée par unité de biomasse. Un fort rendement indique que les microorganismes convertient efficacement leur croissance en produit. La modélisation intégrant $Y_P/_X$ permet d'orienter le choix de la souche et des conditions opératoires selon l'objectif : produire des cellules ou une molécule d'intérêt.


    ## Partie V - Perspectives

    Le notebook constitue une base solide qui permet de quantifier facilement les paramètres opératoires d'une culture de microorganisme en mode discontinue mais de nombreux aspects ont été simplifier pour le modèle. Pour simuler le modèle de manière exhaustive, il est possible de prendre en compte les facteurs de mortalité des microorganismes ou bien le phénomène d'inhibition par le substrat/produit.

    Pour voir au delà des limites du mode discontinu, il serait intéressant de modéliser des réacteurs en mode semi-discontinu ou continu.

    *Voir notebook : XXX*
    """)
    return


if __name__ == "__main__":
    app.run()