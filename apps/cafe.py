import marimo

__generated_with = "unknown"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    return mo, np, plt, solve_ivp


@app.cell
def _(mo):
    mo.md(r"""
    # Introduction aux transferts thermiques : refroidissement d’une tasse de café

    ## Contexte pédagogique

    Avant d’étudier des procédés industriels complexes comme la distillation,
    il est utile de commencer par un phénomène thermique connu de tous :
    le refroidissement d’une tasse de café.

    Cette approche permet d’introduire progressivement plusieurs notions fondamentales du génie des procédés :

    - bilans d’énergie,
    - convection thermique,
    - conduction thermique,
    - résistances thermiques,
    - modélisation dynamique,
    - résolution numérique d’équations différentielles.

    Ces concepts sont ensuite directement réutilisés dans les opérations industrielles étudiées à l’ENSIC, notamment :

    - échangeurs thermiques,
    - réacteurs,
    - colonnes de distillation,
    - procédés de séparation.

    ---
    #  Description physique du problème

    On considère une tasse contenant du café chaud initialement à une température :

    $$
    T_{café}(0)=T_0
    $$

    Le café échange de la chaleur avec :

    - la paroi du récipient ;
    - l’air ambiant ;
    - la surface libre du liquide.

    Le récipient peut être :

    - un mug en céramique ;
    - un gobelet en carton.

    L’utilisateur peut également simuler l’action de souffler sur le café,
    ce qui augmente les coefficients de convection.

    # Hypothèses du modèle

    Afin d’obtenir un modèle simple mais physiquement cohérent,
    les hypothèses suivantes sont retenues :

    - température homogène dans le café ;
    - température homogène dans la paroi ;
    - propriétés physiques constantes ;
    - pas d’évaporation ;
    - géométrie cylindrique simplifiée.

    Le modèle comporte alors deux températures variables :

    $$
    T_c(t) : \text{température du café}
    $$

    $$
    T_p(t) : \text{température de la paroi}
    $$
    ---
    #  Flux thermiques pris en compte

    ## 1. Convection entre le café et la paroi

    Le flux thermique entre le café et la paroi vaut :

    $$
    Q_{int}
    =
    h_{café} A_{paroi}(T_c-T_p)
    $$

    où :

    - $h_{café}$ est le coefficient de convection interne ;
    - $A_{paroi}$ est la surface latérale.

    ## 2. Échange entre la paroi et l’air extérieur

    La conduction dans la paroi et la convection avec l’air
    sont regroupées dans un coefficient global :

    $$
    h_{paroi}
    =
    \frac{1}{
    \frac{e}{k}
    +
    \frac{1}{h_{air}}
    }
    $$

    Le flux thermique externe vaut alors :

    $$
    Q_{ext}
    =
    h_{paroi}A_{paroi}(T_p-T_{\infty})
    $$

    ## 3. Convection à la surface libre

    La surface libre du café échange également avec l’air :

    $$
    Q_{top}
    =
    h_{top}A_{top}(T_c-T_{\infty})
    $$
    ---
    # Schèma

    ![alt](public/image.png)
    ---
    #  Bilans d’énergie

    ##  Bilan sur le café

    Le bilan énergétique sur le café conduit à :

    $$
    m_c c_{p,c}
    \frac{dT_c}{dt}
    =
    -
    Q_{int}
    -
    Q_{top}
    $$

    soit :

    $$
    m_c c_{p,c}
    \frac{dT_c}{dt}
    =
    -
    h_{café}A_{paroi}(T_c-T_p)
    -
    h_{top}A_{top}(T_c-T_{\infty})
    $$

    ##  Bilan sur la paroi

    Le bilan énergétique sur la paroi donne :

    $$
    m_p c_{p,p}
    \frac{dT_p}{dt}
    =
    Q_{int}
    -
    Q_{ext}
    $$

    soit :

    $$
    m_p c_{p,p}
    \frac{dT_p}{dt}
    =
    h_{café}A_{paroi}(T_c-T_p)
    -
    h_{paroi}A_{paroi}(T_p-T_{\infty})
    $$
    ---
    #  Résolution numérique : méthode d’Euler

    Le système obtenu est un système de deux équations différentielles ordinaires couplées.

    Il peut être résolu numériquement par la méthode d’Euler explicite.

    ## Principe de la méthode

    Pour une équation :

    $$
    \frac{dT}{dt}=f(T,t)
    $$

    la méthode d’Euler consiste à approximer :

    $$
    T_{n+1}
    =
    T_n
    +
    \Delta t \, f(T_n,t_n)
    $$

    où :

    - $\Delta t$ est le pas de temps ;
    - $T_n$ est la température au temps $t_n$.

    #  Application au système thermique

    À chaque pas de temps :

    1. on calcule les flux thermiques ;
    2. on calcule les dérivées :
    $$
    \frac{dT_c}{dt}
    \quad
    \text{et}
    \quad
    \frac{dT_p}{dt}
    $$
    3. on réitère pour le pas de temps suivant.

    Cependant, on résoudra les équations directement avec la fonction solve de marimo
    ---
    #  Données numériques utilisées dans la simulation

    Le système étudié correspond à une tasse cylindrique contenant du café chaud.

    Les dimensions géométriques retenues sont :

    - rayon de la tasse :

    $$
    R = 0.03 \ \text{{m}}
    $$

    - hauteur du café :

    $$
    z = 0.08 \ \text{{m}}
    $$

    Les surfaces d’échange thermique sont alors :

    - surface latérale :

    $$
    A_{{paroi}}
    =
    2\pi R z
    $$

    - surface libre :

    $$
    A_{{top}}
    =
    \pi R^2
    $$
    ---
    #  Propriétés thermiques du café

    Le café est assimilé à de l’eau liquide.

    Les propriétés retenues sont :

    - masse de café :

    $$
    m_c = 0.25 \ \text{{kg}}
    $$

    - capacité calorifique :

    $$
    c_{{p,c}}
    =
    4180
    \ \text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}
    $$
    ---
    #  Paramètres du mug en céramique

    Pour le mug en céramique :

    - masse de la paroi :

    $$
    m_p = 0.40 \ \text{{kg}}
    $$

    - capacité calorifique :

    $$
    c_{{p,p}}
    =
    900
    \ \text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}
    $$

    - conductivité thermique :

    $$
    k = 1.5
    \ \text{W} \cdot \text{m}^{-1} \cdot \text{K}^{-1}
    $$

    - épaisseur de paroi :

    $$
    e = 5 \times 10^{{-3}} \ \text{{m}}
    $$
    ---
    #  Paramètres du gobelet en carton

    Pour le gobelet en carton :

    - masse de la paroi :

    $$
    m_p = 0.05 \ \text{{kg}}
    $$

    - capacité calorifique :

    $$
    c_{{p,p}}
    =
    1400
    \ \text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}
    $$

    - conductivité thermique :

    $$
    k = 0.08
    \ \text{W} \cdot \text{m}^{-1} \cdot \text{K}^{-1}
    $$

    - épaisseur de paroi :

    $$
    e = 1.5 \times 10^{{-3}} \ \text{{m}}
    $$
    ---
    #  Coefficients de transfert thermique

    Les coefficients utilisés dans les échanges thermiques sont :

    - convection café → paroi :

    $$
    h_{{café}}
    =
    500
    \ \text{W} \cdot \text{m}^{-2} \cdot \text{K}^{-1}
    $$

    - convection air ambiant :

    $$
    h_{{air}}
    =
    15
    \ \text{W} \cdot \text{m}^{-2} \cdot \text{K}^{-1}
    $$

    - convection surface libre sans souffler :

    $$
    h_{{top}}
    =
    15
    \ \text{W} \cdot \text{m}^{-2} \cdot \text{K}^{-1}
    $$

    - convection surface libre avec soufflage :

    $$
    h_{{top}}
    =
    80
    \ \text{W} \cdot \text{m}^{-2} \cdot \text{K}^{-1}
    $$
    ---
    #  Conditions initiales

    La température initiale du café est réglable entre :

    $$
    60^\circ C
    \leq
    T_0
    \leq
    95^\circ C
    $$

    La température ambiante est réglable entre :

    $$
    10^\circ C
    \leq
    T_\infty
    \leq
    30^\circ C
    $$
    """)
    return


@app.cell
def _(mo):
    type_recipient = mo.ui.radio(
        options=["Mug céramique", "Gobelet carton"],
        value="Mug céramique",
        label="Type de récipient"
    )

    souffler = mo.ui.switch(
        label="Souffler sur le café ?",
        value=False
    )

    T0_c = mo.ui.slider(60, 95, value=85,
                        label="Température initiale café (°C)")

    T_inf = mo.ui.slider(10, 30, value=20,
                         label="Température ambiante (°C)")

    mo.vstack([type_recipient, souffler, T0_c, T_inf])
    return T0_c, T_inf, souffler, type_recipient


@app.cell
def _(np):
    R = 0.03
    z = 0.08

    A_paroi = 2 * np.pi * R * z
    A_top = np.pi * R**2

    m_c = 0.25
    cp_c = 4180

    def get_parameters(type_recipient, souffler):

        h_cafe = 500
        h_air = 15

        if type_recipient == "Mug céramique":
            m_paroi = 0.40
            cp_paroi = 900
            k = 1.5
            e_paroi = 0.005

        elif type_recipient == "Gobelet carton":
            m_paroi = 0.05
            cp_paroi = 1400
            k = 0.08
            e_paroi = 0.0015

        else:
            raise ValueError("Type inconnu")

        # Coefficient global extérieur (conduction + convection air)
        h_paroi = 1 / (e_paroi/k + 1/h_air)

        # Effet souffler
        if souffler:
            h_top = 80
            h_paroi *= 1.5
        else:
            h_top = 15

        return m_paroi, cp_paroi, k, e_paroi, h_paroi, h_top, h_cafe

    return A_paroi, A_top, cp_c, get_parameters, m_c


@app.cell
def _(
    A_paroi,
    A_top,
    T_inf,
    cp_c,
    get_parameters,
    m_c,
    souffler,
    type_recipient,
):
    params = get_parameters(type_recipient.value, souffler.value)
    m_paroi, cp_paroi, k, e_paroi, h_paroi, h_top, h_cafe = params

    def dTdt(t, T):

        Tc, Tparoi= T

        Q_int = h_cafe * A_paroi * (Tc - Tparoi)
        Q_ext = h_paroi * A_paroi * (Tparoi - T_inf.value)
        Q_top = h_top * A_top * (Tc - T_inf.value)

        dTc = (-Q_int - Q_top) / (m_c * cp_c)
        dTparoi = ( Q_int - Q_ext) / (m_paroi * cp_paroi)

        return [dTc, dTparoi]

    return (dTdt,)


@app.cell
def _(T0_c, dTdt, np, plt, solve_ivp, souffler, type_recipient):
    t_eval = np.linspace(0, 7200, 800)

    sol = solve_ivp(
        dTdt,
        (0, 7200),
        [T0_c.value, T0_c.value], 
        t_eval=t_eval
    )

    Tc = sol.y[0]
    Tparoi = sol.y[1]

    plt.figure(figsize=(8,5))
    plt.plot(t_eval/60, Tc, label="Café", lw=2)
    plt.plot(t_eval/60, Tparoi, label="Paroi", lw=2)

    plt.xlabel("Temps (min)")
    plt.ylabel("Température (°C)")
    plt.title(
        f"Refroidissement : {type_recipient.value} | Souffler = {souffler.value}"
    )
    plt.grid(True)
    plt.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
