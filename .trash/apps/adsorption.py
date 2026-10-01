# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///
import marimo

__generated_with = "0.19.11"
app = marimo.App(app_title="Adsorption")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import marimo as mo

    return mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chromatographie d’adsorption
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Objectif du modèle

    Ce notebook simule la séparation de deux espèces chimiques dans une colonne
    chromatographique. Le modèle décrit l’évolution des concentrations dans la phase
    liquide et solide en tenant compte : de la convection dans la colonne, du transfert de masse externe (film liquide), du transfert de masse interne (diffusion dans les particules) et de l’équilibre d’adsorption compétitif

    L’objectif est de prédire les **courbes d’élution** en sortie de colonne.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Lois phénoménologiques

    Cette section regroupe les lois physiques et chimiques utilisées pour repéresenter les phénomènes dans le système de chromatographie. On distingue les lois basées sur la thermodynamique et les lois basées sur les transferts de matière.
    """)
    return


@app.cell
def _(mo):
    tab_thermo = mo.md(r"""
    Les espèces se disputent un nombre limité de sites d’adsorption selon une isotherme de Langmuir de la forme :

    \[
    q_i =
    \frac{q_{max,i} K_i C_i}{1 + \sum_j K_j C_j}
    \]

    Conséquences physiques :

    - saturation des sites (lorsque $C_i \rightarrow +\infty$ alors $q_i \rightarrow q_{max,i}$)
    - compétition entre espèces (toutes les espèces interviennent dans le terme $\sum_j K_j C_j$)
    - comportement non linéaire (sera observé dès lors que $\sum_j K_j C_j > 1$) qui impactera la forme des pics chromatographiques (pics non symétriques)
    """)

    tab_transfert_ext = mo.md(r"""
    La résistance entre phase liquide et surface du solide est modélisée par :

    \[
    \varphi_{liq \rightarrow surf.} = \frac{C_s - C}{t_{ext}}
    \]

    où :

    - \(C\) : concentration bulk
    - \(C_s\) : concentration à la surface
    - \(t_{ext}\) : temps caractéristique du film
    """)

    tab_transfert_int = mo.md(r"""
    La diffusion interne est modélisée par le modèle LDF (Linear Driving Force) :

    \[
    \varphi_{surf. \rightarrow sol.} = \frac{q - q_{eq}(C_s)}{t_{int}}
    \]

    Ce modèle approxime la diffusion dans les pores par une loi linéaire plus simple.
    """)

    tabs1 = mo.ui.tabs({
        "Thermodynamique": tab_thermo,
        "Transfert externe": tab_transfert_ext,
        "Transfert interne": tab_transfert_int
    })

    tabs1

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bilans de matière

    La colonne est représentée par N cellules parfaitement mélangées en série. Cela approxime un réacteur piston avec dispersion.

    Chaque cellule contient :

    - une phase liquide externe
    - une phase solide poreuse supposée homogène
    - une surface d’interface liquide–solide

    La porosité externe $\epsilon_e$ de la colonne (ration entre le volume de liquide externe et le volume de colonne) est pris égal à 40%.

    Le modèle repose sur trois bilans couplés. Chaque bilan doit être écrit pour chaque espèce $i$ dans chaque cellule de mélange $j$ (ainsi pour un système avec 30 cellules de mélange et 2 espèces, le système différentiel sera constitué de 180 équations différentielles à résoudre simultanément) :
    """)
    return


@app.cell
def _(mo):
    tab1 = mo.md(r"""
    Le bilan sur la phase liquide s'écrit : 

    \[
    QC_{i}^{(j-1)} = QC_{i}^{(j)} + \varphi_{i, liq \rightarrow surf.}^{(j)} \epsilon_e V + \frac{dn_i^{(j)}}{dt}
    \]

    Soit après réarrangement :

    \[
    \frac{dC_i^{(j)}}{dt}
    =
    \frac{Q}{\epsilon_e V^{(j)}}(C_{i}^{(j-1)}-C_{i}^{(j)})
    +
    \frac{C_i^{(j)}-C_{s,i}^{(j)}}{t_{i, ext}}
    \]

    Par définition, le temps de séjour d'une espèce ayant accès uniquement au volume de liquide externe dans la cellule de mélange $j$ s'écrit : 

    $$t_0^{(j)} = \frac{\epsilon_e V^{(j)}}{Q}$$

    L'équation différentielle devient alors : 

    \[
    \frac{dC_i^{(j)}}{dt}
    =
    \frac{C_{i}^{(j-1)}-C_{i}^{(j)}}{t_0^{(j)}}
    +
    \frac{C_i^{(j)}-C_{s,i}^{(j)}}{t_{i, ext}}
    \]
    """)

    tab2 = mo.md(r"""
    Le bilan à l'interface s'écrit

    \[
    \varphi_{i, liq. \rightarrow surf.}^{(j)}
    = \varphi_{i, surf. \rightarrow sol.}^{(j)} + \frac{dC_{s,i}^{(j)}}{dt}
    \]

    Soit après réarrangement :

    \[
    \frac{dC_{s,i}^{(j)}}{dt} = \frac{C_{s,i}^{(j)} - C_i^{(j)}}{t_{i, ext}} - \frac{q_i^{(j)} - q_{eq,i}^{(j)}(C_s)}{t_{i, int}}
    \]
    """)

    tab3 = mo.md(r"""
    Le bilan sur le solide s'écrit : 

    \[
    \frac{dn_{i,sol.}^{(j)}}{dt} = \varphi_{i, surf. \rightarrow sol.}^{(j)} (1-\epsilon_e) V^{(j)}
    \]

    Soit après réarrangement :

    \[
    \frac{dq_{i}^{(j)}}{dt} = \frac{q_i^{(j)} - q_{eq,i}^{(j)}(C_s)}{t_{i, int}}
    \]
    """)

    tabs = mo.ui.tabs({
        "Phase liquide": tab1,
        "Interface liquide-solide": tab2,
        "Phase solide": tab3
    })

    tabs

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Simulations

    Le système obtenu est fortement couplé, contient plusieurs échelles de temps et est raide (stiff). La résolution utilise donc la méthode implicite BDF adaptée à ce type de problème.

    La simulation produit les concentrations en sortie de colonne, les pics d’élution et l’influence des paramètres physico-chimiques

    La largeur et la séparation des pics dépendent notamment de la compétition d’adsorption, les temps de transfert, la porosité, le débit.
    """)
    return


@app.cell
def _(mo):
    mo.md("## ⚙️ Paramètres de simulation")

    # ==================================================
    # 1️⃣ Nombre d'espèces
    # ==================================================
    mo.md("### 🔬 Nombre d'espèces")

    n_species = mo.ui.number(
        value=2,
        start=1,
        step=1,
        label="Nombre d'espèces"
    )

    n_species
    return (n_species,)


@app.cell
def _(mo, n_species):
    # --------------------------------------------------
    # Génération dynamique du tableau espèces
    # --------------------------------------------------
    M = int(n_species.value)

    mo.md("### 🧪 Paramètres par espèce")

    espece = []
    C_feed = []
    K = []
    q_max = []
    t_i = []
    t_ext = []

    for i in range(M):
        espece.append(
            mo.ui.text(f"Espèce {i+1}")
        )
        C_feed.append(
            mo.ui.number(i+1)
        )
        K.append(
            mo.ui.number(start=1e-3, value=10**(i+2))
        )
        q_max.append(
            mo.ui.number(start=1e-3, value=1e-2)
        )
        t_i.append(
            mo.ui.number(start=1e-3, value=1e-3)
        )
        t_ext.append(
            mo.ui.number(start=1e-3, value=1e-3)
        )

    species_table = mo.hstack([
        mo.vstack([mo.md("**Espèce**"), *espece]),
        mo.vstack([mo.md("$C_{feed}(mol.L^{-1})$"), *C_feed]),
        mo.vstack([mo.md("$K(L.mol^{-1})$"), *K]),
        mo.vstack([mo.md("$q_{max}(mol.L^{-1})$"), *q_max]),
        mo.vstack([mo.md("$t_{int}(min)$"), *t_i]),
        mo.vstack([mo.md("$t_{ext}(min)$"), *t_ext]),
    ])

    species_table
    return C_feed, K, M, q_max, t_ext, t_i


@app.cell
def _(mo):
    # ==================================================
    # 2️⃣ Paramètres procédé
    # ==================================================
    mo.md("### 🏭 Paramètres procédé")

    Q = mo.ui.number(start=1e-3, value=1.0, label="Débit Q (mL/min)")
    V_col = mo.ui.number(start=1e-3, value=50.0, label="Volume colonne (mL)")
    N = mo.ui.number(start=1, value=30, label="Nombre cellules")
    eps = mo.ui.number(start=0.36, value=0.4, label="Porosité externe")
    t_inj = mo.ui.number(start=0.0, value=0.1, label="Temps injection")
    t_final = mo.ui.number(start=0.0, value=300.0, label="Temps final")

    process_table = mo.vstack([Q, V_col, N, eps, t_inj, t_final])
    process_table
    return N, Q, V_col, eps, t_final, t_inj


@app.cell
def _(np):
    def inlet_concentration(t, t_inj, C_feed):
        return C_feed if t <= t_inj else np.zeros_like(C_feed)

    def chromatograph_model(
        t, C_flat, N, Q, V_cell, eps, q_max, K, C_feed, t_inj, t_i, t_ext
    ):
        M = len(C_feed)

        C = C_flat[: N * M].reshape((N, M))
        q = C_flat[N * M : 2 * N * M].reshape((N, M))
        C_s = C_flat[2 * N * M :].reshape((N, M))

        dCdt = np.zeros_like(C)
        dqdt = np.zeros_like(q)
        dC_s_dt = np.zeros_like(C_s)

        for i in range(N):
            C_up = inlet_concentration(t, t_inj, C_feed) if i == 0 else C[i - 1]

            for j in range(M):
                D = 1.0 + np.sum(K * C_s[i])
                q_eq = q_max[j] * K[j] * C_s[i, j] / D

                dqdt[i, j] = (q_eq - q[i, j]) / t_i[j]

                dC_s_dt[i, j] = (
                    (C[i, j] - C_s[i, j]) / t_ext[j]
                    - ((1.0 - eps) / eps) * dqdt[i, j]
                )

                dCdt[i, j] = (
                    (Q / (eps * V_cell)) * (C_up[j] - C[i, j])
                    - (C[i, j] - C_s[i, j]) / t_ext[j]
                )

        return np.concatenate([dCdt.ravel(), dqdt.ravel(), dC_s_dt.ravel()])

    return (chromatograph_model,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶️ Lancer la simulation")

    mo.vstack([
        run_button
    ])
    return (run_button,)


@app.cell
def _(
    C_feed,
    K,
    M,
    N,
    Q,
    V_col,
    chromatograph_model,
    eps,
    mo,
    np,
    plt,
    q_max,
    run_button,
    solve_ivp,
    t_ext,
    t_final,
    t_i,
    t_inj,
):
    mo.stop(not run_button.value)

    # --------------------------------------------------
    # Conversion widgets → floats / numpy arrays
    # --------------------------------------------------
    N_val = int(N.value)
    Q_val = float(Q.value)
    V_col_val = float(V_col.value)
    eps_val = float(eps.value)
    t_inj_val = float(t_inj.value)
    t_final_val = float(t_final.value)

    C_feed_arr = np.array([float(w.value) for w in C_feed], dtype=float)
    K_arr = np.array([float(w.value) for w in K], dtype=float)
    q_max_arr = np.array([float(w.value) for w in q_max], dtype=float)
    t_i_arr = np.array([float(w.value) for w in t_i], dtype=float)
    t_ext_arr = np.array([float(w.value) for w in t_ext], dtype=float)

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------
    V_cell = V_col_val / N_val
    C0 = np.zeros(N_val * M * 3)

    sol = solve_ivp(
        fun=lambda t, y: chromatograph_model(
            t,
            y,
            N_val,
            Q_val,
            V_cell,
            eps_val,
            q_max_arr,
            K_arr,
            C_feed_arr,
            t_inj_val,
            t_i_arr,
            t_ext_arr,
        ),
        t_span=(0, t_final_val),
        y0=C0,
        method="BDF",
        atol=1e-12,
        rtol=1e-5,
    )

    times = sol.t
    C_all = sol.y[: N_val * M].T.reshape((-1, N_val, M))
    C_out = C_all[:, -1]

    # Plot

    fig, ax = plt.subplots(figsize=(8, 5))

    for j in range(M):
        ax.plot(times, C_out[:, j], label=f"Espèce {j+1}")

    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Concentration sortie")
    ax.set_title("Courbes d'élution")
    ax.grid(True)
    ax.legend()

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
