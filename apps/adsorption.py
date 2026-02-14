import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import marimo as mo

    return mo, np, plt, solve_ivp


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Chromatographie d’adsorption

    ## Objectif du modèle

    Ce notebook simule la séparation de deux espèces chimiques dans une colonne
    chromatographique. Le modèle décrit l’évolution des concentrations dans la phase
    liquide et solide en tenant compte :

    - de la convection dans la colonne
    - du transfert de masse externe (film liquide)
    - du transfert de masse interne (diffusion dans les particules)
    - de l’équilibre d’adsorption compétitif

    L’objectif est de prédire les **courbes d’élution** en sortie de colonne.

    ---

    ## Hypothèses de modélisation

    Le modèle repose sur plusieurs hypothèses classiques en génie des procédés :

    ### Colonne discrétisée en cellules de mélange
    La colonne est représentée par **N cellules parfaitement mélangées en série**
    (modèle tanks-in-series).

    Chaque cellule contient :

    - une phase liquide (bulk)
    - une phase solide poreuse
    - une surface d’interface liquide–solide

    Cela approxime un réacteur piston avec dispersion.

    ---

    ### Convection axiale uniquement
    Le transport axial est modélisé par :

    - un débit volumique constant
    - pas de diffusion axiale
    - mélange parfait dans chaque cellule

    Le flux entre cellules provient uniquement du débit.

    ---

    ### Résistances de transfert de masse

    Deux résistances sont modélisées.

    #### Transfert externe (film liquide)
    La résistance entre phase liquide et surface du solide est modélisée par :

    \[
    \frac{C - C_s}{t_{ext}}
    \]

    où :

    - \(C\) : concentration bulk
    - \(C_s\) : concentration à la surface
    - \(t_{ext}\) : temps caractéristique du film

    ---

    #### Transfert interne (diffusion dans les particules)
    La diffusion interne est modélisée par le modèle **LDF (Linear Driving Force)** :

    \[
    \frac{dq}{dt} = \frac{q_{eq}(C_s) - q}{t_i}
    \]

    Ce modèle approxime la diffusion dans les pores par une loi linéaire plus simple.

    ---

    ### Équilibre d’adsorption : isotherme de Langmuir compétitive

    Les espèces se disputent un nombre limité de sites d’adsorption :

    \[
    q_i =
    \frac{q_{max,i} K_i C_i}{1 + \sum_j K_j C_j}
    \]

    Conséquences physiques :

    - saturation des sites
    - compétition entre espèces
    - comportement non linéaire
    - modification de la forme des pics chromatographiques

    ---

    ## Bilans de matière

    Le modèle repose sur trois bilans couplés.

    ### Phase liquide (bulk)

    \[
    \varepsilon V \frac{dC}{dt}
    =
    Q(C_{amont}-C)
    -
    \frac{(1-\varepsilon)V}{t_{ext}}(C-C_s)
    \]

    - convection entre cellules
    - transfert vers le solide

    ---

    ### Surface du solide

    \[
    \frac{dC_s}{dt}
    =
    \frac{C - C_s}{t_{ext}}
    -
    \frac{1-\varepsilon}{\varepsilon}\frac{dq}{dt}
    \]

    ---

    ### Phase adsorbée

    \[
    \frac{dq}{dt} = \frac{q_{eq}(C_s) - q}{t_i}
    \]

    ---

    ## Nature numérique du problème

    Le système obtenu est

    - est fortement couplé
    - contient plusieurs échelles de temps
    - est **raide (stiff)**

    La résolution utilise donc la méthode implicite **BDF**.

    ---

    ## Résultat simulé

    La simulation produit :

    - les concentrations en sortie de colonne
    - les pics d’élution
    - l’influence des paramètres physico-chimiques

    La largeur et la séparation des pics dépendent notamment de :

    - la compétition d’adsorption
    - les temps de transfert
    - la porosité
    - le débit

    ---
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
            mo.ui.number(1.0)
        )
        K.append(
            mo.ui.number(2.0)
        )
        q_max.append(
            mo.ui.number(1.0)
        )
        t_i.append(
            mo.ui.number(1.0)
        )
        t_ext.append(
            mo.ui.number(1.0)
        )

    species_table = mo.hstack([
        mo.vstack([mo.md("**Espèce**"), *espece]),
        mo.vstack([mo.md("**C_feed**"), *C_feed]),
        mo.vstack([mo.md("**K**"), *K]),
        mo.vstack([mo.md("**q_max**"), *q_max]),
        mo.vstack([mo.md("**t_i**"), *t_i]),
        mo.vstack([mo.md("**t_ext**"), *t_ext]),
    ])

    species_table
    return C_feed, K, M, q_max, t_ext, t_i


@app.cell
def _(mo):
    # ==================================================
    # 2️⃣ Paramètres procédé
    # ==================================================
    mo.md("### 🏭 Paramètres procédé")

    Q = mo.ui.number(1.0, label="Débit Q (mL/min)")
    V_col = mo.ui.number(50.0, label="Volume colonne (mL)")
    N = mo.ui.number(50, label="Nombre cellules")
    eps = mo.ui.number(0.4, label="Porosité externe")
    t_inj = mo.ui.number(2.0, label="Temps injection")
    t_final = mo.ui.number(200.0, label="Temps final")

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
        mo.md("## ▶️ Exécution"),
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
