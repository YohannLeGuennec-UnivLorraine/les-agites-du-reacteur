import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Modélisation d'un réacteur parfaitement agité en régime transitoire
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie I - Enoncé
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    On considère un réacteur parfaitement agité de volume $V$ contenant initialement un espèce inerte. A $t=0$, le réacteur est alimenté à un débit volumique $Q$ par une solution contenant les espèces $A$ et $B$ à des concentrations respectives de $C_{Ae}$ et $C_{Ae}$ $mol/L$. La réaction $A+2B \rightarrow C$ a lieu dans le réacteur. L'expression de la vitesse volumique est $r=k_1 C_A C_B^2 - k_2 C_C$. On souhaite calculer l'évolution temporelle du profil de concentration dans le réacteur.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie II - Modélisation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Le bilan sur une espèce $i\in \left(A,B,C\right)$ s'écrit :

    $$QC_{i,e} + \nu_i r V = QC_i + V\frac{dC_i}{dt}$$

    Le système différentiel à résoudre s'écrit donc :

    $$\left\{
    \begin{aligned}
    &QC_{A,e} - \left(k_1 C_A C_B^2 - k_2 C_C\right) V = QC_A + V\frac{dC_A}{dt} \\
    &QC_{B,e} -2 \left(k_1 C_A C_B^2 - k_2 C_C\right) V = QC_B + V\frac{dC_B}{dt} \\
    &QC_{C,e} + \left(k_1 C_A C_B^2 - k_2 C_C\right) V = QC_C + V\frac{dC_C}{dt} \\
    \end{aligned}
    \right.$$

    Le système peut être réécrit en faisant apparaître le temps de séjour $\tau = V/Q$ :

    $$\left\{
    \begin{aligned}
    &\frac{dC_A}{dt} = \frac{C_{A,e}-C_A}{\tau} - \left(k_1 C_A C_B^2 - k_2 C_C\right)\\
    &\frac{dC_B}{dt} = \frac{C_{B,e}-C_B}{\tau} - 2\left(k_1 C_A C_B^2 - k_2 C_C\right)\\
    &\frac{dC_C}{dt} = \frac{C_{C,e}-C_C}{\tau} + \left(k_1 C_A C_B^2 - k_2 C_C\right)\\
    \end{aligned}
    \right.$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie III - Simulation
    """)
    return


@app.cell
def _(mo):
    tau_slider = mo.ui.slider(start=0.1, stop=3.0, step=0.1, value = 1.0)
    mo.md(f"Temps de séjour $\\tau$ : {tau_slider}")
    return (tau_slider,)


@app.cell
def _(mo):
    CAe_slider = mo.ui.slider(start=0.0, stop=10, step=0.1, value = 1.0)
    mo.md(f"Concentration $C_{{Ae}}$ : {CAe_slider}")
    return (CAe_slider,)


@app.cell
def _(mo):
    CBe_slider = mo.ui.slider(start=0.0, stop=10, step=0.1, value=1.0)
    mo.md(f"Concentration $C_{{Be}}$ : {CBe_slider}")
    return (CBe_slider,)


@app.cell
def _(mo):
    k1_slider = mo.ui.slider(start=0.0, stop=10, step=0.1, value=1.0)
    mo.md(f"Constante $k_1$ ➡️ : {k1_slider}")
    return (k1_slider,)


@app.cell
def _(mo):
    k2_slider = mo.ui.slider(start=0.0, stop=10, step=0.1, value=1.0)
    mo.md(f"Constante $k_2$ ⬅️ : {k2_slider}")
    return (k2_slider,)


@app.cell
def _(mo):
    mo.md(rf"""
    La résolution du système différentiel conduit aux courbes suivantes :
    """)
    return


@app.cell
def _(CAe_slider, CBe_slider, k1_slider, k2_slider, tau_slider):
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    def rate(C):
        k1 = k1_slider.value
        k2 = k2_slider.value
        r = k1 * C[0] * C[1] ** 2 - k2 * C[2]
        return r


    def ode(t, C):
        r = rate(C)
        tau = tau_slider.value
        CAe = CAe_slider.value
        CBe = CBe_slider.value
        CCe = 0.0
        dCdt = [0.0, 0.0, 0.0]
        dCdt[0] = (CAe - C[0]) / tau - 1.0 * r
        dCdt[1] = (CBe - C[1]) / tau - 2.0 * r
        dCdt[2] = (CCe - C[2]) / tau + 1.0 * r
        return dCdt


    def solve_ode():
        t_ini = 0.0
        tau = tau_slider.value
        t_end = 10.0 * tau
        CA0 = 0.0 # mol/L
        CB0 = 0.0 # mol/L
        CC0 = 0.0 # mol/L
        C0 = [CA0, CB0, CC0]
        sol = solve_ivp(ode, [t_ini, t_end], C0)
        return sol



    def plot_ode_results():
        sol = solve_ode()

        fig, ax = plt.subplots() # On crée explicitement la figure
        ax.plot(sol.t, sol.y[0], label="A")
        ax.plot(sol.t, sol.y[1], label="B")
        ax.plot(sol.t, sol.y[2], label="C")
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration (mol/L)")

        return fig



    plot_ode_results()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Partie IV - Code Python

    Le code suivant est utilisé afin de générer le graphique ci-dessus.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```python
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    def rate(C):
        k1 = k1_slider.value
        k2 = k2_slider.value
        r = k1 * C[0] * C[1] ** 2 - k2 * C[2]


    def ode(t, C):
        r = rate(C)
        tau = tau_slider.value
        CAe = CAe_slider.value
        CBe = CBe_slider.value
        CCe = 0.0
        dCdt = [0.0, 0.0, 0.0]
        dCdt[0] = (CAe - C[0]) / tau - 1.0 * r
        dCdt[1] = (CBe - C[1]) / tau - 2.0 * r
        dCdt[2] = (CCe - C[2]) / tau + 1.0 * r



    def solve_ode():
        t_ini = 0.0
        tau = tau_slider.value
        t_end = 10.0 * tau
        CA0 = 0.0 # mol/L
        CB0 = 0.0 # mol/L
        CC0 = 0.0 # mol/L
        C0 = [CA0, CB0, CC0]
        sol = solve_ivp(ode, [t_ini, t_end], C0)



    def plot_ode_results():
        sol = solve_ode()

        plt.figure()
        plt.plot(sol.t, sol.y[0], label="A")
        plt.plot(sol.t, sol.y[1], label="B")
        plt.plot(sol.t, sol.y[2], label="C")
        plt.legend(loc="upper left")
        plt.xlabel("Time (s)")
        plt.ylabel("Concentration (mol/L)")



    plot_ode_results()
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
