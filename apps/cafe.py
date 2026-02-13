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
    from scipy.integrate import solve_ivp

    return mo, np, plt, solve_ivp


@app.cell
def _(mo):
    # =====================================================
    # 🔘 Widgets
    # =====================================================
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
    # =====================================================
    # PARAMÈTRES GÉOMÉTRIQUES
    # =====================================================
    R = 0.03
    z = 0.08

    A_paroi = 2 * np.pi * R * z
    A_top = np.pi * R**2

    # =====================================================
    # CAFÉ
    # =====================================================
    m_c = 0.25
    cp_c = 4180

    # =====================================================
    # FONCTION PARAMÈTRES
    # =====================================================
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
    # =====================================================
    # RÉCUPÉRATION PARAMÈTRES
    # =====================================================
    params = get_parameters(type_recipient.value, souffler.value)
    m_paroi, cp_paroi, k, e_paroi, h_paroi, h_top, h_cafe = params


    # =====================================================
    # SYSTÈME D'ODE
    # =====================================================
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
    # =====================================================
    # RÉSOLUTION
    # =====================================================
    t_eval = np.linspace(0, 7200, 800)

    sol = solve_ivp(
        dTdt,
        (0, 7200),
        [T0_c.value, T0_c.value],  # paroi initialement chaude
        t_eval=t_eval
    )

    Tc = sol.y[0]
    Tparoi = sol.y[1]


    # =====================================================
    # AFFICHAGE
    # =====================================================
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
