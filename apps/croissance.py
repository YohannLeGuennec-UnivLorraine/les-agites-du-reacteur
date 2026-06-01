import marimo

__generated_with = "unknown"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md("#Modélisation de la croissance bactérienne instantanée au sein d'un bioréacteur")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie I - Énoncé

    Couramment utilisé dans les domaines de la biotechnologie et de la médecine, les bioréacteurs sont des dispositifs conçus pour le développement de culture cellulaire ou de micro-organismes grâce à un envrionnement contrôlé. Les bioréacteurs maintiennent les paramètres clés à leur developppement tel que le pH, la température, l'apport en substrat et oxygène afin de maximiser leur activité : la croissance et la production de composés cibles. Ces composés sont valorisés dans les industries pharmaceutiques, pour la production de médicaments ou de vaccins, et dans les industries alimentaires, pour la production de bières ou de yaourts.

    La vitesse de croissance des micro-organismes $\mu$ peut être mesuré par la **Loi de Monod**

    L'équation de Monod est un modèle mathématique qui modélise la vitesse de croissance de la biomasse microbienne en milieu aqueux en fonction de la concentration en substrat, le facteur limitant.

    L'**objectif** de ce notebook est de comprendre l'importance de l'**influence de la concentration en substrat** sur une **croissance bactérienne** : le substrat est un ensemble de molécule nécessaire à la construction d'une cellule ( la nourriture du micro-organisme ), paramètre que peut modifier l'expérimentateur.
    """)
    return


@app.cell
def _(mo):
    from IPython.display import display
    from PIL import Image
    import requests
    from io import BytesIO


    urlmonod = "https://img-4.linternaute.com/5fiT4_UKOe0_BUFax-_XUAdg0Pg=/1500x/smart/bd95ac84b8f540cea98907ae9a48794e/ccmcms-linternaute/18774071.jpg"
    responsemonod = requests.get(urlmonod)
    imgmonod = Image.open(BytesIO(responsemonod.content))

    mo.image(urlmonod,width = 300)

    display(imgmonod)

    return BytesIO, Image, display, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lien vers la loi de Monod : https://fr.wikipedia.org/wiki/%C3%89quation_de_Monod
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partie II - Modélisation

    L'équation de Monod peut être décrite selon la formule suivante :
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \[
    \mu = \mu_{\max} \cdot \frac{[S]}{K_S + [S]}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    où :

    - $μ$ ( s^-1^ ) est la vitesse de croissance des micro-organismes considérés;
    - $μmax$ ( s^-1^ ) est la vitesse de croissance maximale de ces micro-organismes;
    - $[S]$ ( g · L^-1^ ) est la concentration du substrat S limitant la croissance des micro-organismes considérés (carence, pénurie, en substrat limitant, p. ex. le phosphate indispensable à la synthèse de l'ATP);
    - $Ks$ est la constante de demi-vitesse c'est-à-dire la valeur de [S] quand μ/μmax vaut 0,5.
    """)
    return


@app.cell
def _(BytesIO, Image, display, mo, requests):
    urlloi = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Monod_3.svg/500px-Monod_3.svg.png"
    responseloi = requests.get(urlloi)
    imgloi = Image.open(BytesIO(responseloi.content))

    mo.image(urlloi,width = 300)

    display(imgloi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Vitesse de croissance des micro-organismes considérés en fonction de la concentration en substrat $[S]$ limitant cette croissance.
    """)
    return


@app.cell
def _(BytesIO, Image, display, requests):

    url = "https://www.meer.com/attachments/28778dfe6824d81efff6b9cf26eda629404dd66c/store/fill/330/186/39158ea0c917f14dbf773ae9339a3966c534e3791febf0a87e58a78de64f/Bacteries-Au-debut-de-sa-carriere-le-biochimiste-et-microbiologiste-Jacques-Monod-soccupe-de.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))



    # Afficher l'image
    display(img)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    De gauche à droite :
    $E. Coli$ — Bactérie intestinale | $S. Cerevisae$ —  Levure de bière | $T.Pallidum$ — Bactérie responsable de la syphilis | $C.Albicans$ — Champignon pathogène
    $S. Aerus$ — Bactérie pathogène | $Paramecium$ — Eucaryote unicellulaire ( Organisme avec un noyau ) | $Cyanobactérie$ — Bactérie photosynthétique | $S. Cerevisae$ ( Autre forme )
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Partie III - Simulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On cherche à montrer que la croissance dépend du substrat, elle n'est jamais infinie. La vitesse de croissance de la bactérie dépend de la disponibilité en nutriments. La variation de $[S]$ dans cette exemple permet de l'imager.
    """)
    return


@app.cell
def _(mo):
    s_slider = mo.ui.slider(0.1, 10, step=0.1, value = 1.0)
    mo.md(f"Concentration de substrat $[S]$ : {s_slider}")
    return (s_slider,)


@app.cell
def _(mo):
    mumax_slider = mo.ui.slider(0.01, 1, step=0.01, value = 0.05)
    mo.md(f"Vitesse de croissance maximale $\mu_{{max}}$ : {mumax_slider}")
    return (mumax_slider,)


@app.cell
def _(mo):
    Ks_slider = mo.ui.slider(0.01, 1, step=0.01, value = 0.05)
    mo.md(f"Constante de demi-vitesse $Ks$ : {Ks_slider}")
    return (Ks_slider,)


@app.cell
def _(mo):
    mo.md(r"""
    En réalité, quels sont les facteurs de variation des paramètres?
    - $[S]$ = Contrôle de l'alimentation ou de la carence du milieu
    - $\mu_{max}$ = La hauteur de la courbe dépend de l'espèce et de son état physiologique : On peut modifier la température, l'exposition à la lumière ou l'oxygénation pour favoriser la croissance maximale.
    - $Ks$ = La sensibilité du substrat : Plus $Ks$ est faible, plus la croissance est rapide même à faible concentration (Dépend de la disponibilité du substrat)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    En modifiant les paramètres, il est possible de dresser le graphique suivant :
    """)
    return


@app.cell
def _(Ks_slider, mumax_slider, s_slider):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Loi de Monod ---
    def monod_curve(S, mumax, Ks):
        return mumax * S / (Ks + S)

    # --- Fonction pour tracer la courbe ---
    def plot_monod():
        s = s_slider.value       # Valeur du curseur [S]
        mumax = mumax_slider.value
        Ks = Ks_slider.value

        S = np.linspace(0, 10, 500)
        mu = monod_curve(S, mumax, Ks)
        mu_at_s = monod_curve(s, mumax, Ks)

        plt.figure(figsize=(8,5))
        plt.plot(S, mu, color='green', lw=2, label='μ([S])')
        plt.axvline(s, color='blue', linestyle='--', label=f'[S]={s:.2f}')
        plt.scatter(s, mu_at_s, color='red', zorder=5, label=f'μ={mu_at_s:.2f}')

        plt.xlabel('[S] (substrat)')
        plt.ylabel('μ (vitesse de croissance)')
        plt.title(f'Loi de Monod : μmax={mumax}, Ks={Ks}')

        # --- Fixer les ordonnées pour voir la vraie variation ---
        plt.ylim(0, 1.2)  # Plage fixe, pas dépendante de μmax
        plt.grid(True)
        plt.legend()
        return plt.gcf()



    # --- Tracer initialement ---
    plot_monod()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Conclusion

    Quand le substrat est rare, les bactéries manquent de nourritures, quand il est abondant, leur croissance atteint une limite biologique ( plateau de saturation ) : La loi de Monod relie bien mathématiquement la croissance microbienne à la disponibilité en substrat.

    Contrairement à ce que l'on pourrait penser, l'ajout de nourriture n'est pas toujours synonyme de croissance bactérienne, surtout à grande concentration.

    $K_s$ mesure la capacité d'un micro-organisme à croître lorsque le substrat rare, plus il est petit, plus la croissance de l'organisme est efficace ( rapide ) et inversement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Partie IV - Code Python

    Le code utilisé ci-dessus pour afficher le graphique est le suivant :

    ```python
    import numpy as np
    import matplotlib.pyplot as plt


    def monod_curve(S, mumax, Ks):
        return mumax * S / (Ks + S)


    def plot_monod():
        s = s_slider.value       # Valeur du curseur [S]
        mumax = mumax_slider.value
        Ks = Ks_slider.value

        S = np.linspace(0, 10, 500)
        mu = monod_curve(S, mumax, Ks)
        mu_at_s = monod_curve(s, mumax, Ks)

        plt.figure(figsize=(8,5))
        plt.plot(S, mu, color='green', lw=2, label='μ([S])')
        plt.axvline(s, color='blue', linestyle='--', label=f'[S]={s:.2f}')
        plt.scatter(s, mu_at_s, color='red', zorder=5, label=f'μ={mu_at_s:.2f}')

        plt.xlabel('[S] (substrat)')
        plt.ylabel('μ (vitesse de croissance)')
        plt.title(f'Loi de Monod : μmax={mumax}, Ks={Ks}')

        plt.ylim(0, 1.2)  # Plage fixe, pas dépendante de μmax
        plt.grid(True)
        plt.legend()
        return plt.gcf()


    plot_monod()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perspectives
    La loi de Monod n'est qu'une approximation, d'autres paramètres sont à prendre en compte. Elle n'étudie la vitesse de croissance qu'à un instant $t$.

    Dans les bioprocédés, connaître l'évolution des produits et des substrats au cours du temps est primordial en vue de l'optimisation de la taille des équipements utilisés et de la quantité de produits. Pour comprendre l'évolution d'une culture microbienne, il est nécessaire de coupler la loi de Monod avec un bilan de matières.

    *voir Notebook "Modélisation de la croissance bactérienne dynamique au sein d'un bioréacteur"*
    """)
    return


if __name__ == "__main__":
    app.run()