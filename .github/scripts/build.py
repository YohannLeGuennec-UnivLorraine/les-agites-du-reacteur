"""
Build script for marimo notebooks.

This script exports marimo notebooks to HTML/WebAssembly format and generates
an index.html file that lists all the notebooks. It handles both regular notebooks
(from the notebooks/ directory) and apps (from the apps/ directory).

The script can be run from the command line with optional arguments:
    uv run .github/scripts/build.py [--output-dir OUTPUT_DIR]

The exported files will be placed in the specified output directory (default: _site).
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.3",
#     "fire==0.7.0",
#     "loguru==0.7.0"
# ]
# ///

import subprocess
import shutil
import html
from typing import List, Union
from pathlib import Path

import jinja2
import fire

from loguru import logger

APP_METADATA = {
    "réacteur parfaitement agité": {
        "category": "🏭 Procédés",
        "display_name": "Réacteur parfaitement agité",
        "order": 10,
    },
    "distillation": {
        "category": "🏭 Procédés",
        "display_name": "Distillation binaire",
        "order": 20,
    },
    "croissance": {
        "category": "🧫 Bioprocédés",
        "display_name": "Cinétique de croissance bactérienne",
        "order": 30,
    },
    "bioreacteur": {
        "category": "🧫 Bioprocédés",
        "display_name": "Bioréacteur fermé",
        "order": 40,
    },
    "adsorption": {
        "category": "🧫 Bioprocédés",
        "display_name": "Chromatographie",
        "order": 50,
    },
    "cafe": {
        "category": "🍵 Transferts",
        "display_name": "Café",
        "order": 60,
    },
}

CATEGORY_ORDER = ["🏭 Procédés", "🧫 Bioprocédés", "🍵 Transferts"]


def _export_html_wasm(notebook_path: Path, output_dir: Path, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML/WebAssembly format.

    This function takes a marimo notebook (.py file) and exports it to HTML/WebAssembly format.
    If as_app is True, the notebook is exported in "run" mode with code hidden, suitable for
    applications. Otherwise, it's exported in "edit" mode, suitable for interactive notebooks.

    Args:
        notebook_path (Path): Path to the marimo notebook (.py file) to export
        output_dir (Path): Directory where the exported HTML file will be saved
        as_app (bool, optional): Whether to export as an app (run mode) or notebook (edit mode).
                                Defaults to False.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    # Convert .py extension to .html for the output file
    output_path: Path = notebook_path.with_suffix(".html")

    # Base command for marimo export
    cmd: List[str] = ["uvx", "marimo", "export", "html-wasm", "--sandbox"]

    # Configure export mode based on whether it's an app or a notebook
    if as_app:
        logger.info(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])  # Apps run in "run" mode with hidden code
    else:
        logger.info(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])  # Notebooks run in "edit" mode

    try:
        # Create full output path and ensure directory exists
        output_file: Path = output_dir / notebook_path.with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add notebook path and output file to command
        cmd.extend([str(notebook_path), "-o", str(output_file)])

        # Run marimo export command
        logger.debug(f"Running command: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully exported {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        # Handle marimo export errors
        logger.error(f"Error exporting {notebook_path}:")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def _generate_index(
    output_dir: Path,
    template_file: Path,
    notebooks_data: List[dict] | None = None,
    apps_data: List[dict] | None = None,
    app_groups: List[dict] | None = None,
) -> None:
    """Generate an index.html file that lists all the notebooks.

    This function creates an HTML index page that displays links to all the exported
    notebooks. The index page includes the marimo logo and displays each notebook
    with a formatted title and a link to open it.

    Args:
        notebooks_data (List[dict]): List of dictionaries with data for notebooks
        apps_data (List[dict]): List of dictionaries with data for apps
        output_dir (Path): Directory where the index.html file will be saved
        template_file (Path, optional): Path to the template file. If None, uses the default template.

    Returns:
        None
    """
    logger.info("Generating index.html")

    # Create the full path for the index.html file
    index_path: Path = output_dir / "index.html"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set up Jinja2 environment and load template
        template_dir = template_file.parent
        template_name = template_file.name
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"])
        )
        template = env.get_template(template_name)

        # Render the template with notebook and app data
        rendered_html = template.render(notebooks=notebooks_data, apps=apps_data, app_groups=app_groups)

        # Write the rendered HTML to the index.html file
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        logger.info(f"Successfully generated index.html at {index_path}")

    except IOError as e:
        # Handle file I/O errors
        logger.error(f"Error generating index.html: {e}")
    except jinja2.exceptions.TemplateError as e:
        # Handle template errors
        logger.error(f"Error rendering template: {e}")


def _copy_static_assets(output_dir: Path) -> None:
    """Copy static site assets, such as institutional logos, to the output directory."""
    static_dir = Path("static")
    if not static_dir.exists():
        return

    target_dir = output_dir / "static"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(static_dir, target_dir)
    logger.info(f"Copied static assets to {target_dir}")


def _inject_app_chrome(app_html_path: Path, display_name: str, category: str | None = None) -> None:
    """Add project branding around exported Marimo apps."""
    if not app_html_path.exists():
        return

    html_text = app_html_path.read_text(encoding="utf-8")
    if "data-les-agites-chrome" in html_text:
        return

    safe_title = html.escape(display_name)
    safe_category = html.escape(category or "Application interactive")

    chrome_styles = """
    <style data-les-agites-chrome>
      :root {
        --lad-blue: #1a3a52;
        --lad-blue-light: #2d5a7b;
        --lad-border: #d9e4eb;
        --text-font: "Segoe UI", Arial, sans-serif;
        --heading-font: "Segoe UI", Arial, sans-serif;
        --marimo-text-font: "Segoe UI", Arial, sans-serif;
        --marimo-heading-font: "Segoe UI", Arial, sans-serif;
      }

      body {
        margin: 0;
        background: linear-gradient(180deg, #f8fbfd 0%, #eef5f9 52%, #f8fbfd 100%);
        color: #151515;
        font-family: "Segoe UI", Arial, sans-serif;
      }

      .lad-app-header,
      .lad-app-footer {
        font-family: "Segoe UI", Arial, sans-serif;
      }

      .lad-app-header {
        position: relative;
        z-index: 1;
        padding: 18px 20px 14px;
        background: linear-gradient(180deg, #f8fbfd 0%, rgba(248, 251, 253, 0.96) 100%);
        box-shadow: 0 8px 20px rgba(26, 58, 82, 0.08);
      }

      .lad-app-header-inner {
        max-width: 1120px;
        margin: 0 auto;
      }

      .lad-app-logos {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 28px;
        margin-bottom: 18px;
        flex-wrap: wrap;
      }

      .lad-app-logos img {
        display: block;
        max-width: min(260px, 42vw);
        max-height: 76px;
        object-fit: contain;
      }

      .lad-app-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        padding: 16px 22px;
        background: linear-gradient(135deg, var(--lad-blue) 0%, var(--lad-blue-light) 100%);
        color: #ffffff;
        border-radius: 4px;
        box-shadow: 0 8px 22px rgba(26, 58, 82, 0.14);
      }

      .lad-app-title {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
      }

      .lad-app-title img {
        width: 70px;
        height: 52px;
        object-fit: contain;
        filter: brightness(0) invert(1);
        flex: 0 0 auto;
      }

      .lad-app-title p {
        margin: 0 0 2px;
        opacity: 0.86;
        font-size: 0.88rem;
        font-weight: 700;
      }

      .lad-app-title h1 {
        margin: 0;
        padding: 0;
        border: 0;
        color: #ffffff;
        font-size: clamp(1.25rem, 3vw, 1.8rem);
        line-height: 1.08;
        text-transform: none;
      }

      .lad-app-home {
        color: #ffffff;
        text-decoration: none;
        font-weight: 700;
        border-bottom: 2px solid rgba(255, 255, 255, 0.55);
        white-space: nowrap;
      }

      .lad-app-home:hover,
      .lad-app-home:focus-visible {
        border-bottom-color: #ffffff;
        outline: none;
      }

      #root {
        font-family: "Segoe UI", Arial, sans-serif;
        max-width: 1120px;
        margin: 0 auto;
        padding: 18px 20px 28px;
      }

      .lad-app-header {
        transition:
          opacity 180ms ease,
          transform 180ms ease,
          max-height 220ms ease,
          padding 220ms ease,
          margin 220ms ease;
        max-height: 260px;
        overflow: hidden;
      }

      .lad-app-header.is-hidden {
        opacity: 0;
        transform: translateY(-12px);
        max-height: 0;
        padding-top: 0;
        padding-bottom: 0;
        margin: 0;
        box-shadow: none;
        pointer-events: none;
      }

      #root h1,
      #root h2,
      #root h3,
      #root h4,
      #root p,
      #root li,
      #root table,
      #root label,
      #root button,
      #root input,
      #root textarea {
        font-family: "Segoe UI", Arial, sans-serif;
      }

      .lad-app-footer {
        max-width: 1120px;
        margin: 0 auto;
        padding: 22px 20px 30px;
        color: #606060;
        text-align: center;
        border-top: 1px solid var(--lad-border);
        font-size: 0.92rem;
      }

      @media (max-width: 760px) {
        .lad-app-banner {
          align-items: flex-start;
          flex-direction: column;
        }

        .lad-app-title {
          align-items: flex-start;
        }
      }
    </style>
    """

    app_header = f"""
    <header class="lad-app-header">
      <div class="lad-app-header-inner">
        <div class="lad-app-logos" aria-label="Logos ENSIC et Université de Lorraine">
          <a href="https://www.ensic.univ-lorraine.fr/fr" target="_blank" rel="noopener noreferrer" aria-label="Site de l'ENSIC">
            <img src="../static/logo-ensic.png" alt="ENSIC">
          </a>
          <img src="../static/logo-universite-de-lorraine.png" alt="Université de Lorraine">
        </div>
        <div class="lad-app-banner">
          <div class="lad-app-title">
            <img src="../static/logo.png" alt="Les agités du réacteur">
            <div>
              <p>{safe_category}</p>
              <h1>{safe_title}</h1>
            </div>
          </div>
          <a class="lad-app-home" href="../index.html">Retour à l'accueil</a>
        </div>
      </div>
    </header>
    """

    app_footer = """
    <footer class="lad-app-footer">
      Projet conduit par Yohann Le Guennec. Il existe grâce aux contenus proposés par les étudiants Imran Baraka, Alex Jacquey, Maxime Lin et Nitish Pierre.
    </footer>
    """

    scroll_script = """
    <script data-les-agites-scroll>
      (() => {
        const header = document.querySelector(".lad-app-header");
        const root = document.getElementById("root");
        if (!header || !root) {
          return;
        }

        const updateHeader = (scrollTop) => {
          header.classList.toggle("is-hidden", scrollTop > 24);
        };

        const offsetScrollableContent = (element) => {
          if (!element || element.dataset.ladOffsetApplied === "true") {
            return;
          }

          const rect = element.getBoundingClientRect();
          const headerBottom = header.getBoundingClientRect().bottom;
          if (rect.top >= headerBottom - 4) {
            return;
          }

          const currentPadding = parseFloat(window.getComputedStyle(element).paddingTop) || 0;
          const neededOffset = Math.ceil(headerBottom - rect.top + 18);
          element.style.setProperty("box-sizing", "border-box", "important");
          element.style.setProperty("padding-top", `${currentPadding + neededOffset}px`, "important");
          element.style.setProperty("scroll-padding-top", `${neededOffset}px`, "important");
          element.dataset.ladOffsetApplied = "true";
        };

        const attachScrollHandlers = () => {
          const candidates = [document.scrollingElement, document.documentElement, document.body];
          root.querySelectorAll("*").forEach((element) => {
            const style = window.getComputedStyle(element);
            const scrollsVertically = /auto|scroll/.test(style.overflowY) || /auto|scroll/.test(style.overflow);
            if (scrollsVertically && element.scrollHeight > element.clientHeight + 8) {
              candidates.push(element);
              if (element.clientHeight > window.innerHeight * 0.45) {
                offsetScrollableContent(element);
              }
            }
          });

          candidates.forEach((element) => {
            if (!element || element.dataset.ladScrollBound === "true") {
              return;
            }
            element.dataset.ladScrollBound = "true";
            element.addEventListener("scroll", () => updateHeader(element.scrollTop), { passive: true });
          });
        };

        attachScrollHandlers();
        setTimeout(attachScrollHandlers, 250);
        setTimeout(attachScrollHandlers, 1000);

        const observer = new MutationObserver(attachScrollHandlers);
        observer.observe(root, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ["class", "style"],
        });
      })();
    </script>
    """

    html_text = html_text.replace("</head>", f"{chrome_styles}\n</head>", 1)
    html_text = html_text.replace("<body>", f"<body>\n{app_header}", 1)
    html_text = html_text.replace("</body>", f"{scroll_script}\n{app_footer}\n</body>", 1)
    app_html_path.write_text(html_text, encoding="utf-8")


def _format_display_name(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def _apply_app_metadata(apps_data: List[dict]) -> List[dict]:
    enriched_apps = []
    for app in apps_data:
        metadata = APP_METADATA.get(app["key"], {})
        enriched_app = {
            **app,
            "category": metadata.get("category", "Autres"),
            "display_name": metadata.get("display_name", app["display_name"]),
            "order": metadata.get("order", 999),
        }
        enriched_apps.append(enriched_app)
    return sorted(enriched_apps, key=lambda app: (app["order"], app["display_name"]))


def _group_apps(apps_data: List[dict]) -> List[dict]:
    groups = []
    for category in CATEGORY_ORDER:
        items = [app for app in apps_data if app.get("category") == category]
        if items:
            groups.append({"title": category, "apps": items})

    remaining = [app for app in apps_data if app.get("category") not in CATEGORY_ORDER]
    if remaining:
        groups.append({"title": "Autres", "apps": remaining})

    return groups


def _export(folder: Path, output_dir: Path, as_app: bool=False) -> List[dict]:
    """Export all marimo notebooks in a folder to HTML/WebAssembly format.

    This function finds all Python files in the specified folder and exports them
    to HTML/WebAssembly format using the export_html_wasm function. It returns a
    list of dictionaries containing the data needed for the template.

    Args:
        folder (Path): Path to the folder containing marimo notebooks
        output_dir (Path): Directory where the exported HTML files will be saved
        as_app (bool, optional): Whether to export as apps (run mode) or notebooks (edit mode).

    Returns:
        List[dict]: List of dictionaries with "display_name" and "html_path" for each notebook
    """
    # Check if the folder exists
    if not folder.exists():
        logger.warning(f"Directory not found: {folder}")
        return []

    # Find all Python files recursively in the folder
    notebooks = list(folder.rglob("*.py"))
    logger.debug(f"Found {len(notebooks)} Python files in {folder}")

    # Exit if no notebooks were found
    if not notebooks:
        logger.warning(f"No notebooks found in {folder}!")
        return []

    # For each successfully exported notebook, add its data to the notebook_data list
    notebook_data = []
    for nb in notebooks:
        if not _export_html_wasm(nb, output_dir, as_app=as_app):
            continue

        key = nb.stem.casefold()
        metadata = APP_METADATA.get(key, {}) if as_app else {}
        display_name = metadata.get("display_name", _format_display_name(nb))
        data = {
            "key": key,
            "display_name": display_name,
            "html_path": str(nb.with_suffix(".html")),
        }
        notebook_data.append(data)

        if as_app:
            _inject_app_chrome(
                output_dir / nb.with_suffix(".html"),
                display_name=display_name,
                category=metadata.get("category"),
            )

    logger.info(f"Successfully exported {len(notebook_data)} out of {len(notebooks)} files from {folder}")
    return notebook_data

def main(
    output_dir: Union[str, Path] = "_site",
    template: Union[str, Path] = "templates/index_agite.html.j2",
) -> None:
    """Main function to export marimo notebooks.

    This function:
    1. Parses command line arguments
    2. Exports all marimo notebooks in the 'notebooks' and 'apps' directories
    3. Generates an index.html file that lists all the notebooks

    Command line arguments:
        --output-dir: Directory where the exported files will be saved (default: _site)
        --template: Path to the template file (default: templates/index.html.j2)

    Returns:
        None
    """
    logger.info("Starting marimo build process")

    # Convert output_dir explicitly to Path (not done by fire)
    output_dir: Path = Path(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_static_assets(output_dir)

    # Convert template to Path if provided
    template_file: Path = Path(template)
    logger.info(f"Using template file: {template_file}")

    # Export notebooks from the notebooks/ directory
    notebooks_data = _export(Path("notebooks"), output_dir, as_app=False)

    # Export apps from the apps/ directory
    apps_data = _apply_app_metadata(_export(Path("apps"), output_dir, as_app=True))
    app_groups = _group_apps(apps_data)

    # Exit if no notebooks or apps were found
    if not notebooks_data and not apps_data:
        logger.warning("No notebooks or apps found!")
        return

    # Generate the index.html file that lists all notebooks and apps
    _generate_index(
        output_dir=output_dir,
        notebooks_data=notebooks_data,
        apps_data=apps_data,
        app_groups=app_groups,
        template_file=template_file,
    )

    logger.info(f"Build completed successfully. Output directory: {output_dir}")


if __name__ == '__main__':
    fire.Fire(main)
