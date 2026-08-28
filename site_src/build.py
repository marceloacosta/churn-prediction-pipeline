"""Build the Churn Prediction Pipeline course site from the module notebooks."""
import base64, html, json, pathlib, re, shutil

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site"
REPO = "marceloacosta/churn-prediction-pipeline"
SUBSTACK = "https://buildwithaws.substack.com/"

STAGES = [
    ("01-data-contracts", "Data contracts and mapping", "Every client names things differently. Standardise first."),
    ("02-schema-validation", "Schema validation and cleaning", "Reject bad rows loudly instead of training on them."),
    ("03-feature-engineering", "Feature engineering", "Turn raw columns into signals a model can use."),
    ("04-training-fundamentals", "Training and scoring", "Fit XGBoost and produce ranked risk scores."),
    ("05-evaluation", "Evaluation gates", "Decide what is good enough to ship."),
    ("06-drift-monitoring", "Drift monitoring", "Notice when the world stops matching your training data."),
    ("07-llm-integration", "LLM integration", "Bedrock for column mapping and plain-English reasons."),
]
SOON = [
    ("SageMaker Pipelines", "Move the whole thing onto managed AWS infrastructure."),
    ("MLflow tracking", "Version experiments, parameters and models."),
    ("End-to-end production", "Wire it together and run it for real."),
]

MD = markdown.Markdown(extensions=["fenced_code", "tables", "attr_list"])
FMT = HtmlFormatter(nowrap=True)


def render_markdown(text):
    MD.reset()
    return MD.convert(text)


def render_outputs(cell):
    out = []
    for o in cell.get("outputs", []):
        data = o.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"]
            b64 = b64 if isinstance(b64, str) else "".join(b64)
            out.append(f'<img class="nb-img" src="data:image/png;base64,{b64.strip()}" alt="">')
        elif "text/html" in data:
            out.append(f'<div class="nb-html">{"".join(data["text/html"])}</div>')
        elif "text/plain" in data:
            out.append(f'<pre class="nb-out">{html.escape("".join(data["text/plain"]))}</pre>')
        elif o.get("output_type") == "stream":
            out.append(f'<pre class="nb-out">{html.escape("".join(o.get("text", [])))}</pre>')
        elif o.get("output_type") == "error":
            tb = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(o.get("traceback", [])))
            out.append(f'<pre class="nb-out nb-err">{html.escape(tb)}</pre>')
    return "\n".join(out)


def render_notebook(path):
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    # Drop the Colab-only "save a copy in Drive" banner: it has no meaning on the web.
    if cells and cells[0]["cell_type"] == "markdown" and "Save a copy in Drive" in "".join(cells[0]["source"]):
        cells = cells[1:]
    parts, title = [], None
    for cell in cells:
        src = "".join(cell["source"]).strip()
        if not src:
            continue
        if cell["cell_type"] == "markdown":
            if title is None and src.startswith("# "):
                title = src.split("\n", 1)[0][2:].strip()
                src = src.split("\n", 1)[1] if "\n" in src else ""
                if not src.strip():
                    continue
            parts.append(f'<div class="nb-md">{render_markdown(src)}</div>')
        else:
            code = highlight(src, PythonLexer(), FMT)
            parts.append(
                '<div class="nb-code"><div class="nb-code-bar">python'
                '<button class="nb-copy" type="button">Copy</button></div>'
                f'<pre><code>{code}</code></pre></div>'
            )
            outs = render_outputs(cell)
            if outs:
                parts.append(f'<div class="nb-outwrap">{outs}</div>')
    return title, "\n".join(parts)


def rail(active):
    """The signature element: chapters drawn as stages of one pipeline."""
    rows = []
    for i, (slug, name, _) in enumerate(STAGES, 1):
        state = "done" if active and i < active else ("here" if i == active else "next")
        rows.append(
            f'<a class="stage {state}" href="../{slug}/index.html">'
            f'<span class="node"></span><span class="num">{i:02d}</span>'
            f'<span class="label">{html.escape(name)}</span></a>'
        )
    rows.append(
        '<a class="stage soon" href="../coming-soon/index.html">'
        '<span class="node"></span><span class="num">08</span>'
        '<span class="label">SageMaker, MLflow, production</span></a>'
    )
    return f'<nav class="rail">{"".join(rows)}</nav>'


SHELL = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{root}assets/style.css">
</head><body>
<header class="top">
  <a class="brand" href="{root}index.html"><span class="brand-mark"></span>Churn Prediction Pipeline</a>
  <div class="top-right"><a href="https://github.com/{repo}">GitHub</a><a class="cta" href="{substack}">Subscribe</a></div>
</header>
<div class="shell">
  <aside class="side">{rail}</aside>
  <main class="main">{body}</main>
</div>
<footer class="foot">A free course from <a href="{substack}">Build with AWS</a>.</footer>
<script>
document.querySelectorAll('.nb-copy').forEach(function(b){{
  b.addEventListener('click', function(){{
    var c = b.closest('.nb-code').querySelector('code');
    navigator.clipboard.writeText(c.innerText).then(function(){{
      b.textContent='Copied'; setTimeout(function(){{b.textContent='Copy';}},1400);
    }});
  }});
}});
</script>
</body></html>"""


def page(path, title, desc, body, active=None, root="../"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SHELL.format(title=html.escape(title), desc=html.escape(desc), body=body,
                                 rail=rail(active).replace('href="../', f'href="{root}'), root=root, repo=REPO, substack=SUBSTACK))


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for i, (slug, name, blurb) in enumerate(STAGES, 1):
        nb_path = next((ROOT / "modules" / slug).glob("*.ipynb"))
        nb_title, body = render_notebook(nb_path)
        colab = f"https://colab.research.google.com/github/{REPO}/blob/main/modules/{slug}/{nb_path.name}"
        prev_link = (f'<a class="pager-prev" href="../{STAGES[i-2][0]}/index.html">Previous stage</a>' if i > 1 else "")
        nxt = (f'<a class="pager-next" href="../{STAGES[i][0]}/index.html">Next stage</a>' if i < len(STAGES)
               else '<a class="pager-next" href="../coming-soon/index.html">What is next</a>')
        head = f"""<div class="lesson-head">
  <p class="eyebrow">Stage {i:02d} of 10</p>
  <h1>{html.escape(nb_title or name)}</h1>
  <p class="lede">{html.escape(blurb)}</p>
  <div class="runbox">
    <p>This page is for reading. To run the code, open the notebook in Colab, then use <b>File &rsaquo; Save a copy in Drive</b> so your changes stick.</p>
    <a class="run" href="{colab}" target="_blank" rel="noopener">Run stage {i:02d} in Colab</a>
  </div>
</div>"""
        page(OUT / slug / "index.html", f"{nb_title or name} — Churn Prediction Pipeline", blurb,
             head + f'<article class="prose">{body}</article>'
             + f'<div class="pager">{prev_link}{nxt}</div>', active=i)

    soon_rows = "".join(
        f'<li><span class="num">{i:02d}</span><div><h3>{html.escape(n)}</h3><p>{html.escape(d)}</p></div></li>'
        for i, (n, d) in enumerate(SOON, 8))
    page(OUT / "coming-soon" / "index.html", "Coming soon — Churn Prediction Pipeline",
         "Stages 8 to 10 take the pipeline to production on AWS.",
         f"""<div class="lesson-head"><p class="eyebrow">Stages 08 to 10</p>
<h1>Not written yet</h1>
<p class="lede">The first seven stages give you a pipeline that works on your machine. These three put it in production.</p></div>
<ol class="soon-list">{soon_rows}</ol>
<div class="runbox"><p>New stages go out to the Build with AWS list first.</p>
<a class="run" href="{SUBSTACK}">Get an email when they land</a></div>""")

    cards = "".join(
        f'<a class="card" href="{slug}/index.html"><span class="num">{i:02d}</span>'
        f'<h3>{html.escape(n)}</h3><p>{html.escape(b)}</p></a>'
        for i, (slug, n, b) in enumerate(STAGES, 1))
    page(OUT / "index.html", "Churn Prediction Pipeline — from messy CSV to production on AWS",
         "A free hands-on course: build a churn prediction system step by step, from a dirty CSV to a production pipeline on AWS.",
         f"""<div class="hero">
  <p class="eyebrow">Free course &middot; 7 of 10 stages published</p>
  <h1>A churn model is easy.<br>The pipeline around it is the job.</h1>
  <p class="lede">Start with one dirty CSV from a client. Finish with a system that validates it, scores it, explains itself and tells you when it has gone stale. Every stage is a notebook you can read here and run in Colab.</p>
  <div class="hero-cta"><a class="run" href="{STAGES[0][0]}/index.html">Start with stage 01</a>
  <a class="ghost" href="https://github.com/{REPO}">View the repo</a></div>
</div>
<section class="block"><h2>What you end up with</h2>
<p>A pipeline that takes any client's raw export, maps their column names onto a shared contract, drops the rows that would poison training, builds features, fits an XGBoost model behind evaluation gates, ranks customers by risk with a readable reason attached, and watches for drift once it is live. Bedrock does the column mapping and writes the explanations.</p>
<p class="stack"><b>Stack</b> Python 3.11, pandas, scikit-learn, XGBoost, SHAP, SageMaker, S3, Amazon Bedrock, MLflow</p></section>
<section class="block"><h2>The stages</h2><div class="cards">{cards}</div>
<a class="card soon-card" href="coming-soon/index.html"><span class="num">08</span>
<h3>SageMaker, MLflow, production</h3><p>Not written yet. Subscribe and they land in your inbox.</p></a></section>""",
         root="")
    print(f"Built {len(list(OUT.rglob('*.html')))} pages")


if __name__ == "__main__":
    build()
