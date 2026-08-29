"""Build the Churn Prediction Pipeline course site from the module notebooks."""
import base64, html, json, pathlib, re, shutil

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site"
# Only used to build Colab URLs. Colab opens a public notebook by its GitHub path,
# so this is machinery for the Run buttons, not a link to the repo.
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
# The rail draws the pipeline itself, so credential setup gets its own entry outside
# the numbering, beside the first stage that needs it.
SETUP = ("setup-aws-credentials", "AWS credentials",
         "One Bedrock API key, stored in Colab Secrets. Needed from stage 07 on.")
SETUP_BEFORE = 7

# Each upcoming stage gets its own page. One shared "coming soon" page meant clicking
# MLflow tracking landed you on a description of all three.
SOON = [
    ("08-sagemaker-pipelines", "SageMaker Pipelines",
     "Move the whole thing onto managed AWS infrastructure.",
     "Everything so far runs on one machine, in order, because you pressed shift-enter. "
     "This stage defines the same steps as a SageMaker Pipeline: each one gets its own "
     "instance, its inputs and outputs live in S3, and a full run becomes a single API "
     "call you can trigger from anywhere."),
    ("09-mlflow-tracking", "MLflow tracking",
     "Version experiments, parameters and models.",
     "Today a training run leaves an AUC in a notebook output and nothing else. Once a "
     "model is in front of someone, you need to answer what changed between the version "
     "you shipped and the one you are looking at now. This stage records the parameters, "
     "the metrics and the model artifact for every run, so that question has an answer."),
    ("10-end-to-end-production", "End-to-end production",
     "Wire it together and run it for real.",
     "The last stage joins the pieces: a schedule, the S3 layout that keeps one client's "
     "data away from another's, the drift job from stage 06 running on a cadence instead "
     "of on demand, and what is supposed to happen when a run fails at three in the "
     "morning."),
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
    # Everything before the chapter title is Colab plumbing: the read-only banner,
    # the pip install, and the cells that replay earlier chapters. Needed to run the
    # notebook, pure noise when you are only reading it.
    start = next((i for i, c in enumerate(cells)
                  if c["cell_type"] == "markdown" and "".join(c["source"]).lstrip().startswith("# ")), 0)
    cells = cells[start:]
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
        if i == SETUP_BEFORE:
            rows.append(
                f'<a class="stage setup{" here" if active == "setup" else ""}"'
                f' href="../{SETUP[0]}/index.html">'
                f'<span class="node"></span><span class="num">&middot;&middot;</span>'
                f'<span class="label">{html.escape(SETUP[1])}</span></a>'
            )
        if active == "setup":
            state = "done" if i < SETUP_BEFORE else "next"
        else:
            state = "done" if active and i < active else ("here" if i == active else "next")
        rows.append(
            f'<a class="stage {state}" href="../{slug}/index.html">'
            f'<span class="node"></span><span class="num">{i:02d}</span>'
            f'<span class="label">{html.escape(name)}</span></a>'
        )
    for i, (slug, name, _, _) in enumerate(SOON, len(STAGES) + 1):
        rows.append(
            f'<a class="stage soon{" here" if i == active else ""}" href="../{slug}/index.html">'
            f'<span class="node"></span><span class="num">{i:02d}</span>'
            f'<span class="label">{html.escape(name)}</span></a>'
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
  <div class="top-right"><a class="cta" href="{substack}">Subscribe</a></div>
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
                                 rail=rail(active).replace('href="../', f'href="{root}'), root=root, substack=SUBSTACK))


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
               else f'<a class="pager-next" href="../{SOON[0][0]}/index.html">What is next</a>')
        creds_note = (f'<p class="creds-note">This stage calls Amazon Bedrock. '
                      f'<a href="../{SETUP[0]}/index.html">Set up your credentials first</a> \u2014 '
                      f'one API key in Colab Secrets, about five minutes.</p>' if i >= SETUP_BEFORE else "")
        head = f"""<div class="lesson-head">
  <p class="eyebrow">Stage {i:02d} of 10</p>
  <h1>{html.escape(nb_title or name)}</h1>
  <p class="lede">{html.escape(blurb)}</p>
  <div class="runbox">
    {creds_note}<p>This page is for reading, so it skips the setup cells. To run the code, open the notebook in Colab and use <b>File &rsaquo; Save a copy in Drive</b> so your changes stick.</p>
    <a class="run" href="{colab}" target="_blank" rel="noopener">Run stage {i:02d} in Colab</a>
  </div>
</div>"""
        page(OUT / slug / "index.html", f"{nb_title or name} — Churn Prediction Pipeline", blurb,
             head + f'<article class="prose">{body}</article>'
             + f'<div class="pager">{prev_link}{nxt}</div>', active=i)

    setup_slug, setup_name, setup_blurb = SETUP
    nb_path = next((ROOT / "modules" / setup_slug).glob("*.ipynb"))
    nb_title, body = render_notebook(nb_path)
    colab = f"https://colab.research.google.com/github/{REPO}/blob/main/modules/{setup_slug}/{nb_path.name}"
    page(OUT / setup_slug / "index.html", f"{nb_title or setup_name} \u2014 Churn Prediction Pipeline", setup_blurb,
         f"""<div class="lesson-head">
  <p class="eyebrow">Setup &middot; needed from stage {SETUP_BEFORE:02d}</p>
  <h1>{html.escape(nb_title or setup_name)}</h1>
  <p class="lede">{html.escape(setup_blurb)}</p>
  <div class="runbox">
    <p>Stages 01 to {SETUP_BEFORE - 1:02d} need nothing but a browser. This page only matters once you reach stage {SETUP_BEFORE:02d}, the first one that calls a real model.</p>
    <a class="run" href="{colab}" target="_blank" rel="noopener">Open the setup notebook in Colab</a>
  </div>
</div>"""
         + f'<article class="prose">{body}</article>'
         + f'<div class="pager">'
           f'<a class="pager-prev" href="../{STAGES[SETUP_BEFORE - 2][0]}/index.html">Previous stage</a>'
           f'<a class="pager-next" href="../{STAGES[SETUP_BEFORE - 1][0]}/index.html">On to stage {SETUP_BEFORE:02d}</a></div>',
         active="setup")

    total = len(STAGES) + len(SOON)
    for n, (slug, name, blurb, detail) in enumerate(SOON, len(STAGES) + 1):
        prev_slug = STAGES[-1][0] if n == len(STAGES) + 1 else SOON[n - len(STAGES) - 2][0]
        nxt = (f'<a class="pager-next" href="../{SOON[n - len(STAGES)][0]}/index.html">Next stage</a>'
               if n < total else "")
        page(OUT / slug / "index.html", f"{name} — Churn Prediction Pipeline", blurb,
             f"""<div class="lesson-head">
  <p class="eyebrow">Stage {n:02d} of {total}</p>
  <h1>{html.escape(name)}</h1>
  <p class="lede">{html.escape(blurb)}</p>
</div>
<div class="wip">
  <p class="wip-tag">Not published yet</p>
  <p>This stage is still being written.</p>
</div>
<article class="prose"><p>{html.escape(detail)}</p></article>
<div class="runbox">
  <p>Stage {n:02d} goes out to the Build with AWS list the day it is published.</p>
  <a class="run" href="{SUBSTACK}">Email me when stage {n:02d} is live</a>
</div>
<div class="pager"><a class="pager-prev" href="../{prev_slug}/index.html">Previous stage</a>{nxt}</div>""",
             active=n)

    card_items = []
    for i, (slug, n, b) in enumerate(STAGES, 1):
        if i == SETUP_BEFORE:
            card_items.append(
                f'<a class="card setup-card" href="{SETUP[0]}/index.html">'
                f'<span class="num">&middot;&middot;</span>'
                f'<h3>{html.escape(SETUP[1])}</h3><p>{html.escape(SETUP[2])}</p></a>')
        card_items.append(
            f'<a class="card" href="{slug}/index.html"><span class="num">{i:02d}</span>'
            f'<h3>{html.escape(n)}</h3><p>{html.escape(b)}</p></a>')
    cards = "".join(card_items)
    soon_cards = "".join(
        f'<a class="card soon-card" href="{slug}/index.html"><span class="num">{i:02d}</span>'
        f'<h3>{html.escape(n)}</h3><p>{html.escape(b)}</p>'
        f'<span class="card-tag">Not published yet</span></a>'
        for i, (slug, n, b, _) in enumerate(SOON, len(STAGES) + 1))
    page(OUT / "index.html", "Churn Prediction Pipeline — from messy CSV to production on AWS",
         "A free hands-on course: build a churn prediction system step by step, from a dirty CSV to a production pipeline on AWS.",
         f"""<div class="hero">
  <p class="eyebrow">Free course &middot; 7 of 10 stages published</p>
  <h1>A churn model is easy.<br>The pipeline around it is the job.</h1>
  <p class="lede">Start with one dirty CSV from a client. Finish with a system that validates it, scores it, explains itself and tells you when it has gone stale. Every stage is a notebook you can read here and run in Colab.</p>
  <div class="hero-cta"><a class="run" href="{STAGES[0][0]}/index.html">Start with stage 01</a></div>
</div>
<section class="block"><h2>What you end up with</h2>
<p>A pipeline that takes any client's raw export, maps their column names onto a shared contract, drops the rows that would poison training, builds features, fits an XGBoost model behind evaluation gates, ranks customers by risk with a readable reason attached, and watches for drift once it is live. Bedrock does the column mapping and writes the explanations.</p>
<p class="stack"><b>Stack</b> Python 3.11, pandas, scikit-learn, XGBoost, SHAP, SageMaker, S3, Amazon Bedrock, MLflow</p></section>
<section class="block"><h2>The stages</h2><div class="cards">{cards}</div>
<div class="cards">{soon_cards}</div></section>""",
         root="")
    print(f"Built {len(list(OUT.rglob('*.html')))} pages")


if __name__ == "__main__":
    build()
