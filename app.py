"""Semantic search engine over the Google Product Type taxonomy (it-IT).

Pipeline:
  1. (optional) translate the query to Italian  -> cross-lingual FR/EN/IT
  2. lexical score  : TF-IDF char n-grams (handles singular/plural, typos)
  3. semantic score : multilingual embeddings (MiniLM) + cosine
  4. weighted RRF fusion (lexical <-> semantic slider)

Pass the product straight in the URL:  ?q=running+shoes

Free hosting: Streamlit Community Cloud (see README.md).
"""

from pathlib import Path
import html as _html

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE = Path(__file__).parent
TAXO_FILE = BASE / "taxonomy.it-IT.txt"
EMB_FILE = BASE / "embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

st.set_page_config(page_title="Recherche taxonomie Google", page_icon="🔎", layout="wide")


def dense_doc(leaf, parent):
    return f"{leaf} — {parent}" if parent else leaf


def tfidf_doc(leaf, parent):
    # feuille pondérée x2 : la catégorie réelle prime sur son contexte
    return f"{leaf} {leaf} {parent}".strip()


@st.cache_data(show_spinner=False)
def load_taxonomy():
    ids, paths, leaves, parents, tops = [], [], [], [], []
    for line in TAXO_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cat_id, sep, path = line.partition(" - ")
        if not sep or not path.strip():
            continue
        parts = [p.strip() for p in path.split(">")]
        ids.append(cat_id.strip())
        paths.append(path.strip())
        leaves.append(parts[-1])
        parents.append(" > ".join(parts[:-1]))
        tops.append(parts[0])
    return ids, paths, leaves, parents, tops


@st.cache_resource(show_spinner="Building lexical index…")
def build_tfidf(docs_tuple):
    docs = list(docs_tuple)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    X = vec.fit_transform(docs)
    return vec, X


@st.cache_resource(show_spinner="Loading semantic model…")
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Computing embeddings…")
def build_embeddings(docs_tuple):
    docs = list(docs_tuple)
    if EMB_FILE.exists():
        emb = np.load(EMB_FILE)
        if emb.shape[0] == len(docs):
            return emb.astype(np.float32)
    model = load_model()
    emb = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    try:
        np.save(EMB_FILE, emb)
    except OSError:
        pass
    return emb


@st.cache_data(show_spinner=False)
def translate_to_it(text):
    """Translate to Italian. Returns (text, ok). Silent if offline."""
    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source="auto", target="it").translate(text)
        return (out or text), True
    except Exception:
        return text, False


def hybrid_search(query, vec, X, emb, w_sem, top_k, allowed_idx=None):
    """Weighted RRF fusion of lexical and semantic ranks. w_sem in [0, 1]."""
    tf = linear_kernel(vec.transform([query]), X).ravel()
    model = load_model()
    q = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
    dense = emb @ q

    if allowed_idx is not None:
        keep = np.zeros(tf.shape, dtype=bool)
        keep[allowed_idx] = True
        tf = np.where(keep, tf, -np.inf)
        dense = np.where(keep, dense, -np.inf)

    k = 60
    fused = {}
    for rank, i in enumerate(np.argsort(-tf)):
        if not np.isfinite(tf[i]):
            break
        fused[i] = fused.get(i, 0.0) + (1 - w_sem) / (k + rank)
    for rank, i in enumerate(np.argsort(-dense)):
        if not np.isfinite(dense[i]):
            break
        fused[i] = fused.get(i, 0.0) + w_sem / (k + rank)

    order = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return order, tf, dense


_TABLE_CSS_JS = """
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; background: transparent;
         font-family: "Source Sans Pro", system-ui, -apple-system, sans-serif; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; color: #111; }
  th, td { padding: 6px 10px; text-align: left; vertical-align: top; }
  th { font-weight: 600; color: #333; border-bottom: 2px solid #ccc; white-space: nowrap; }
  td { border-bottom: 1px solid #eee; }
  /* colonnes réduites au minimum */
  .num, .id, .sc, .cp { width: 1%; white-space: nowrap; }
  .sc { text-align: right; color: #777; font-variant-numeric: tabular-nums; }
  .num { color: #999; }
  /* Path prend tout l'espace restant */
  .path { width: 100%; word-break: break-word; }
  .cpbtn { cursor: pointer; border: 1px solid #d0d0d0; border-radius: 6px;
           background: #fff; padding: 1px 7px; font-size: 0.95rem; line-height: 1.4; }
  .cpbtn:hover { background: #f0f2f6; }
  @media (prefers-color-scheme: dark) {
    table { color: #eaeaea; }
    th { color: #ddd; border-bottom-color: #555; }
    td { border-bottom-color: #333; }
    .cpbtn { background: #262730; color: #eaeaea; border-color: #555; }
    .cpbtn:hover { background: #3a3b45; }
  }
</style>
<table>
  <thead><tr>
    <th class="num">#</th><th class="id">ID</th>
    <th class="sc">sem</th><th class="sc">lex</th>
    <th class="cp"></th><th class="path">Path</th>
  </tr></thead>
  <tbody>__ROWS__</tbody>
</table>
<script>
  document.querySelectorAll('.cpbtn').forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-t'), done = false;
      try {
        var ta = document.createElement('textarea');
        ta.value = t; ta.style.position = 'fixed'; ta.style.top = '-1000px';
        document.body.appendChild(ta); ta.focus(); ta.select();
        done = document.execCommand('copy'); document.body.removeChild(ta);
      } catch (e) {}
      if (!done && navigator.clipboard) { navigator.clipboard.writeText(t); }
      b.textContent = '✓';
      setTimeout(function () { b.textContent = '📋'; }, 1200);
    });
  });
</script>
"""


def render_results(order, ids, paths, dense, tf):
    def esc(s):
        return _html.escape(str(s), quote=True)

    trs = []
    for rank, i in enumerate(order, start=1):
        trs.append(
            "<tr>"
            f'<td class="num">{rank}</td>'
            f'<td class="id">{esc(ids[i])}</td>'
            f'<td class="sc">{max(dense[i], 0) * 100:.0f}</td>'
            f'<td class="sc">{max(tf[i], 0) * 100:.0f}</td>'
            f'<td class="cp"><button class="cpbtn" data-t="{esc(paths[i])}" '
            'title="Copy path to clipboard">📋</button></td>'
            f'<td class="path">{esc(paths[i])}</td>'
            "</tr>"
        )
    html_table = _TABLE_CSS_JS.replace("__ROWS__", "".join(trs))
    components.html(html_table, height=60 + len(order) * 46, scrolling=True)


# --- UI -------------------------------------------------------------------

st.title("🔎 Category Search — Google Product Type (it-IT)")
st.caption("5,595 official categories. Type a product in Italian, French or English.")

ids, paths, leaves, parents, tops = load_taxonomy()
vec, X = build_tfidf(tuple(tfidf_doc(l, p) for l, p in zip(leaves, parents)))
emb = build_embeddings(tuple(dense_doc(l, p) for l, p in zip(leaves, parents)))
unique_tops = sorted(set(tops))

# Seed the query from the URL (?q=...) on first load, so links are shareable.
if "q" not in st.session_state:
    st.session_state["q"] = st.query_params.get("q", "")

with st.sidebar:
    st.header("Options")
    top_k = st.slider("Number of results", 1, 30, 10)
    w_sem = st.slider(
        "Lexical ⟷ Semantic", 0.0, 1.0, 0.5, 0.05,
        help="Left = exact word match · Right = closeness in meaning.",
    )
    auto_tr = st.checkbox(
        "Translate query to Italian", value=True,
        help="Strongly improves French/English queries.",
    )
    branch = st.multiselect("Filter by department (level 1)", options=unique_tops)
    st.divider()
    st.caption(f"{len(ids)} categories · TF-IDF char(3-5) + {emb.shape[1]}-dim dense")

examples = ["Running shoes", "Phone case", "Robot vacuum", "Chitarra elettrica"]
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex, width="stretch"):
        st.session_state["q"] = ex

query = st.text_input(
    "Product to classify",
    key="q",
    placeholder="e.g. « running shoes », « macchina per caffè », « robot vacuum »",
)

# Keep the URL in sync with the current query (shareable / bookmarkable).
if query:
    if st.query_params.get("q") != query:
        st.query_params["q"] = query
elif "q" in st.query_params:
    del st.query_params["q"]

if query:
    search_q, translated = query, False
    if auto_tr:
        search_q, ok = translate_to_it(query)
        translated = ok and search_q.lower() != query.lower()

    allowed_idx = None
    if branch:
        allowed_idx = np.array([i for i, t in enumerate(tops) if t in branch], dtype=int)
        if allowed_idx.size == 0:
            st.warning("No category in the selected filter.")
            st.stop()

    order, tf, dense = hybrid_search(search_q, vec, X, emb, w_sem, top_k, allowed_idx)

    if translated:
        st.caption(f"Query translated to Italian: **{search_q}**")
    st.subheader(f"Results for « {query} »")
    render_results(order, ids, paths, dense, tf)
    if order:
        b = order[0]
        st.success(f"**Best match** — ID `{ids[b]}` · {paths[b]}")
else:
    st.info("Type a product or click an example to search.")

