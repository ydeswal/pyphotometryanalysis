#!/usr/bin/env python3
"""
theme.py
========

One stylesheet, applied once, that does not depend on anything else loading.

Two lessons are baked into this file.

1. The original app injected three competing <style> blocks full of blanket
   !important rules. Two were destructive:

       section[data-testid="stSidebar"] *:not(svg):not(path) { color: ... !important; }
       div[data-baseweb="select"] div { background-color: #fff !important; }

   The first repainted every descendant in the sidebar, including text meant to
   be light-on-coloured inside widget chips. The second matched every nested div
   in a select rather than just the outer control, so multiselect tag chips,
   their remove buttons and the clear button all lost their styling. That is
   what produced the unreadable red-on-black chips.

2. The fix for (1) was to move base colours into .streamlit/config.toml and keep
   only scoped CSS here. That is cleaner, but it fails badly if config.toml does
   not deploy: you get Streamlit's dark base with light cards painted on top,
   and dark text sitting on dark backgrounds.

So this file now forces light mode itself, and config.toml is a bonus rather
than a requirement. The rules are still scoped: text colours are applied to
elements that actually carry text, never with a universal selector, and the
widget chips are styled explicitly at the end so nothing can repaint them by
accident.
"""

import streamlit as st

PALETTE = {
    "bg": "#f6f8fb",            # page
    "surface": "#ffffff",       # cards, graphs, sidebar
    "surface_alt": "#f5f7fa",
    "border": "#e5e9f0",
    "border_strong": "#cbd5e1",
    "text": "#0f172a",
    "text_muted": "#64748b",
    "primary": "#2563eb",       # the one accent colour
    "primary_dark": "#1d4ed8",
    "accent": "#7c3aed",        # used only with primary, for the AI panel
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
}

# Colour-blind-safe trace palette (Okabe-Ito derived).
TRACE_COLORS = {
    "sig_raw": "#0072B2",
    "uv_raw": "#009E73",
    "uv_fit": "#CC79A7",
    "deltaF": "#E69F00",
    "dff": "#D55E00",
    "raw_z": "#6b7280",
    "z_smooth": "#0072B2",
    "range_average": "#E69F00",
    "lick": "#7C3AED",
    "lick_rate": "#0891b2",
    "bout": "#f59e0b",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, 'Segoe UI', Roboto, sans-serif",
              size=12, color=PALETTE["text"]),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=62, r=24, t=44, b=48),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=PALETTE["border_strong"],
                    font=dict(size=12, color=PALETTE["text"])),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, bgcolor="rgba(255,255,255,0.85)",
                bordercolor=PALETTE["border"], borderwidth=1),
)

PLOTLY_AXIS = dict(
    showgrid=True, gridcolor="#eef2f7", gridwidth=1,
    zeroline=False, linecolor=PALETTE["border_strong"], linewidth=1,
    ticks="outside", tickcolor=PALETTE["border_strong"], ticklen=4,
    title_font=dict(color=PALETTE["text"]),
    tickfont=dict(color=PALETTE["text"]),
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* =========================================================================
   0. TYPE
   Set on text-carrying elements only; Material icon spans set their own font.
   ========================================================================= */
.stApp, p, li, label, td, th, h1, h2, h3, h4, h5, h6, button, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3, [data-testid="stHeadingWithActionElements"] { letter-spacing: -0.015em; }
h3 { font-weight: 650 !important; }
/* =========================================================================
   1. FORCE LIGHT MODE
   Containers only. No universal selector, so widget internals keep their
   own styling.
   ========================================================================= */
:root, .stApp { color-scheme: light; }

html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main,
.block-container {
  background-color: #f6f8fb !important;
  color: #0f172a !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
  background-color: rgba(246,248,251,0.92) !important;
  color: #0f172a !important;
}
[data-testid="stDecoration"] { background-image: none !important; }

[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] > div {
  background-color: #ffffff !important;
  border-right: 1px solid #e5e9f0;
}

/* =========================================================================
   2. TEXT
   Elements that carry text. Widget chips are re-asserted in section 5 and
   are deliberately not covered here.
   ========================================================================= */
h1, h2, h3, h4, h5, h6,
p, li, td, th, dt, dd,
.stMarkdown,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stHeadingWithActionElements"],
.stCheckbox label span,
.stRadio label span,
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
  color: #111827 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
small {
  color: #5b6472 !important;
}

code, kbd {
  background-color: #f1f5f9 !important;
  color: #0f172a !important;
  border-radius: 4px;
  padding: 0.1em 0.35em;
}

/* =========================================================================
   3. LAYOUT
   ========================================================================= */
.block-container { padding-top: 3.4rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stHeader"] { height: 2.75rem; }
section[data-testid="stSidebar"] { min-width: 340px; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* =========================================================================
   4. HEADER, CARDS, NOTES
   ========================================================================= */
.pp-header {
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
  padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  background: #ffffff;
  border: 1px solid #e5e9f0; border-radius: 14px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px -8px rgba(15,23,42,0.08);
  position: relative; overflow: hidden;
}
.pp-header::before {            /* thin accent rule along the top */
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px;
  background: linear-gradient(90deg, #2563eb 0%, #3b82f6 60%, #93c5fd 100%);
}
.pp-header-mark {
  width: 44px; height: 44px; flex: 0 0 44px; border-radius: 11px;
  background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
  display: flex; align-items: center; justify-content: center;
  color: #ffffff !important; font-weight: 700; font-size: 1.02rem; letter-spacing: 0.02em;
  box-shadow: 0 4px 12px -4px rgba(37,99,235,0.55);
}
.pp-header-text { flex: 1 1 260px; min-width: 0; }
.pp-header-title { font-size: 1.28rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em;
                   margin: 0; color: #0f172a !important; }
.pp-header-sub { font-size: 0.84rem; color: #64748b !important; margin-top: 0.2rem; }
.pp-header-tags { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.pp-tag {
  font-size: 0.72rem; font-weight: 600; color: #334155 !important;
  background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 999px;
  padding: 0.18rem 0.6rem; white-space: nowrap;
}
.pp-tag.ai { color: #4338ca !important; background: #eef2ff; border-color: #c7d2fe; }

.pp-side-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: #64748b !important;
  margin: 1.25rem 0 0.45rem 0; padding: 0 0 0.35rem 0.55rem;
  border-bottom: 1px solid #eef1f5; border-left: 3px solid #2563eb;
}

.pp-card { background:#fff; border:1px solid #e5e9f0; border-radius:12px;
           padding:0.9rem 1.05rem; margin-bottom:0.9rem; }

/* credit line, on every page */
.pp-footer {
  margin: 2.5rem 0 0.5rem 0; padding: 1rem 0 0.25rem 0;
  border-top: 1px solid #e5e9f0; text-align: center;
  font-size: 0.82rem; color: #64748b !important;
}
.pp-footer b { color: #0f172a !important; font-weight: 600; }
.pp-footer .heart { color: #e11d48 !important; }
.pp-side-credit {
  margin-top: 1.6rem; padding-top: 0.8rem; border-top: 1px solid #eef1f5;
  font-size: 0.78rem; color: #64748b !important; text-align: center;
}
.pp-side-credit b { color: #0f172a !important; font-weight: 600; }
.pp-side-credit .heart { color: #e11d48 !important; }
.pp-note {
  border: 1px solid #dbe6fb; border-left: 3px solid #2563eb; background: #f5f9ff;
  color: #1e3a8a !important; padding: 0.6rem 0.85rem;
  border-radius: 0 10px 10px 0; font-size: 0.85rem; line-height: 1.55; margin: 0.5rem 0 0.9rem 0;
}
.pp-note.warn { border-color:#fde7c2; border-left-color:#d97706; background:#fffbf2; color:#78350f !important; }
.pp-note.ok   { border-color:#c9eedd; border-left-color:#059669; background:#f2fbf7; color:#064e3b !important; }
.pp-note b, .pp-note code { color: inherit !important; background: transparent !important; }

/* =========================================================================
   5. FORM CONTROLS
   Inputs are painted white first, then chips get their own colours so they
   survive. Order and specificity both matter here.
   ========================================================================= */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="base-input"],
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
  background-color: #ffffff !important;
  color: #111827 !important;
  border-color: #cbd5e1 !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: #94a3b8 !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 2px rgba(37,99,235,0.12);
}

/* dropdown menus */
div[data-baseweb="popover"] div[data-baseweb="menu"],
ul[role="listbox"] {
  background-color: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
}
li[role="option"] { background-color: #ffffff !important; color: #111827 !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background-color: #eff6ff !important; color: #1e40af !important;
}

/* multiselect chips - explicit, and last so nothing above repaints them */
div[data-baseweb="select"] span[data-baseweb="tag"] {
  background-color: #e0edff !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 6px !important;
  font-size: 0.8rem; font-weight: 500;
}
div[data-baseweb="select"] span[data-baseweb="tag"],
div[data-baseweb="select"] span[data-baseweb="tag"] span,
div[data-baseweb="select"] span[data-baseweb="tag"] div {
  color: #1e40af !important;
}
div[data-baseweb="select"] span[data-baseweb="tag"] svg { fill: #1e40af !important; }
div[data-baseweb="select"] span[data-baseweb="tag"] [role="presentation"]:hover {
  background-color: #bfdbfe !important; border-radius: 4px;
}

/* checkboxes */
[data-testid="stCheckbox"] label > span:first-child {
  background-color: #ffffff !important; border-color: #94a3b8 !important;
}
[data-testid="stCheckbox"] label > span[aria-checked="true"] {
  background-color: #2563eb !important; border-color: #2563eb !important;
}

/* sliders */
.stSlider [data-baseweb="slider"] [role="slider"] { background-color: #2563eb !important; }
.stSlider [data-testid="stTickBar"] { background: #e2e8f0 !important; }
.stSlider [data-testid="stThumbValue"] { color: #111827 !important; }

/* buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  background-color: #ffffff !important; color: #0f172a !important;
  border: 1px solid #d5dbe4 !important; border-radius: 9px;
  font-weight: 550; font-size: 0.875rem; transition: all 0.13s ease;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #2563eb !important; color: #2563eb !important;
  background-color: #f8fbff !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(180deg, #3170f0 0%, #2563eb 100%) !important;
  border-color: #1d4ed8 !important; color: #ffffff !important; font-weight: 600;
  box-shadow: 0 1px 2px rgba(15,23,42,0.08), 0 4px 12px -6px rgba(37,99,235,0.6);
}
.stButton > button[kind="primary"]:hover {
  background: #1d4ed8 !important; color: #ffffff !important;
}
.stButton > button[kind="primary"] p { color: #ffffff !important; }

/* file uploader */
[data-testid="stFileUploaderDropzone"] {
  background-color: #f5f7fa !important;
  border: 1.5px dashed #cbd5e1 !important; border-radius: 10px;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #2563eb !important; background-color: #f0f6ff !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {
  color: #475569 !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background-color: #ffffff !important; color: #111827 !important;
  border: 1px solid #cbd5e1 !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] div { color: #111827 !important; }

/* =========================================================================
   6. CONTAINERS
   ========================================================================= */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem; border-bottom: 1px solid #e2e8f0;
  background-color: #ffffff !important;
}
.stTabs [data-baseweb="tab"] {
  height: 2.4rem; padding: 0 0.95rem; font-size: 0.875rem; font-weight: 500;
  color: #5b6472 !important; background-color: transparent !important;
  border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"]:hover {
  background-color: #f5f7fa !important; color: #111827 !important;
}
.stTabs [aria-selected="true"] { color: #2563eb !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2563eb !important; }

div[data-testid="stExpander"] {
  border: 1px solid #e5e9f0 !important; border-radius: 12px;
  overflow: hidden; background-color: #ffffff !important;
  box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
div[data-testid="stExpander"] details { border: none !important; }
div[data-testid="stExpander"] summary {
  background-color: #ffffff !important; font-weight: 600; font-size: 0.9rem;
}
div[data-testid="stExpander"] summary:hover { background-color: #f8fafc !important; }
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background-color: #ffffff !important;
}

div[data-testid="stMetric"] {
  background-color: #ffffff !important; border: 1px solid #e5e9f0;
  border-radius: 12px; padding: 0.7rem 0.95rem; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.04em; color: #5b6472 !important;
}
[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.55rem !important;
                                font-weight: 600; letter-spacing: -0.02em; }
h1 code, h2 code, h3 code, [data-testid="stHeadingWithActionElements"] code {
  font-size: 0.72em !important; font-weight: 500; vertical-align: 0.12em;
  background: #eef2f7 !important; color: #1e293b !important; padding: 0.15em 0.45em;
  border-radius: 6px;
}

div[data-testid="stDataFrame"], .stDataFrame {
  border: 1px solid #e5e9f0; border-radius: 10px; background-color: #ffffff !important;
}

/* graphs sit on white cards */
[data-testid="stPlotlyChart"] {
  background: #ffffff; border: 1px solid #e5e9f0; border-radius: 12px;
  padding: 6px 6px 2px 6px; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}

/* horizontal radio groups become a segmented control */
[data-testid="stRadioGroup"][aria-orientation="horizontal"] {
  display: inline-flex !important; flex-wrap: wrap; gap: 4px; max-width: 100%;
  background: #eef1f6; border: 1px solid #e2e7ef; border-radius: 11px; padding: 4px;
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] > div { margin: 0 !important; }
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] {
  padding: 0.32rem 0.85rem !important; border-radius: 8px; cursor: pointer;
  margin: 0 !important; transition: background 0.12s ease;
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] > div > div:first-child {
  display: none !important;        /* the radio circle */
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] p {
  font-size: 0.86rem; font-weight: 500; color: #475569 !important; margin: 0;
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"]:hover {
  background: rgba(255,255,255,0.65);
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] {
  background: #ffffff; box-shadow: 0 1px 3px rgba(15,23,42,0.10), 0 0 0 1px #dbe3ef;
}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] p {
  color: #1d4ed8 !important; font-weight: 600;
}

/* =========================================================================
   7. THE AI PANEL - the one place with a second colour
   ========================================================================= */
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] {
  border: 1px solid #c7d2fe !important;
  background: #ffffff !important;
  box-shadow: none !important;
}
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary {
  background: #f5f7ff !important;
}
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary:hover {
  background: #eef1ff !important;
}
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary,
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary p,
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary span {
  color: #3730a3 !important; font-weight: 600;
}
.st-key-ai-panel [data-testid="stChatInput"] > div {
  border-color: #c7d2fe !important;
}
.st-key-ai-panel > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > div {
  background: #f1f5f9 !important;          /* chat area: soft slate, set apart from the white page cards */
}
.st-key-ai-panel [data-testid="stChatMessage"] {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
}
.st-key-ai-panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
.st-key-ai-panel [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: #eef2ff; border-color: #c7d2fe;
}

[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stNotification"] { background-color: #ffffff !important; }

.js-plotly-plot .plotly, .stPlotlyChart { background-color: #ffffff !important; }

hr { margin: 1.1rem 0; border-color: #e2e8f0; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


def inject_theme():
    """Apply the stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title, subtitle="", mark="FP", tags=()):
    tag_html = "".join(f'<span class="pp-tag{" ai" if t.lower().startswith("ai") else ""}">{t}</span>'
                       for t in tags)
    st.markdown(
        f"""
        <div class="pp-header">
          <div class="pp-header-mark">{mark}</div>
          <div class="pp-header-text">
            <div class="pp-header-title">{title}</div>
            <div class="pp-header-sub">{subtitle}</div>
          </div>
          <div class="pp-header-tags">{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


CREDIT = 'Made with <span class="heart">❤️</span> by <b>Yash D</b>'


def footer():
    """Credit line at the bottom of the page and of the sidebar."""
    st.markdown(f'<div class="pp-footer">{CREDIT}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="pp-side-credit">{CREDIT}</div>', unsafe_allow_html=True)


def sidebar_label(text):
    st.sidebar.markdown(f'<div class="pp-side-label">{text}</div>',
                        unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="pp-note {kind}">{text}</div>', unsafe_allow_html=True)


def explain(label, text, container=None, use_container_width=False):
    """A small button that opens an explanation only when someone asks for it."""
    where = container if container is not None else st
    with where.popover(label, use_container_width=use_container_width):
        st.markdown(text, unsafe_allow_html=True)


def style_figure(fig, height=None, showlegend=True):
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    if height:
        fig.update_layout(height=height)
    fig.update_layout(showlegend=showlegend)
    return fig
