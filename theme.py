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
    "bg": "#ffffff",
    "surface": "#ffffff",
    "surface_alt": "#f5f7fa",
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "text": "#111827",
    "text_muted": "#5b6472",
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "accent": "#0ea5e9",
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
    "raw_z": "#8c96a3",
    "z_smooth": "#0072B2",
    "range_average": "#E69F00",
    "lick": "#7C3AED",
    "lick_rate": "#0891b2",
    "bout": "#f59e0b",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
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
  background-color: #ffffff !important;
  color: #111827 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
  background-color: #ffffff !important;
  color: #111827 !important;
}
[data-testid="stDecoration"] { background-image: none !important; }

[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] > div {
  background-color: #ffffff !important;
  border-right: 1px solid #e2e8f0;
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
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px; }
section[data-testid="stSidebar"] { min-width: 340px; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* =========================================================================
   4. HEADER, CARDS, NOTES
   ========================================================================= */
.pp-header {
  display: flex; align-items: center; gap: 0.9rem;
  padding: 0.85rem 1.15rem; margin-bottom: 1.15rem;
  background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
  border: 1px solid #e2e8f0; border-radius: 10px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.pp-header-mark {
  width: 38px; height: 38px; flex: 0 0 38px; border-radius: 9px;
  background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
  display: flex; align-items: center; justify-content: center;
  color: #ffffff !important; font-weight: 700; font-size: 1rem;
}
.pp-header-title { font-size: 1.12rem; font-weight: 650; line-height: 1.2;
                   margin: 0; color: #111827 !important; }
.pp-header-sub { font-size: 0.82rem; color: #5b6472 !important; margin-top: 0.15rem; }

.pp-side-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; color: #5b6472 !important;
  margin: 1.1rem 0 0.35rem 0; padding-bottom: 0.3rem;
  border-bottom: 1px solid #e2e8f0;
}

.pp-card { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
           padding:0.9rem 1.05rem; margin-bottom:0.9rem; }
.pp-note {
  border-left: 3px solid #2563eb; background: #eff6ff;
  color: #1e3a8a !important; padding: 0.55rem 0.8rem;
  border-radius: 0 8px 8px 0; font-size: 0.85rem; margin: 0.5rem 0 0.9rem 0;
}
.pp-note.warn { border-left-color:#d97706; background:#fffbeb; color:#78350f !important; }
.pp-note.ok   { border-left-color:#059669; background:#ecfdf5; color:#064e3b !important; }
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
  background-color: #ffffff !important; color: #111827 !important;
  border: 1px solid #cbd5e1 !important; border-radius: 8px;
  font-weight: 550; font-size: 0.875rem; transition: all 0.13s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #2563eb !important; color: #2563eb !important;
  background-color: #f8fbff !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background-color: #2563eb !important; border-color: #2563eb !important;
  color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: #1d4ed8 !important; color: #ffffff !important;
}

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
  border: 1px solid #e2e8f0 !important; border-radius: 10px;
  overflow: hidden; background-color: #ffffff !important;
}
div[data-testid="stExpander"] summary {
  background-color: #f9fafb !important; font-weight: 550; font-size: 0.9rem;
}
div[data-testid="stExpander"] summary:hover { background-color: #f1f5f9 !important; }
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background-color: #ffffff !important;
}

div[data-testid="stMetric"] {
  background-color: #ffffff !important; border: 1px solid #e2e8f0;
  border-radius: 9px; padding: 0.65rem 0.85rem;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.04em; color: #5b6472 !important;
}
[data-testid="stMetricValue"] { color: #111827 !important; }

div[data-testid="stDataFrame"], .stDataFrame {
  border: 1px solid #e2e8f0; border-radius: 8px; background-color: #ffffff !important;
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


def page_header(title, subtitle="", mark="FP"):
    st.markdown(
        f"""
        <div class="pp-header">
          <div class="pp-header-mark">{mark}</div>
          <div>
            <div class="pp-header-title">{title}</div>
            <div class="pp-header-sub">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_label(text):
    st.sidebar.markdown(f'<div class="pp-side-label">{text}</div>',
                        unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="pp-note {kind}">{text}</div>', unsafe_allow_html=True)


def style_figure(fig, height=None, showlegend=True):
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS)
    if height:
        fig.update_layout(height=height)
    fig.update_layout(showlegend=showlegend)
    return fig
