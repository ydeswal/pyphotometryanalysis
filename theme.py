#!/usr/bin/env python3
"""
theme.py
========

One stylesheet, applied once.

The app previously injected three separate <style> blocks that fought each other
with blanket `!important` rules. Two were especially destructive:

    section[data-testid="stSidebar"] *:not(svg):not(path) { color: #111827 !important; }
    div[data-baseweb="select"] div { background-color: #ffffff !important; }

The first repainted every descendant in the sidebar, including text inside
widget chips that is meant to be white on a coloured background. The second
matched *every* nested div inside a select, not just the outer control, so
multiselect tag chips, their remove buttons and the clear button all lost their
styling. That is why the "All recordings" and "All graphs" chips in the sidebar
rendered as red-on-black boxes with detached white x marks.

The rules below are scoped to specific elements, set the multiselect chips
explicitly rather than by accident, and never use a universal selector. Base
colours come from .streamlit/config.toml so Streamlit styles its own widgets.
"""

import streamlit as st

PALETTE = {
    "bg": "#ffffff",
    "surface": "#ffffff",
    "surface_alt": "#f4f6fa",
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "text": "#111827",
    "text_muted": "#64748b",
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
    "raw_z": "#94a3b8",
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
    paper_bgcolor="rgba(0,0,0,0)",
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
)

_CSS = """
<style>
:root {
  --pp-border: #e2e8f0;
  --pp-border-strong: #cbd5e1;
  --pp-text: #111827;
  --pp-muted: #64748b;
  --pp-primary: #2563eb;
  --pp-surface-alt: #f4f6fa;
  --pp-radius: 10px;
}

/* ---------- layout ---------- */
.block-container {
  padding-top: 1.1rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}
section[data-testid="stSidebar"] { min-width: 340px; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ---------- page header ---------- */
.pp-header {
  display: flex; align-items: center; gap: 0.9rem;
  padding: 0.85rem 1.15rem; margin-bottom: 1.15rem;
  background: linear-gradient(135deg, #ffffff 0%, #f4f8ff 100%);
  border: 1px solid var(--pp-border);
  border-radius: var(--pp-radius);
  box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.pp-header-mark {
  width: 38px; height: 38px; flex: 0 0 38px; border-radius: 9px;
  background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 700; font-size: 1rem; letter-spacing: -0.02em;
}
.pp-header-title { font-size: 1.12rem; font-weight: 650; line-height: 1.2; margin: 0; }
.pp-header-sub { font-size: 0.82rem; color: var(--pp-muted); margin-top: 0.15rem; }

/* ---------- sidebar section labels ---------- */
.pp-side-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--pp-muted);
  margin: 1.1rem 0 0.35rem 0; padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--pp-border);
}

/* ---------- cards and notes ---------- */
.pp-card {
  background: #fff; border: 1px solid var(--pp-border);
  border-radius: var(--pp-radius); padding: 0.9rem 1.05rem; margin-bottom: 0.9rem;
}
.pp-note {
  border-left: 3px solid var(--pp-primary);
  background: #eff6ff; color: #1e3a8a;
  padding: 0.55rem 0.8rem; border-radius: 0 8px 8px 0;
  font-size: 0.85rem; margin: 0.5rem 0 0.9rem 0;
}
.pp-note.warn { border-left-color: #d97706; background: #fffbeb; color: #78350f; }
.pp-note.ok   { border-left-color: #059669; background: #ecfdf5; color: #064e3b; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem; border-bottom: 1px solid var(--pp-border); padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  height: 2.4rem; padding: 0 0.95rem; font-size: 0.875rem; font-weight: 500;
  color: var(--pp-muted); border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--pp-surface-alt); color: var(--pp-text); }
.stTabs [aria-selected="true"] { color: var(--pp-primary); font-weight: 600; }

/* ---------- multiselect chips ----------
   Styled explicitly. Previously a blanket rule on `div[data-baseweb="select"] div`
   overrode these and produced unreadable red-on-black chips. */
div[data-baseweb="select"] span[data-baseweb="tag"] {
  background-color: #e0edff;
  color: #1e40af;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 500;
}
div[data-baseweb="select"] span[data-baseweb="tag"] svg { fill: #1e40af; }
div[data-baseweb="select"] span[data-baseweb="tag"] [role="presentation"]:hover {
  background-color: #bfdbfe; border-radius: 4px;
}

/* ---------- inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
  border-color: var(--pp-border-strong); border-radius: 8px;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--pp-primary); box-shadow: 0 0 0 2px rgba(37,99,235,0.12);
}

/* ---------- expanders, metrics, tables ---------- */
div[data-testid="stExpander"] {
  border: 1px solid var(--pp-border); border-radius: var(--pp-radius);
  overflow: hidden; background: #fff;
}
div[data-testid="stExpander"] summary { font-weight: 550; font-size: 0.9rem; }
div[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--pp-border);
  border-radius: 9px; padding: 0.65rem 0.85rem;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--pp-muted);
}

/* ---------- file uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--pp-surface-alt);
  border: 1.5px dashed var(--pp-border-strong); border-radius: var(--pp-radius);
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--pp-primary); background: #f0f6ff;
}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px; font-weight: 550; font-size: 0.875rem;
  border: 1px solid var(--pp-border-strong); transition: all 0.13s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--pp-primary); color: var(--pp-primary);
}
.stButton > button[kind="primary"] {
  background: var(--pp-primary); border-color: var(--pp-primary); color: #fff;
}
.stButton > button[kind="primary"]:hover { background: #1d4ed8; color: #fff; }

/* ---------- misc ---------- */
hr { margin: 1.1rem 0; border-color: var(--pp-border); }
div[data-testid="stDataFrame"] { border: 1px solid var(--pp-border); border-radius: 8px; }
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
