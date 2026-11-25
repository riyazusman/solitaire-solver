import streamlit as st

st.set_page_config(
    page_title="Solitaire Solver Studio",
    page_icon="🃏",
    layout="centered",
)

tripeaks_page = st.Page(
    "ui/pages/tripeaks_page.py",
    title="Tripeaks Solver",
    icon="🏔️",
    url_path="tripeaks",
)
pyramid_page = st.Page(
    "ui/pages/pyramid_page.py",
    title="Pyramid Solver",
    icon="🛕",
    url_path="pyramid",
)
freecell_page = st.Page(
    "ui/pages/freecell_page.py",
    title="Freecell Solver",
    icon="🃏",
    url_path="freecell",
)


def _default_redirect():
    st.switch_page(tripeaks_page)


# Hidden default route so all three solver pages keep explicit direct URL paths
# (/tripeaks, /pyramid, /freecell) instead of the first page collapsing to "/".
default_page = st.Page(_default_redirect, title="Home", default=True, visibility="hidden")

pg = st.navigation([default_page, tripeaks_page, pyramid_page, freecell_page])
pg.run()
