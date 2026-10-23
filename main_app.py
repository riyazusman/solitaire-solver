import streamlit as st

st.set_page_config(
    page_title="Solitaire Solver Studio", 
    page_icon="🃏", 
    layout="centered"
)

tripeaks_page = st.Page("tripeaks_solver.py", title="Tripeaks Solver", icon="🏔️")
pyramid_page = st.Page("pyramid_solver.py", title="Pyramid Solver", icon="🛕")
freecell_page = st.Page("freecell_solver.py", title="Freecell Solver", icon="🃏")

pg = st.navigation([tripeaks_page, pyramid_page, freecell_page])
pg.run()