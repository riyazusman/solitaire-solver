"""Visual playing-card board previews and interactive step-by-step playback."""

from html import escape

from solvers.models import Card

_BOARD_CSS = """
<style>
.ss-board { font-family: ui-sans-serif, system-ui, sans-serif; padding: 8px 4px; }
.ss-row { display: flex; justify-content: center; gap: 6px; margin-bottom: 6px; flex-wrap: nowrap; }
.ss-gap { width: 22px; display: inline-block; }
.ss-card {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 40px; height: 32px; padding: 0 5px;
  border-radius: 6px; border: 1px solid #cbd5e1;
  background: #ffffff; font-weight: 700; font-size: 13px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.08);
  white-space: nowrap;
}
.ss-card.red { color: #c62828; }
.ss-card.black { color: #1e293b; }
.ss-card.empty {
  background: transparent; border: 1px dashed #94a3b8;
  color: #94a3b8; font-weight: 500; box-shadow: none;
}
.ss-bar {
  display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;
  margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(148,163,184,0.3);
  font-size: 13px; align-items: center;
}
.ss-fc-grid {
  display: grid; grid-template-columns: repeat(8, minmax(46px, 1fr));
  gap: 6px; margin-top: 10px;
}
.ss-fc-col {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
}
.ss-fc-hdr { font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 2px; }
</style>
"""

_RANK_LABELS = ("—", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_SUIT_INFO = (("S", "♠️", "black"), ("H", "♥️", "red"), ("D", "♦️", "red"), ("C", "♣️", "black"))


def _card_pill(card: Card | None, placeholder: str = "·") -> str:
    if card is None:
        return f'<span class="ss-card empty">{escape(placeholder)}</span>'
    cls = "red" if card.is_red else "black"
    return f'<span class="ss-card {cls}">{escape(card.display)}</span>'


def render_tripeaks_board(st, board_cards: list[Card], board_mask: int, waste: Card, stock_remaining: int):
    def slot(idx: int) -> str:
        return _card_pill(board_cards[idx] if (board_mask & (1 << idx)) else None)

    gap = '<span class="ss-gap"></span>'
    row1 = gap.join(slot(i) for i in (0, 1, 2))
    row2 = f"{slot(3)}{slot(4)}{gap}{slot(5)}{slot(6)}{gap}{slot(7)}{slot(8)}"
    row3 = f"{slot(9)}{slot(10)}{slot(11)}{gap}{slot(12)}{slot(13)}{slot(14)}{gap}{slot(15)}{slot(16)}{slot(17)}"
    row4 = "".join(slot(i) for i in range(18, 28))

    html = (
        f"{_BOARD_CSS}<div class='ss-board'>"
        f"<div class='ss-row'>{row1}</div>"
        f"<div class='ss-row'>{row2}</div>"
        f"<div class='ss-row'>{row3}</div>"
        f"<div class='ss-row'>{row4}</div>"
        f"<div class='ss-bar'>"
        f"<span><strong>Waste:</strong> {_card_pill(waste)}</span>"
        f"<span><strong>Stock left:</strong> {stock_remaining}</span>"
        f"</div></div>"
    )
    st.html(html)


def render_pyramid_board(
    st,
    board_cards: list[Card],
    board_mask: int,
    drawn: Card | None,
    waste_top: Card | None,
    waste_count: int,
    stock_remaining: int,
    passes: int = 1,
):
    rows_html = []
    idx = 0
    for row_len in range(1, 8):
        pills = "".join(
            _card_pill(board_cards[idx + offset] if (board_mask & (1 << (idx + offset))) else None)
            for offset in range(row_len)
        )
        rows_html.append(f"<div class='ss-row'>{pills}</div>")
        idx += row_len

    html = (
        f"{_BOARD_CSS}<div class='ss-board'>"
        f"{''.join(rows_html)}"
        f"<div class='ss-bar'>"
        f"<span><strong>Stock left:</strong> {stock_remaining} (Pass {passes})</span>"
        f"<span><strong>Drawn:</strong> {_card_pill(drawn, '—')}</span>"
        f"<span><strong>Waste top:</strong> {_card_pill(waste_top, '—')} ({waste_count})</span>"
        f"</div></div>"
    )
    st.html(html)


def render_freecell_board(st, cascades, freecells, foundations: dict[str, int]):
    fc_pills = "".join(
        _card_pill(freecells[i] if i < len(freecells) else None, f"F{i + 1}")
        for i in range(4)
    )
    found_pills = []
    for suit, sym, cls in _SUIT_INFO:
        rank = foundations[suit]
        if rank == 0:
            found_pills.append(f'<span class="ss-card empty">{sym}</span>')
        else:
            found_pills.append(f'<span class="ss-card {cls}">{_RANK_LABELS[rank]}{sym}</span>')

    cols_html = []
    for c_idx in range(8):
        col = cascades[c_idx]
        cards_html = "".join(_card_pill(c) for c in col) if col else _card_pill(None, "—")
        cols_html.append(f"<div class='ss-fc-col'><div class='ss-fc-hdr'>C{c_idx + 1}</div>{cards_html}</div>")

    html = (
        f"{_BOARD_CSS}<div class='ss-board'>"
        f"<div class='ss-bar' style='margin-top:0; padding-top:0; border-top:none;'>"
        f"<span><strong>Freecells:</strong> {fc_pills}</span>"
        f"<span><strong>Foundations:</strong> {''.join(found_pills)}</span>"
        f"</div>"
        f"<div class='ss-fc-grid'>{''.join(cols_html)}</div>"
        f"</div>"
    )
    st.html(html)


def render_step_playback(st, *, key_prefix: str, snapshots, render_snapshot_fn):
    """Interactive Prev/Next + slider step viewer for a replayed solution."""
    if not snapshots:
        return

    max_step = len(snapshots) - 1
    step_key = f"{key_prefix}_playback_step"
    if step_key not in st.session_state or st.session_state[step_key] > max_step:
        st.session_state[step_key] = 0

    def _prev():
        st.session_state[step_key] = max(0, st.session_state[step_key] - 1)

    def _next():
        st.session_state[step_key] = min(max_step, st.session_state[step_key] + 1)

    prev_col, slider_col, next_col = st.columns([1, 6, 1])
    with prev_col:
        st.button("◀ Prev", key=f"{key_prefix}_pb_prev", on_click=_prev, disabled=st.session_state[step_key] <= 0)
    with slider_col:
        if max_step > 0:
            st.slider(
                "Step",
                min_value=0,
                max_value=max_step,
                key=step_key,
                label_visibility="collapsed",
            )
    with next_col:
        st.button("Next ▶", key=f"{key_prefix}_pb_next", on_click=_next, disabled=st.session_state[step_key] >= max_step)

    snap = snapshots[st.session_state[step_key]]
    st.caption(f"**Step {snap.step_num} / {max_step}:** {snap.move_text}")
    render_snapshot_fn(snap)
