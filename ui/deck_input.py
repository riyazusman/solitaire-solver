import random

from solvers.models import DEFAULT_DECK_STR, STANDARD_DECK, DeckAnalysis, analyze_deck


def _random_deck_str() -> str:
    cards = sorted(STANDARD_DECK)
    random.shuffle(cards)
    return ", ".join(cards)


def _sample_deck_str(seed: int) -> str:
    rng = random.Random(seed)
    cards = sorted(STANDARD_DECK)
    rng.shuffle(cards)
    return ", ".join(cards)


def render_deck_input(
    st,
    *,
    label: str = "Enter your 52 cards:",
    key: str,
    default_value: str = DEFAULT_DECK_STR,
    sample_seed: int = 1,
    height: int = 150,
    disabled: bool = False,
) -> tuple[str, DeckAnalysis]:
    """Render quick-fill deck buttons, textarea, and live validation status warnings."""
    if key not in st.session_state:
        st.session_state[key] = default_value

    def _set_random():
        st.session_state[key] = _random_deck_str()

    def _set_sample():
        st.session_state[key] = _sample_deck_str(sample_seed)

    def _set_ordered():
        st.session_state[key] = DEFAULT_DECK_STR

    def _clear():
        st.session_state[key] = ""

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.button("🎲 Random Deal", key=f"{key}_rand", on_click=_set_random, disabled=disabled, use_container_width=True)
    with b2:
        st.button("📋 Sample Deal", key=f"{key}_sample", on_click=_set_sample, disabled=disabled, use_container_width=True)
    with b3:
        st.button("🃏 Ordered Deck", key=f"{key}_ord", on_click=_set_ordered, disabled=disabled, use_container_width=True)
    with b4:
        st.button("🗑️ Clear", key=f"{key}_clear", on_click=_clear, disabled=disabled, use_container_width=True)

    card_input = st.text_area(label, height=height, key=key, disabled=disabled)
    analysis = analyze_deck(card_input)

    if analysis.is_valid:
        st.success("✅ **Deck Status:** 52 valid cards ready to go!")
    else:
        st.info(
            f"📊 **Deck Status:** {analysis.total_entries}/52 entries | "
            f"{analysis.valid_unique} valid unique cards"
        )
        for status_label, items in (
            ("Invalid entries", analysis.invalids),
            ("Duplicate cards", analysis.duplicate_labels),
            ("Missing cards", analysis.missing),
        ):
            if items:
                st.warning(f"**{status_label} ({len(items)}):** {', '.join(items)}")

    return card_input, analysis
