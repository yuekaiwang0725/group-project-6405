"""Generate sentiment-coloured word clouds.

Positive-leaning words appear in green tones, negative-leaning words in
red tones.  The cloud is returned as a PIL Image so it can be displayed
directly by Streamlit.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def _sentiment_color_func(
    word: str,
    font_size: int,
    position: tuple[int, int],
    orientation: Any,
    random_state: Any = None,
    **kwargs: Any,
) -> str:
    """Colour mapping used by the WordCloud library callback."""
    # The colour is determined _after_ generation via recolor(),
    # so we use a simple hash-based green/blue palette here.
    hue = 140 + (hash(word) % 60)  # green-teal range
    return f"hsl({hue}, 70%, 50%)"


def generate_sentiment_wordcloud(
    positive_texts: list[str],
    negative_texts: list[str],
    width: int = 800,
    height: int = 400,
) -> Image.Image:
    """Build a word cloud with green (positive) and red (negative) words.

    Returns a PIL Image.
    """
    try:
        from wordcloud import WordCloud  # type: ignore
    except ImportError:
        # Return a blank placeholder if wordcloud is not installed.
        return Image.new("RGB", (width, height), (30, 30, 46))

    pos_text = " ".join(positive_texts)
    neg_text = " ".join(negative_texts)

    # Build frequency dicts and tag polarity
    pos_wc = WordCloud(width=width, height=height).process_text(pos_text)
    neg_wc = WordCloud(width=width, height=height).process_text(neg_text)

    merged: dict[str, float] = {}
    word_polarity: dict[str, str] = {}
    for w, freq in pos_wc.items():
        merged[w] = merged.get(w, 0) + freq
        word_polarity[w] = "positive"
    for w, freq in neg_wc.items():
        merged[w] = merged.get(w, 0) + freq
        if w not in word_polarity:
            word_polarity[w] = "negative"
        elif neg_wc.get(w, 0) > pos_wc.get(w, 0):
            word_polarity[w] = "negative"

    if not merged:
        return Image.new("RGB", (width, height), (30, 30, 46))

    def _color_func(
        word: str, **kwargs: Any,
    ) -> str:
        if word_polarity.get(word) == "positive":
            hue = 145 + (hash(word) % 30)
            return f"hsl({hue}, 75%, 55%)"
        else:
            hue = 0 + (hash(word) % 20)
            return f"hsl({hue}, 75%, 50%)"

    wc = WordCloud(
        width=width,
        height=height,
        background_color="#1e1e2e",
        max_words=120,
        prefer_horizontal=0.7,
        colormap="Set2",
        contour_width=0,
    ).generate_from_frequencies(merged)
    wc.recolor(color_func=_color_func)
    return wc.to_image()


def generate_emotion_wordcloud(
    texts_by_emotion: dict[str, list[str]],
    width: int = 800,
    height: int = 400,
) -> Image.Image:
    """Build a word cloud coloured by dominant emotion category."""
    try:
        from wordcloud import WordCloud  # type: ignore
    except ImportError:
        return Image.new("RGB", (width, height), (30, 30, 46))

    emotion_hues = {
        "sadness": 210,
        "joy": 50,
        "love": 340,
        "anger": 0,
        "fear": 270,
        "surprise": 30,
    }

    merged: dict[str, float] = {}
    word_emotion: dict[str, str] = {}
    for emotion, texts in texts_by_emotion.items():
        freq = WordCloud(width=width, height=height).process_text(" ".join(texts))
        for w, f in freq.items():
            if w not in merged or f > merged[w]:
                word_emotion[w] = emotion
            merged[w] = merged.get(w, 0) + f

    if not merged:
        return Image.new("RGB", (width, height), (30, 30, 46))

    def _color_func(word: str, **kwargs: Any) -> str:
        emotion = word_emotion.get(word, "joy")
        hue = emotion_hues.get(emotion, 50)
        return f"hsl({hue}, 72%, 55%)"

    wc = WordCloud(
        width=width,
        height=height,
        background_color="#1e1e2e",
        max_words=120,
        prefer_horizontal=0.7,
    ).generate_from_frequencies(merged)
    wc.recolor(color_func=_color_func)
    return wc.to_image()
