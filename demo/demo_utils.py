from IPython.display import HTML, display

from lettucedetect_api.models import TokenDetectionItem


def display_output(
    predictions: list[TokenDetectionItem],
    factor: float = 0.8,
    highlight_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Display LLM answer highlighted with confidences.

    :param predictions: A list of `TokenDetectionItem`s from the LettuceDetect
        client library.
    :param factor: Between 0 and 1. Controls the intensity of the highlights.
        Optional, default is 0.8.
    :param highlight_color: Base color used for the text highlights. A tuple
        with three integers between 0 and 255 (RGB). Optional, default is (255, 0, 0).
    """
    text = [item.token for item in predictions]
    r, g, b = highlight_color
    colors = [f"rgba({r}, {g}, {b}, {item.hallucination_score * factor})" for item in predictions]
    html_elements = [
        f'<span style="background-color: {color};">{text}</span>'
        for color, text in zip(colors, text)
    ]
    html = "".join(html_elements)
    display(HTML(html))
