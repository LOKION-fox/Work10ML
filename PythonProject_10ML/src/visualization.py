import os

import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

from src.analytics import get_daily_reviews, get_rating_distribution

def create_rating_distribution_chart(df):
    """
    Создаёт график распределения оценок.
    """

    rating_distribution = get_rating_distribution(df)

    fig = px.bar(
        rating_distribution,
        x="rating",
        y="count",
        title="Распределение оценок",
        labels={
            "rating": "Оценка",
            "count": "Количество отзывов"
        }
    )

    return fig


def create_daily_reviews_chart(df):
    """
    Создаёт график динамики отзывов по дням.
    """

    daily = get_daily_reviews(df)

    fig = px.line(
        daily,
        x="date",
        y="reviews_count",
        markers=True,
        title="Динамика количества отзывов по дням",
        labels={
            "date": "Дата",
            "reviews_count": "Количество отзывов"
        }
    )

    return fig


def create_product_rating_chart(product_stats):
    """
    Создаёт график среднего рейтинга товаров.
    """

    fig = px.bar(
        product_stats,
        x="product",
        y="average_rating",
        title="Средний рейтинг товаров",
        labels={
            "product": "Товар",
            "average_rating": "Средний рейтинг"
        }
    )

    fig.update_layout(xaxis_tickangle=-45)

    return fig


def get_russian_font_path():
    """
    Ищет системный шрифт для корректного отображения русского текста в облаке слов.
    """

    possible_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def create_wordcloud_figure(df):
    """
    Создаёт облако слов по всем отзывам.
    """

    text = " ".join(df["text"].astype(str).tolist())

    font_path = get_russian_font_path()

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        font_path=font_path,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig