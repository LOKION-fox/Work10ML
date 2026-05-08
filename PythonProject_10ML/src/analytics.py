import re
from collections import Counter

import pandas as pd


def get_general_statistics(df):
    """
    Возвращает общую статистику по отзывам.
    """

    stats = {
        "total_reviews": len(df),
        "average_rating": df["rating"].mean(),
        "unique_products": df["product"].nunique(),
        "unique_users": df["user_id"].nunique(),
        "positive_count": len(df[df["sentiment"] == 1]),
        "negative_count": len(df[df["sentiment"] == 0]),
        "neutral_count": len(df[df["sentiment"].isna()])
    }

    return stats


def get_product_statistics(df):
    """
    Возвращает статистику по товарам:
    средняя оценка, количество отзывов, доля позитива.
    """

    work_df = df.copy()

    product_stats = work_df.groupby("product").agg(
        average_rating=("rating", "mean"),
        reviews_count=("rating", "count"),
        positive_reviews=("sentiment", lambda x: (x == 1).sum()),
        negative_reviews=("sentiment", lambda x: (x == 0).sum())
    ).reset_index()

    product_stats["positive_share"] = (
        product_stats["positive_reviews"] / product_stats["reviews_count"]
    )

    product_stats = product_stats.sort_values(
        by="average_rating",
        ascending=False
    )

    return product_stats


def get_top_positive_products(df, top_n=5):
    """
    Возвращает товары с наибольшей долей позитивных отзывов.
    """

    product_stats = get_product_statistics(df)

    product_stats = product_stats.sort_values(
        by="positive_share",
        ascending=False
    )

    return product_stats.head(top_n)


def get_top_negative_words(df, top_n=5):
    """
    Возвращает 5 самых частых слов в негативных отзывах.
    """

    negative_df = df[df["sentiment"] == 0].copy()

    all_text = " ".join(negative_df["text"].astype(str).tolist()).lower()

    words = re.findall(r"[а-яёa-z]+", all_text)

    stop_words = {
        "и", "в", "во", "на", "но", "а", "с", "со", "за", "по",
        "к", "у", "от", "до", "для", "не", "что", "это", "как",
        "товар", "покупка", "модель", "очень", "больше", "такой",
        "через", "несколько", "дней", "пришлось", "работает",
        "пользоваться", "ожидал", "ожиданий"
    }

    filtered_words = []

    for word in words:
        if word not in stop_words and len(word) > 2:
            filtered_words.append(word)

    counter = Counter(filtered_words)

    return counter.most_common(top_n)


def get_daily_reviews(df):
    """
    Возвращает количество отзывов по дням.
    """

    daily = df.groupby(df["date"].dt.date).size().reset_index(name="reviews_count")

    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values("date")

    return daily


def get_rating_distribution(df):
    """
    Возвращает распределение оценок.
    """

    rating_distribution = df["rating"].value_counts().sort_index().reset_index()

    rating_distribution.columns = ["rating", "count"]

    return rating_distribution