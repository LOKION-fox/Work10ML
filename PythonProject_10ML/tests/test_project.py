import pandas as pd

from src.privacy import mask_phone, mask_name
from src.data import load_csv_dataset
from src.analytics import get_general_statistics
from src.recommendations import recommend_products_for_user
from src.sentiment import train_base_sentiment_model, predict_sentiment
from src.advanced_models import calculate_metrics


def test_mask_phone():
    """
    Проверяет маскировку номера телефона.
    """

    phone = "+79261234567"

    masked = mask_phone(phone)

    assert masked == "+792***4567"


def test_mask_name():
    """
    Проверяет маскировку имени пользователя.
    """

    name = "Алексей Иванов"

    masked = mask_name(name)

    assert masked.startswith("Ал")
    assert masked.endswith("ов")
    assert "*" in masked


def test_load_dataset():
    """
    Проверяет загрузку CSV-датасета.
    """

    df = load_csv_dataset("data/reviews_dataset_850_complex.csv")

    assert len(df) >= 200
    assert "text" in df.columns
    assert "rating" in df.columns
    assert "product" in df.columns
    assert "date" in df.columns
    assert "user_id" in df.columns


def test_general_statistics():
    """
    Проверяет вычисление общей статистики.
    """

    df = load_csv_dataset("data/reviews_dataset_850_complex.csv")

    stats = get_general_statistics(df)

    assert stats["total_reviews"] >= 200
    assert stats["unique_products"] > 0
    assert stats["unique_users"] > 0
    assert stats["average_rating"] > 0


def test_predict_sentiment():
    """
    Проверяет работу функции predict_sentiment.
    """

    df = load_csv_dataset("data/reviews_dataset_850_complex.csv")

    model = train_base_sentiment_model(df)

    result = predict_sentiment(
        "Товар отличный, качество хорошее, покупкой доволен.",
        model
    )

    assert result["label"] in ["позитивный", "негативный"]
    assert result["sentiment"] in [0, 1]
    assert 0 <= result["confidence"] <= 100


def test_recommendations():
    """
    Проверяет работу рекомендательной системы.
    """

    df = load_csv_dataset("data/reviews_dataset_850_complex.csv")

    user_id = df["user_id"].iloc[0]

    recommendations = recommend_products_for_user(
        df=df,
        user_id=user_id,
        top_n=3
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3

    if len(recommendations) > 0:
        assert "product" in recommendations[0]
        assert "predicted_rating" in recommendations[0]


def test_calculate_metrics():
    """
    Проверяет вычисление метрик классификации.
    """

    y_test = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 1]

    metrics = calculate_metrics(y_test, y_pred)

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1" in metrics

    assert 0 <= metrics["Accuracy"] <= 1
    assert 0 <= metrics["Precision"] <= 1
    assert 0 <= metrics["Recall"] <= 1
    assert 0 <= metrics["F1"] <= 1