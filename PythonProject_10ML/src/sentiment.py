import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def prepare_binary_dataset(df):
    """
    Подготавливает датасет для бинарной классификации тональности.

    Используются только позитивные и негативные отзывы:
    sentiment = 1 — позитивный отзыв
    sentiment = 0 — негативный отзыв

    Нейтральные отзывы, у которых sentiment пустой, не используются
    при обучении моделей классификации.
    """

    df_binary = df[df["sentiment"].notna()].copy()

    df_binary["sentiment"] = df_binary["sentiment"].astype(int)

    X = df_binary["text"].astype(str)
    y = df_binary["sentiment"]

    return X, y


def train_base_sentiment_model(df):
    """
    Обучает основную модель тональности для формы добавления отзыва.

    Эта модель используется в веб-интерфейсе, когда пользователь вводит
    новый отзыв. После ввода текста модель возвращает:
    позитивный / негативный и процент уверенности.

    Здесь используется более сильная настройка TF-IDF, потому что для
    пользовательского предсказания нужна нормальная рабочая модель.
    """

    X, y = prepare_binary_dataset(df)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1500,
                    lowercase=True,
                    ngram_range=(1, 2)
                )
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    model.fit(X, y)

    return model


def predict_sentiment(text, model):
    """
    Возвращает тональность и уверенность модели.

    На вход подаётся текст отзыва.
    На выходе возвращается словарь:

    {
        "sentiment": 1,
        "label": "позитивный",
        "sentiment_label": "positive",
        "confidence": 87.5
    }
    """

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        label = "позитивный"
        sentiment_label = "positive"
    else:
        label = "негативный"
        sentiment_label = "negative"

    return {
        "sentiment": int(prediction),
        "label": label,
        "sentiment_label": sentiment_label,
        "confidence": confidence
    }


def get_models_for_comparison():
    """
    Возвращает модели для сравнения на странице 'Сравнение моделей'.

    Используются:
    1. LogisticRegression
    2. RandomForest
    3. XGBoost, если библиотека установлена
    """

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_split=6,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=70,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    return models


def get_comparison_stop_words():
    """
    Стоп-слова для более честного сравнения моделей.

    В синтетическом датасете часть слов слишком явно выдаёт тональность:
    'доволен', 'понравилось', 'разочаровала', 'недоволен' и т.д.

    Если оставить эти слова, все модели легко получают 100%.
    Поэтому для страницы сравнения моделей мы убираем часть слишком
    очевидных слов, чтобы сравнение было сложнее и реалистичнее.
    """

    return [
        "и", "в", "во", "на", "но", "а", "с", "со", "за", "по",
        "к", "у", "от", "до", "для", "не", "что", "это", "как",
        "товар", "покупка", "модель", "использование", "использования",
        "день", "дней", "после", "сначала", "в целом", "очень",

        "отлично", "отличный", "отличная", "отличное",
        "хороший", "хорошая", "хорошее", "хорошо",
        "понравилось", "понравилась", "понравился",
        "доволен", "довольна", "удобная", "удобный", "удобно",
        "приятно", "стабильно", "стабильность",
        "качественно", "качественный", "качественная",

        "плохо", "плохой", "плохая", "плохое",
        "ужасно", "ужасный", "ужасная", "ужасное",
        "недоволен", "недовольна",
        "разочаровал", "разочаровала", "разочаровало",
        "дефект", "дефекты", "сбои", "сбой",
        "слабое", "слабый", "слабая",
        "неудачная", "неудачный",
        "проблема", "проблемы"
    ]


def compare_sentiment_models(df):
    """
    Сравнивает модели классификации тональности:
    LogisticRegression, RandomForest, XGBoost.

    Для каждой модели рассчитываются:
    Accuracy, Precision, Recall, F1.
    """

    X, y = prepare_binary_dataset(df)

    clean_df = pd.DataFrame(
        {
            "text": X.astype(str),
            "sentiment": y.astype(int)
        }
    )

    clean_df = clean_df.drop_duplicates(subset=["text"])

    X = clean_df["text"].astype(str)
    y = clean_df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    models = get_models_for_comparison()

    results = []

    comparison_stop_words = get_comparison_stop_words()

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=120,
                        lowercase=True,
                        ngram_range=(1, 1),
                        stop_words=comparison_stop_words,
                        min_df=2,
                        max_df=0.85
                    )
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        results.append(
            {
                "Модель": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0)
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    return results_df