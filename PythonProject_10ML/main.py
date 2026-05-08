import os
import re
import pickle
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================
# 1. НАСТРОЙКИ ПРОЕКТА
# =========================

DATASET_PATH = "data/reviews_dataset_850_complex.csv"
REPORT_PATH = "student_report.png"
MODEL_PATH = "models/sentiment_model.pkl"


# =========================
# 2. ЗАГРУЗКА ДАННЫХ
# =========================

def load_dataset(path):
    """
    Загружает датасет из CSV-файла.
    Проверяет наличие обязательных столбцов.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден. Проверь путь к датасету.")

    df = pd.read_csv(path)

    required_columns = [
        "id",
        "text",
        "rating",
        "product",
        "date",
        "user_id",
        "user_name",
        "phone",
        "sentiment",
        "sentiment_label"
    ]

    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    df = df.dropna(subset=["text", "rating", "product", "date"])

    return df


# =========================
# 3. ЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ
# =========================

def mask_phone(phone):
    """
    Маскирует номер телефона.
    Пример:
    +79261234567 -> +792***4567
    """

    phone = str(phone)

    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]

    return phone


def mask_name(name):
    """
    Маскирует имя пользователя.
    Пример:
    Иван Иванов -> Ив*******ов
    """

    name = str(name)

    if len(name) == 0:
        return name

    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)

    return name[:2] + "*" * (len(name) - 4) + name[-2:]


def create_protected_dataframe(df):
    """
    Создаёт копию датасета с замаскированными персональными данными.
    """

    protected_df = df.copy()

    protected_df["user_name"] = protected_df["user_name"].apply(mask_name)
    protected_df["phone"] = protected_df["phone"].apply(mask_phone)

    return protected_df


# =========================
# 4. ОБУЧЕНИЕ МОДЕЛИ ТОНАЛЬНОСТИ
# =========================

def train_sentiment_model(df):
    """
    Обучает модель классификации тональности отзывов.
    Используются только позитивные и негативные отзывы.
    Нейтральные отзывы исключаются из обучения.
    """

    df_binary = df[df["sentiment"].notna()].copy()

    X = df_binary["text"]
    y = df_binary["sentiment"].astype(int)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        lowercase=True
    )

    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, metrics, cm, df_binary


# =========================
# 5. СОХРАНЕНИЕ МОДЕЛИ
# =========================

def save_model(model, vectorizer, path):
    """
    Сохраняет модель и векторизатор в файл.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "vectorizer": vectorizer
            },
            file
        )


# =========================
# 6. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ТОНАЛЬНОСТИ
# =========================

def predict_sentiment(text, model, vectorizer):
    """
    Возвращает тональность отзыва и процент уверенности.

    Результат:
    {
        "label": "позитивный",
        "confidence": 87.5
    }
    """

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        label = "позитивный"
    else:
        label = "негативный"

    return {
        "label": label,
        "confidence": confidence
    }


# =========================
# 7. ТОП ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА
# =========================

def get_top_positive_products(df, top_n=5):
    """
    Возвращает топ товаров по доле позитивных отзывов.
    Учитываются только отзывы с известной тональностью.
    """

    df_binary = df[df["sentiment"].notna()].copy()
    df_binary["sentiment"] = df_binary["sentiment"].astype(int)

    product_stats = df_binary.groupby("product").agg(
        total_reviews=("sentiment", "count"),
        positive_reviews=("sentiment", "sum")
    ).reset_index()

    product_stats["positive_share"] = (
        product_stats["positive_reviews"] / product_stats["total_reviews"]
    )

    product_stats = product_stats.sort_values(
        by="positive_share",
        ascending=False
    )

    return product_stats.head(top_n)


# =========================
# 8. 5 САМЫХ ЧАСТЫХ СЛОВ В НЕГАТИВНЫХ ОТЗЫВАХ
# =========================

def get_top_negative_words(df, top_n=5):
    """
    Возвращает самые частые слова в негативных отзывах.
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

    word_counter = Counter(filtered_words)

    return word_counter.most_common(top_n)


# =========================
# 9. ОБЩАЯ СТАТИСТИКА
# =========================

def get_general_statistics(df):
    """
    Возвращает основную статистику по отзывам.
    """

    total_reviews = len(df)
    average_rating = df["rating"].mean()
    unique_products = df["product"].nunique()
    unique_users = df["user_id"].nunique()

    positive_count = len(df[df["sentiment"] == 1])
    negative_count = len(df[df["sentiment"] == 0])
    neutral_count = len(df[df["sentiment"].isna()])

    stats = {
        "total_reviews": total_reviews,
        "average_rating": average_rating,
        "unique_products": unique_products,
        "unique_users": unique_users,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count
    }

    return stats


# =========================
# 10. СОЗДАНИЕ PNG-ОТЧЁТА
# =========================

def create_report_png(
    df,
    metrics,
    cm,
    top_products,
    negative_words,
    example_prediction,
    output_path
):
    """
    Создаёт итоговый PNG-отчёт student_report.png.
    В файл входят статистика и графики.
    """

    general_stats = get_general_statistics(df)

    daily_counts = df.groupby(df["date"].dt.date).size()
    product_counts = df["product"].value_counts()
    rating_counts = df["rating"].value_counts().sort_index()

    sentiment_counts = {
        "Негативные": general_stats["negative_count"],
        "Нейтральные": general_stats["neutral_count"],
        "Позитивные": general_stats["positive_count"]
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Анализ отзывов клиентов интернет-магазина", fontsize=16)

    # График 1 — распределение оценок
    axes[0, 0].bar(rating_counts.index.astype(str), rating_counts.values)
    axes[0, 0].set_title("Распределение оценок")
    axes[0, 0].set_xlabel("Оценка")
    axes[0, 0].set_ylabel("Количество отзывов")

    # График 2 — динамика отзывов по дням
    axes[0, 1].plot(daily_counts.index, daily_counts.values, marker="o")
    axes[0, 1].set_title("Динамика отзывов по дням")
    axes[0, 1].set_xlabel("Дата")
    axes[0, 1].set_ylabel("Количество отзывов")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # График 3 — количество отзывов по товарам
    axes[0, 2].bar(product_counts.index, product_counts.values)
    axes[0, 2].set_title("Количество отзывов по товарам")
    axes[0, 2].set_xlabel("Товар")
    axes[0, 2].set_ylabel("Количество")
    axes[0, 2].tick_params(axis="x", rotation=75)

    # График 4 — матрица ошибок
    axes[1, 0].imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    axes[1, 0].set_title("Матрица ошибок")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(["Негатив", "Позитив"])
    axes[1, 0].set_yticklabels(["Негатив", "Позитив"])
    axes[1, 0].set_xlabel("Предсказанный класс")
    axes[1, 0].set_ylabel("Истинный класс")

    # График 5 — соотношение тональности
    axes[1, 1].pie(
        sentiment_counts.values(),
        labels=sentiment_counts.keys(),
        autopct="%1.1f%%"
    )
    axes[1, 1].set_title("Соотношение тональностей")

    # График 6 — текстовая статистика
    axes[1, 2].axis("off")

    top_products_text = ""

    for _, row in top_products.iterrows():
        top_products_text += (
            f"{row['product']}: "
            f"{row['positive_share']:.1%} "
            f"({int(row['positive_reviews'])}/{int(row['total_reviews'])})\n"
        )

    negative_words_text = ""

    for word, count in negative_words:
        negative_words_text += f"{word}: {count}\n"

    text_block = (
        f"Общая статистика:\n"
        f"Всего отзывов: {general_stats['total_reviews']}\n"
        f"Средняя оценка: {general_stats['average_rating']:.2f}\n"
        f"Товаров: {general_stats['unique_products']}\n"
        f"Пользователей: {general_stats['unique_users']}\n\n"
        f"Метрики модели:\n"
        f"Accuracy: {metrics['accuracy']:.2%}\n"
        f"Precision: {metrics['precision']:.2%}\n"
        f"Recall: {metrics['recall']:.2%}\n"
        f"F1-score: {metrics['f1']:.2%}\n\n"
        f"Пример предсказания:\n"
        f"{example_prediction['label']}, "
        f"{example_prediction['confidence']:.1f}%\n\n"
        f"Топ товаров по доле позитива:\n"
        f"{top_products_text}\n"
        f"5 частых слов в негативных отзывах:\n"
        f"{negative_words_text}"
    )

    axes[1, 2].text(
        0,
        1,
        text_block,
        fontsize=10,
        verticalalignment="top"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# 11. ОСНОВНОЙ ЗАПУСК
# =========================

def main():
    print("Загрузка датасета...")

    df = load_dataset(DATASET_PATH)

    print(f"Датасет загружен. Количество записей: {len(df)}")

    print("\nСоздание защищённой копии данных...")

    protected_df = create_protected_dataframe(df)

    print("Пример маскировки персональных данных:")
    print(f"Имя до: {df['user_name'].iloc[0]}")
    print(f"Имя после: {protected_df['user_name'].iloc[0]}")
    print(f"Телефон до: {df['phone'].iloc[0]}")
    print(f"Телефон после: {protected_df['phone'].iloc[0]}")

    print("\nОбучение модели классификации тональности...")

    model, vectorizer, metrics, cm, df_binary = train_sentiment_model(df)

    save_model(model, vectorizer, MODEL_PATH)

    print(f"Модель сохранена в файл: {MODEL_PATH}")

    print("\nМетрики модели:")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-score:  {metrics['f1']:.2%}")

    print("\nМатрица ошибок:")
    print(cm)

    print("\nПроверка функции predict_sentiment:")

    test_review = "Товар отличный, качество хорошее, покупкой полностью доволен."
    prediction = predict_sentiment(test_review, model, vectorizer)

    print(f"Текст: {test_review}")
    print(f"Результат: {prediction['label']}")
    print(f"Уверенность: {prediction['confidence']:.1f}%")

    print("\nТоп товаров по доле позитивных отзывов:")

    top_products = get_top_positive_products(df, top_n=5)

    for _, row in top_products.iterrows():
        print(
            f"{row['product']} — "
            f"{row['positive_share']:.1%} "
            f"({int(row['positive_reviews'])}/{int(row['total_reviews'])})"
        )

    print("\n5 самых частых слов в негативных отзывах:")

    negative_words = get_top_negative_words(df, top_n=5)

    for word, count in negative_words:
        print(f"{word}: {count}")

    print(f"\nСоздание PNG-отчёта: {REPORT_PATH}")

    create_report_png(
        df=df,
        metrics=metrics,
        cm=cm,
        top_products=top_products,
        negative_words=negative_words,
        example_prediction=prediction,
        output_path=REPORT_PATH
    )

    print(f"Готово. Отчёт сохранён в файл: {REPORT_PATH}")


if __name__ == "__main__":
    main()