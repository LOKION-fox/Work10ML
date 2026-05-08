import os
import sqlite3
import pandas as pd


DB_PATH = "reviews.db"
CSV_PATH = "data/reviews_dataset_850_complex.csv"


def get_connection(db_path=DB_PATH):
    """
    Создаёт подключение к базе данных SQLite.
    """

    connection = sqlite3.connect(db_path, check_same_thread=False)

    return connection


def create_reviews_table(connection):
    """
    Создаёт таблицу отзывов, если она ещё не существует.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            product TEXT NOT NULL,
            date TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT,
            phone TEXT,
            sentiment INTEGER,
            sentiment_label TEXT
        )
        """
    )

    connection.commit()


def load_csv_dataset(csv_path=CSV_PATH):
    """
    Загружает CSV-датасет.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден.")

    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    df = df.dropna(subset=["text", "rating", "product", "date", "user_id"])

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["rating"] = df["rating"].astype(int)

    return df


def init_database_from_csv(csv_path=CSV_PATH, db_path=DB_PATH):
    """
    Если база пустая, загружает в неё данные из CSV.
    """

    connection = get_connection(db_path)

    create_reviews_table(connection)

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM reviews")
    count = cursor.fetchone()[0]

    if count == 0:
        df = load_csv_dataset(csv_path)

        columns = [
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

        df[columns].to_sql(
            "reviews",
            connection,
            if_exists="append",
            index=False
        )

    connection.close()


def load_reviews_from_db(db_path=DB_PATH):
    """
    Загружает все отзывы из SQLite в DataFrame.
    """

    connection = get_connection(db_path)

    create_reviews_table(connection)

    df = pd.read_sql_query("SELECT * FROM reviews", connection)

    connection.close()

    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    return df


def add_review_to_db(
    text,
    rating,
    product,
    date,
    user_id,
    user_name,
    phone,
    sentiment,
    sentiment_label,
    db_path=DB_PATH
):
    """
    Добавляет новый отзыв пользователя в SQLite.
    """

    connection = get_connection(db_path)

    create_reviews_table(connection)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO reviews (
            text,
            rating,
            product,
            date,
            user_id,
            user_name,
            phone,
            sentiment,
            sentiment_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            text,
            rating,
            product,
            date,
            user_id,
            user_name,
            phone,
            sentiment,
            sentiment_label
        )
    )

    connection.commit()
    connection.close()