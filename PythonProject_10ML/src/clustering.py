import re
from collections import Counter

import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer


def get_top_words_for_texts(texts, top_n=10):
    """
    Возвращает топ слов для группы отзывов.
    """

    all_text = " ".join(texts).lower()

    words = re.findall(r"[а-яёa-z]+", all_text)

    stop_words = {
        "и", "в", "во", "на", "но", "а", "с", "со", "за", "по",
        "к", "у", "от", "до", "для", "не", "что", "это", "как",
        "товар", "покупка", "модель", "использование", "использования",
        "день", "дней", "после", "сначала", "в целом", "очень",
        "есть", "если", "пока", "бы", "был", "была", "было",
        "можно", "нужно", "свою", "своих", "свои"
    }

    filtered_words = []

    for word in words:
        if word not in stop_words and len(word) > 2:
            filtered_words.append(word)

    counter = Counter(filtered_words)

    return counter.most_common(top_n)


def create_sentence_embeddings(df):
    """
    Векторизует отзывы с помощью SentenceTransformer.

    Используется multilingual-модель, подходящая для русскоязычных текстов.
    При первом запуске модель может скачиваться из интернета.
    """

    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


def run_review_clustering(df):
    """
    Выполняет кластеризацию отзывов.

    Этапы:
    1. Векторизация отзывов через SentenceTransformer.
    2. Кластеризация KMeans.
    3. Кластеризация DBSCAN.
    4. Снижение размерности через PCA.
    5. Подсчёт топ-10 слов и доли позитивных отзывов для каждого кластера.
    """

    work_df = df.copy().reset_index(drop=True)

    work_df = work_df.dropna(subset=["text"]).copy()
    work_df["text"] = work_df["text"].astype(str)

    embeddings = create_sentence_embeddings(work_df)

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(embeddings)

    dbscan = DBSCAN(
        eps=0.65,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(embeddings)

    pca = PCA(
        n_components=2,
        random_state=42
    )

    points = pca.fit_transform(embeddings)

    work_df["pca_x"] = points[:, 0]
    work_df["pca_y"] = points[:, 1]
    work_df["kmeans_cluster"] = kmeans_labels
    work_df["dbscan_cluster"] = dbscan_labels

    cluster_rows = []

    for cluster_id in sorted(work_df["kmeans_cluster"].unique()):
        cluster_df = work_df[work_df["kmeans_cluster"] == cluster_id]

        texts = cluster_df["text"].astype(str).tolist()

        top_words = get_top_words_for_texts(texts, top_n=10)

        positive_share = (cluster_df["sentiment"] == 1).mean()

        cluster_rows.append(
            {
                "Алгоритм": "KMeans",
                "Кластер": int(cluster_id),
                "Количество отзывов": len(cluster_df),
                "Доля позитивных": positive_share,
                "Топ-10 слов": ", ".join([word for word, count in top_words])
            }
        )

    for cluster_id in sorted(work_df["dbscan_cluster"].unique()):
        cluster_df = work_df[work_df["dbscan_cluster"] == cluster_id]

        texts = cluster_df["text"].astype(str).tolist()

        top_words = get_top_words_for_texts(texts, top_n=10)

        positive_share = (cluster_df["sentiment"] == 1).mean()

        cluster_rows.append(
            {
                "Алгоритм": "DBSCAN",
                "Кластер": int(cluster_id),
                "Количество отзывов": len(cluster_df),
                "Доля позитивных": positive_share,
                "Топ-10 слов": ", ".join([word for word, count in top_words])
            }
        )

    cluster_stats = pd.DataFrame(cluster_rows)

    scores = {}

    try:
        scores["KMeans silhouette"] = silhouette_score(embeddings, kmeans_labels)
    except Exception:
        scores["KMeans silhouette"] = None

    try:
        valid_mask = dbscan_labels != -1

        valid_labels = dbscan_labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]

        if len(set(valid_labels)) > 1:
            scores["DBSCAN silhouette"] = silhouette_score(
                valid_embeddings,
                valid_labels
            )
        else:
            scores["DBSCAN silhouette"] = None

    except Exception:
        scores["DBSCAN silhouette"] = None

    return work_df, cluster_stats, scores


def create_cluster_plot(clustered_df, algorithm="KMeans"):
    """
    Создаёт интерактивную визуализацию кластеров через Plotly.

    При наведении на точку отображается:
    товар, оценка, тональность и текст отзыва.
    """

    if algorithm == "KMeans":
        cluster_column = "kmeans_cluster"
        title = "Визуализация кластеров отзывов: KMeans"
    else:
        cluster_column = "dbscan_cluster"
        title = "Визуализация кластеров отзывов: DBSCAN"

    plot_df = clustered_df.copy()

    plot_df["cluster"] = plot_df[cluster_column].astype(str)

    fig = px.scatter(
        plot_df,
        x="pca_x",
        y="pca_y",
        color="cluster",
        hover_data={
            "product": True,
            "rating": True,
            "sentiment_label": True,
            "text": True,
            "pca_x": False,
            "pca_y": False,
            "cluster": True
        },
        title=title,
        labels={
            "pca_x": "PCA 1",
            "pca_y": "PCA 2",
            "cluster": "Кластер"
        }
    )

    fig.update_layout(
        height=650
    )

    return fig