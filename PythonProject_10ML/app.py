import os
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from streamlit_cookies_manager_ext import EncryptedCookieManager

from src.data import (
    init_database_from_csv,
    load_reviews_from_db,
    add_review_to_db
)

from src.sentiment import (
    train_base_sentiment_model,
    predict_sentiment,
    compare_sentiment_models
)

from src.advanced_models import compare_models_before_after_tuning

from src.analytics import (
    get_general_statistics,
    get_product_statistics,
    get_top_positive_products,
    get_top_negative_words
)

from src.privacy import create_protected_dataframe

from src.recommendations import recommend_products_for_user

from src.visualization import (
    create_rating_distribution_chart,
    create_daily_reviews_chart,
    create_product_rating_chart,
    create_wordcloud_figure
)

from src.clustering import (
    run_review_clustering,
    create_cluster_plot
)

from src.logger import (
    log_action,
    read_last_logs
)


st.set_page_config(
    page_title="ML-анализ отзывов интернет-магазина",
    page_icon="📊",
    layout="wide"
)


COOKIE_KEY = "user_id"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


cookies = EncryptedCookieManager(
    prefix="ml_reviews_app/",
    password=os.environ.get(
        "COOKIES_PASSWORD",
        "local_dev_secret_password_12345"
    )
)


if not cookies.ready():
    st.stop()


@st.cache_data
def load_data_cached():
    """
    Загружает отзывы из базы данных SQLite.
    """

    df = load_reviews_from_db()

    return df


@st.cache_resource
def train_model_cached(df):
    """
    Обучает базовую модель тональности.
    """

    model = train_base_sentiment_model(df)

    return model


@st.cache_data
def compare_models_cached(df):
    """
    Кэширует сравнение трёх моделей для оценки 4.
    """

    return compare_sentiment_models(df)


@st.cache_data
def compare_advanced_models_cached(df):
    """
    Кэширует расширенное сравнение моделей для оценки 5.
    """

    results_df, roc_data, available_models_count = compare_models_before_after_tuning(df)

    return results_df, roc_data, available_models_count


@st.cache_data
def clustering_cached(df):
    """
    Кэширует кластеризацию отзывов.
    """

    clustered_df, cluster_stats, scores = run_review_clustering(df)

    return clustered_df, cluster_stats, scores


def reset_cache():
    """
    Очищает кэш после добавления нового отзыва.
    """

    st.cache_data.clear()
    st.cache_resource.clear()


def create_new_user_id():
    """
    Создаёт новый уникальный ID пользователя.
    """

    return "web_" + str(uuid.uuid4())[:8]


def init_user():
    """
    Создаёт или восстанавливает пользователя через cookie браузера.
    """

    if COOKIE_KEY in cookies and cookies[COOKIE_KEY].strip() != "":
        user_id = cookies[COOKIE_KEY].strip()
        st.session_state["user_id"] = user_id
        return user_id

    user_id = create_new_user_id()

    cookies[COOKIE_KEY] = user_id
    cookies.save()

    st.session_state["user_id"] = user_id

    return user_id


def change_user(new_user_id):
    """
    Меняет текущего пользователя и сохраняет его в cookie.
    """

    st.session_state["user_id"] = new_user_id
    cookies[COOKIE_KEY] = new_user_id
    cookies.save()


def init_admin_state():
    """
    Инициализирует состояние администратора.
    """

    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False

    return st.session_state["is_admin"]


def check_admin_password(password):
    """
    Проверяет пароль администратора.
    """

    return password == ADMIN_PASSWORD


def show_sidebar():
    """
    Боковое меню приложения.
    Пункт 'Логи приложения' отображается только после входа администратора.
    """

    st.sidebar.title("Навигация")

    st.sidebar.subheader("Пользователь")

    current_user_id = st.session_state["user_id"]

    st.sidebar.info(
        f"Текущий ID пользователя:\n\n`{current_user_id}`"
    )

    with st.sidebar.expander("Управление пользователем"):
        manual_user_id = st.text_input(
            "Войти по user_id",
            value=current_user_id
        )

        if st.button("Применить user_id"):
            if manual_user_id.strip() == "":
                st.error("user_id не может быть пустым.")
            else:
                change_user(manual_user_id.strip())
                st.success("user_id сохранён в cookie.")
                st.rerun()

        if st.button("Создать нового пользователя"):
            new_user_id = create_new_user_id()
            change_user(new_user_id)
            st.success("Создан новый пользователь.")
            st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.subheader("Администратор")

    if st.session_state["is_admin"]:
        st.sidebar.success("Вход администратора выполнен.")

        if st.sidebar.button("Выйти из режима администратора"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        with st.sidebar.expander("Вход администратора"):
            admin_password = st.text_input(
                "Пароль администратора",
                type="password"
            )

            if st.button("Войти как администратор"):
                if check_admin_password(admin_password):
                    st.session_state["is_admin"] = True
                    st.success("Администратор авторизован.")
                    st.rerun()
                else:
                    st.error("Неверный пароль администратора.")

    st.sidebar.markdown("---")

    pages = [
        "Главная",
        "Добавить отзыв",
        "Рекомендации",
        "Аналитика",
        "Статистика товаров",
        "Сравнение моделей",
        "Расширенное сравнение моделей",
        "Кластеры отзывов",
        "Защита данных"
    ]

    if st.session_state["is_admin"]:
        pages.append("Логи приложения")

    page = st.sidebar.radio(
        "Выберите раздел",
        pages
    )

    return page


def show_main_page(df, user_id):
    """
    Главная страница.
    """

    log_action(user_id, "open_page", "Главная")

    st.title("ML-продукт для анализа отзывов интернет-магазина")

    st.write(
        """
        Приложение выполняет анализ отзывов клиентов интернет-магазина:
        классифицирует тональность отзывов, сохраняет данные в SQLite,
        строит аналитику, формирует персонализированные рекомендации,
        сравнивает модели машинного обучения и выполняет кластеризацию отзывов.
        """
    )

    stats = get_general_statistics(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Всего отзывов", stats["total_reviews"])
    col2.metric("Средняя оценка", f"{stats['average_rating']:.2f}")
    col3.metric("Товаров", stats["unique_products"])
    col4.metric("Пользователей", stats["unique_users"])

    st.markdown("---")

    col5, col6, col7 = st.columns(3)

    col5.metric("Позитивных отзывов", stats["positive_count"])
    col6.metric("Негативных отзывов", stats["negative_count"])
    col7.metric("Нейтральных отзывов", stats["neutral_count"])

    st.subheader("Последние отзывы")

    latest = df.sort_values("date", ascending=False).head(10)

    st.dataframe(
        latest[["date", "product", "rating", "text", "sentiment_label"]],
        use_container_width=True
    )


def show_add_review_page(df, model, user_id):
    """
    Страница добавления отзыва.
    """

    log_action(user_id, "open_page", "Добавить отзыв")

    st.title("Добавить отзыв")

    st.write(
        """
        Пользователь может выбрать товар, поставить оценку от 1 до 5
        и оставить текстовый отзыв. После отправки отзыв сохраняется в SQLite,
        а модель показывает предсказанную тональность.
        """
    )

    products = sorted(df["product"].dropna().unique().tolist())

    with st.form("review_form"):
        product = st.selectbox("Выберите товар", products)

        rating = st.slider(
            "Оценка товара",
            min_value=1,
            max_value=5,
            value=5
        )

        text = st.text_area(
            "Текст отзыва",
            placeholder="Например: товар понравился, качество хорошее..."
        )

        user_name = st.text_input(
            "Ваше имя",
            value="Пользователь сайта"
        )

        phone = st.text_input(
            "Телефон",
            value="+79000000000"
        )

        submitted = st.form_submit_button("Отправить отзыв")

    if submitted:
        if len(text.strip()) < 5:
            log_action(user_id, "add_review", "Ошибка: короткий текст")
            st.error("Текст отзыва слишком короткий.")
            return

        prediction = predict_sentiment(text, model)

        add_review_to_db(
            text=text,
            rating=rating,
            product=product,
            date=datetime.now().strftime("%Y-%m-%d"),
            user_id=user_id,
            user_name=user_name,
            phone=phone,
            sentiment=prediction["sentiment"],
            sentiment_label=prediction["sentiment_label"]
        )

        reset_cache()

        log_action(
            user_id,
            "add_review",
            f"Отзыв сохранён, sentiment={prediction['sentiment_label']}"
        )

        st.success("Отзыв успешно сохранён в SQLite.")

        st.info(
            f"Предсказанная тональность: "
            f"**{prediction['label']}** "
            f"({prediction['confidence']:.1f}% уверенности)"
        )

        st.write(f"Отзыв сохранён для пользователя: `{user_id}`")


def show_recommendations_page(df, user_id):
    """
    Страница персонализированных рекомендаций.
    """

    log_action(user_id, "open_page", "Рекомендации")

    st.title("Персонализированные рекомендации")

    st.write(
        """
        Рекомендации строятся методом user-based collaborative filtering.
        Система ищет пользователей с похожими оценками и прогнозирует,
        какие товары могут понравиться текущему пользователю.
        """
    )

    st.info(f"Рекомендации строятся для пользователя: `{user_id}`")

    user_reviews = df[df["user_id"] == user_id]

    if len(user_reviews) == 0:
        st.warning(
            """
            У текущего пользователя пока нет собственных отзывов.
            Поэтому система покажет популярные товары по среднему рейтингу.
            Чтобы получить персональные рекомендации, добавьте 2–3 отзыва.
            """
        )
    else:
        st.subheader("История оценок текущего пользователя")

        st.dataframe(
            user_reviews[["date", "product", "rating", "text"]].sort_values(
                "date",
                ascending=False
            ),
            use_container_width=True
        )

    recommendations = recommend_products_for_user(
        df=df,
        user_id=user_id,
        top_n=5
    )

    st.subheader("Рекомендуемые товары")

    if len(recommendations) == 0:
        st.warning("Пока нет данных для рекомендаций.")
        return

    for item in recommendations:
        st.success(
            f"{item['product']} — прогнозируемая оценка: "
            f"{item['predicted_rating']:.2f}"
        )


def show_analytics_page(df, user_id):
    """
    Страница аналитики.
    """

    log_action(user_id, "open_page", "Аналитика")

    st.title("Аналитика отзывов")

    col1, col2 = st.columns(2)

    with col1:
        rating_fig = create_rating_distribution_chart(df)
        st.plotly_chart(rating_fig, use_container_width=True)

    with col2:
        daily_fig = create_daily_reviews_chart(df)
        st.plotly_chart(daily_fig, use_container_width=True)

    st.subheader("Облако слов по отзывам")

    wordcloud_fig = create_wordcloud_figure(df)

    st.pyplot(wordcloud_fig)

    st.subheader("Топ товаров по доле позитива")

    top_products = get_top_positive_products(df, top_n=5)

    st.dataframe(
        top_products,
        use_container_width=True
    )

    st.subheader("5 самых частых слов в негативных отзывах")

    negative_words = get_top_negative_words(df, top_n=5)

    negative_words_df = pd.DataFrame(
        negative_words,
        columns=["Слово", "Количество"]
    )

    st.dataframe(
        negative_words_df,
        use_container_width=True
    )


def show_product_statistics_page(df, user_id):
    """
    Страница статистики товаров.
    """

    log_action(user_id, "open_page", "Статистика товаров")

    st.title("Статистика по товарам")

    product_stats = get_product_statistics(df)

    product_stats_show = product_stats.copy()

    product_stats_show["average_rating"] = product_stats_show["average_rating"].round(2)
    product_stats_show["positive_share"] = product_stats_show["positive_share"].round(3)

    product_stats_show = product_stats_show.rename(
        columns={
            "product": "Товар",
            "average_rating": "Средний рейтинг",
            "reviews_count": "Количество отзывов",
            "positive_reviews": "Позитивных отзывов",
            "negative_reviews": "Негативных отзывов",
            "neutral_reviews": "Нейтральных отзывов",
            "positive_share": "Доля позитива"
        }
    )

    st.dataframe(
        product_stats_show,
        use_container_width=True
    )

    fig = create_product_rating_chart(product_stats)

    st.plotly_chart(fig, use_container_width=True)


def show_model_comparison_page(df, user_id):
    """
    Страница сравнения трёх базовых моделей.
    """

    log_action(user_id, "open_page", "Сравнение моделей")

    st.title("Сравнение моделей классификации тональности")

    st.write(
        """
        На этой странице сравниваются три модели:
        LogisticRegression, RandomForest и XGBoost.
        Для каждой модели рассчитываются Accuracy, Precision, Recall и F1-score.
        """
    )

    with st.spinner("Обучение и сравнение моделей..."):
        results_df = compare_models_cached(df)

    shown = results_df.copy()

    for column in ["Accuracy", "Precision", "Recall", "F1"]:
        shown[column] = shown[column].apply(lambda value: f"{value:.2%}")

    st.dataframe(
        shown,
        use_container_width=True
    )

    best_model = results_df.iloc[0]

    st.success(
        f"Лучшая модель по F1-score: "
        f"{best_model['Модель']} "
        f"с результатом {best_model['F1']:.2%}."
    )


def show_advanced_model_comparison_page(df, user_id):
    """
    Страница расширенного сравнения моделей для оценки 5.
    """

    log_action(user_id, "open_page", "Расширенное сравнение моделей")

    st.title("Расширенное сравнение моделей")

    st.write(
        """
        Здесь сравниваются пять моделей:
        LogisticRegression, RandomForest, XGBoost, MLPClassifier и LightGBM.
        Для каждой модели выполняется GridSearchCV минимум по трём гиперпараметрам.
        В таблице показаны метрики до и после подбора гиперпараметров.
        """
    )

    with st.spinner("Выполняется обучение моделей и GridSearchCV. Это может занять несколько минут..."):
        results_df, roc_data, available_models_count = compare_advanced_models_cached(df)

    if available_models_count < 5:
        st.warning(
            """
            Доступно меньше пяти моделей. Проверьте установку библиотек:
            xgboost и lightgbm.
            Команда установки:
            pip install xgboost lightgbm
            """
        )

    shown = results_df.copy()

    for column in ["Accuracy", "Precision", "Recall", "F1"]:
        shown[column] = shown[column].apply(lambda value: f"{value:.2%}")

    st.subheader("Сводная таблица метрик до и после подбора гиперпараметров")

    st.dataframe(
        shown,
        use_container_width=True
    )

    st.subheader("ROC-кривые моделей")

    if len(roc_data) == 0:
        st.warning("Нет данных для построения ROC-кривых.")
    else:
        fig = go.Figure()

        for model_name, values in roc_data.items():
            fig.add_trace(
                go.Scatter(
                    x=values["fpr"],
                    y=values["tpr"],
                    mode="lines",
                    name=f"{model_name}, AUC={values['auc']:.3f}"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Случайная модель",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="ROC-кривые моделей",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    best_row = results_df.sort_values("F1", ascending=False).iloc[0]

    st.success(
        f"Лучший результат: {best_row['Модель']} "
        f"на этапе '{best_row['Этап']}', F1 = {best_row['F1']:.2%}."
    )


def show_clustering_page(df, user_id):
    """
    Страница кластеризации отзывов.
    """

    log_action(user_id, "open_page", "Кластеры отзывов")

    st.title("Кластеризация отзывов")

    st.write(
        """
        На этой странице отзывы векторизуются с помощью SentenceTransformer.
        Затем к векторам применяются алгоритмы KMeans и DBSCAN.
        Для визуализации используется PCA и интерактивный график Plotly.
        """
    )

    with st.spinner("Выполняется SentenceTransformer-векторизация и кластеризация..."):
        clustered_df, cluster_stats, scores = clustering_cached(df)

    st.subheader("Качество кластеризации")

    col1, col2 = st.columns(2)

    col1.metric(
        "KMeans silhouette",
        "нет данных" if scores["KMeans silhouette"] is None else f"{scores['KMeans silhouette']:.3f}"
    )

    col2.metric(
        "DBSCAN silhouette",
        "нет данных" if scores["DBSCAN silhouette"] is None else f"{scores['DBSCAN silhouette']:.3f}"
    )

    st.subheader("Статистика кластеров")

    show_stats = cluster_stats.copy()

    show_stats["Доля позитивных"] = show_stats["Доля позитивных"].round(3)

    st.dataframe(
        show_stats,
        use_container_width=True
    )

    algorithm = st.selectbox(
        "Выберите алгоритм для визуализации",
        ["KMeans", "DBSCAN"]
    )

    fig = create_cluster_plot(
        clustered_df,
        algorithm=algorithm
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Примеры отзывов из выбранного кластера")

    if algorithm == "KMeans":
        cluster_column = "kmeans_cluster"
    else:
        cluster_column = "dbscan_cluster"

    cluster_ids = sorted(clustered_df[cluster_column].unique().tolist())

    selected_cluster = st.selectbox(
        "Кластер",
        cluster_ids
    )

    examples = clustered_df[clustered_df[cluster_column] == selected_cluster].head(10)

    st.dataframe(
        examples[["product", "rating", "sentiment_label", "text"]],
        use_container_width=True
    )


def show_privacy_page(df, user_id):
    """
    Страница защиты персональных данных.
    """

    log_action(user_id, "open_page", "Защита данных")

    st.title("Защита персональных данных")

    st.write(
        """
        Здесь показана деидентификация данных:
        имя пользователя и телефон маскируются перед отображением.
        """
    )

    protected_df = create_protected_dataframe(df)

    st.subheader("Исходные данные")

    st.dataframe(
        df[["user_id", "user_name", "phone", "product", "rating"]].head(10),
        use_container_width=True
    )

    st.subheader("После маскировки")

    st.dataframe(
        protected_df[["user_id", "user_name", "phone", "product", "rating"]].head(10),
        use_container_width=True
    )


def show_logs_page(user_id):
    """
    Страница просмотра последних записей app.log.
    Доступна только администратору.
    """

    if not st.session_state.get("is_admin", False):
        log_action(user_id, "open_page_denied", "Попытка доступа к логам без прав администратора")
        st.error("Доступ запрещён. Страница доступна только администратору.")
        return

    log_action(user_id, "open_page", "Логи приложения")

    st.title("Логи приложения")

    st.write(
        """
        Ниже отображаются последние 50 записей из файла app.log.
        Эта страница предназначена только для разработчика или администратора.
        """
    )

    logs = read_last_logs(lines_count=50)

    if len(logs) == 0:
        st.warning("Файл app.log пока пустой или не создан.")
        return

    st.code(
        "".join(logs),
        language="text"
    )


def main():
    """
    Основная функция Streamlit-приложения.
    """

    init_database_from_csv()

    init_admin_state()

    user_id = init_user()

    page = show_sidebar()

    df = load_data_cached()

    if len(df) == 0:
        st.error("В базе данных нет отзывов.")
        return

    model = train_model_cached(df)

    if page == "Главная":
        show_main_page(df, user_id)

    elif page == "Добавить отзыв":
        show_add_review_page(df, model, user_id)

    elif page == "Рекомендации":
        show_recommendations_page(df, user_id)

    elif page == "Аналитика":
        show_analytics_page(df, user_id)

    elif page == "Статистика товаров":
        show_product_statistics_page(df, user_id)

    elif page == "Сравнение моделей":
        show_model_comparison_page(df, user_id)

    elif page == "Расширенное сравнение моделей":
        show_advanced_model_comparison_page(df, user_id)

    elif page == "Кластеры отзывов":
        show_clustering_page(df, user_id)

    elif page == "Защита данных":
        show_privacy_page(df, user_id)

    elif page == "Логи приложения":
        show_logs_page(user_id)


if __name__ == "__main__":
    main()