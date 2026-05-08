import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


def prepare_binary_dataset(df):
    """
    Подготавливает данные для бинарной классификации тональности.

    Используются только позитивные и негативные отзывы:
    sentiment = 1 — позитивный отзыв
    sentiment = 0 — негативный отзыв.

    Нейтральные отзывы исключаются из обучения моделей классификации.
    """

    work_df = df[df["sentiment"].notna()].copy()

    work_df["sentiment"] = work_df["sentiment"].astype(int)

    clean_df = pd.DataFrame(
        {
            "text": work_df["text"].astype(str),
            "sentiment": work_df["sentiment"]
        }
    )

    clean_df = clean_df.drop_duplicates(subset=["text"])

    X = clean_df["text"].astype(str)
    y = clean_df["sentiment"].astype(int)

    return X, y


def get_five_models():
    """
    Возвращает модели для расширенного сравнения.

    Используются:
    1. LogisticRegression
    2. RandomForest
    3. MLPClassifier
    4. XGBoost, если библиотека установлена
    5. LightGBM, если библиотека установлена
    """

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            random_state=42
        ),

        "MLPClassifier": MLPClassifier(
            max_iter=500,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            random_state=42,
            verbose=-1
        )

    return models


def get_param_grids():
    """
    Возвращает сетки гиперпараметров для GridSearchCV.

    Для каждой модели указано минимум три гиперпараметра.
    """

    param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
            "classifier__class_weight": [None, "balanced"]
        },

        "RandomForest": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [4, 8, None],
            "classifier__min_samples_split": [2, 5]
        },

        "MLPClassifier": {
            "classifier__hidden_layer_sizes": [(30,), (50,), (50, 20)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.0001, 0.001]
        },

        "XGBoost": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.05, 0.1]
        },

        "LightGBM": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [-1, 3],
            "classifier__learning_rate": [0.05, 0.1]
        }
    }

    return param_grids


def calculate_metrics(y_test, y_pred):
    """
    Рассчитывает метрики классификации.
    """

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics


def create_pipeline(classifier):
    """
    Создаёт пайплайн:
    TF-IDF-векторизация текста + классификатор.
    """

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=700,
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9
                )
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    return pipeline


def compare_models_before_after_tuning(df):
    """
    Сравнивает модели до и после подбора гиперпараметров.

    Выполняется:
    1. Разделение данных на train/test.
    2. Обучение базовой версии каждой модели.
    3. Расчёт Accuracy, Precision, Recall, F1 до подбора.
    4. GridSearchCV для каждой модели.
    5. Расчёт Accuracy, Precision, Recall, F1 после подбора.
    6. Подготовка данных для ROC-кривых.

    Возвращает:
    results_df — сводная таблица метрик;
    roc_data — данные для ROC-графика;
    available_models_count — количество реально доступных моделей.
    """

    X, y = prepare_binary_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    models = get_five_models()
    param_grids = get_param_grids()

    rows = []
    roc_data = {}

    for model_name, classifier in models.items():
        pipeline = create_pipeline(classifier)

        pipeline.fit(X_train, y_train)

        y_pred_before = pipeline.predict(X_test)

        before_metrics = calculate_metrics(y_test, y_pred_before)

        rows.append(
            {
                "Модель": model_name,
                "Этап": "До подбора",
                "Accuracy": before_metrics["Accuracy"],
                "Precision": before_metrics["Precision"],
                "Recall": before_metrics["Recall"],
                "F1": before_metrics["F1"]
            }
        )

        if model_name in param_grids:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring="f1",
                cv=3,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_pred_after = best_model.predict(X_test)

            after_metrics = calculate_metrics(y_test, y_pred_after)

            rows.append(
                {
                    "Модель": model_name,
                    "Этап": "После подбора",
                    "Accuracy": after_metrics["Accuracy"],
                    "Precision": after_metrics["Precision"],
                    "Recall": after_metrics["Recall"],
                    "F1": after_metrics["F1"]
                }
            )

            if hasattr(best_model, "predict_proba"):
                y_score = best_model.predict_proba(X_test)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_score)

                roc_auc = auc(fpr, tpr)

                roc_data[model_name] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": roc_auc
                }

    results_df = pd.DataFrame(rows)

    results_df = results_df.sort_values(
        by=["F1", "Accuracy"],
        ascending=False
    )

    available_models_count = len(models)

    return results_df, roc_data, available_models_count