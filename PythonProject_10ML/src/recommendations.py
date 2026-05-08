import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_products_for_user(df, user_id, top_n=3):
    """
    Персонализированные рекомендации на основе user-based collaborative filtering.
    Система ищет похожих пользователей по истории оценок.
    """

    if len(df) == 0:
        return []

    ratings_df = df[["user_id", "product", "rating"]].copy()

    user_product_matrix = ratings_df.pivot_table(
        index="user_id",
        columns="product",
        values="rating",
        aggfunc="mean"
    )

    all_products = list(user_product_matrix.columns)

    if user_id not in user_product_matrix.index:
        return get_popular_recommendations(df, top_n)

    filled_matrix = user_product_matrix.fillna(0)

    similarity_matrix = cosine_similarity(filled_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=filled_matrix.index,
        columns=filled_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(ascending=False)

    similar_users = similar_users.drop(index=user_id)

    current_user_ratings = user_product_matrix.loc[user_id]

    unrated_products = current_user_ratings[current_user_ratings.isna()].index.tolist()

    recommendations = []

    for product in unrated_products:
        weighted_sum = 0
        similarity_sum = 0

        for other_user, similarity in similar_users.items():
            other_rating = user_product_matrix.loc[other_user, product]

            if pd.notna(other_rating) and similarity > 0:
                weighted_sum += similarity * other_rating
                similarity_sum += similarity

        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum

            recommendations.append(
                {
                    "product": product,
                    "predicted_rating": predicted_rating
                }
            )

    recommendations = sorted(
        recommendations,
        key=lambda item: item["predicted_rating"],
        reverse=True
    )

    if len(recommendations) == 0:
        return get_popular_recommendations(df, top_n)

    return recommendations[:top_n]


def get_popular_recommendations(df, top_n=3):
    """
    Рекомендации по среднему рейтингу.
    Используются, если у пользователя ещё нет истории оценок.
    """

    popular = df.groupby("product").agg(
        predicted_rating=("rating", "mean"),
        reviews_count=("rating", "count")
    ).reset_index()

    popular = popular.sort_values(
        by=["predicted_rating", "reviews_count"],
        ascending=False
    )

    recommendations = []

    for _, row in popular.head(top_n).iterrows():
        recommendations.append(
            {
                "product": row["product"],
                "predicted_rating": row["predicted_rating"]
            }
        )

    return recommendations