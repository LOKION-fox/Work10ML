def mask_phone(phone):
    """
    Маскирует номер телефона.
    """

    phone = str(phone)

    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]

    return phone


def mask_name(name):
    """
    Маскирует имя пользователя.
    """

    name = str(name)

    if len(name) == 0:
        return name

    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)

    return name[:2] + "*" * (len(name) - 4) + name[-2:]


def create_protected_dataframe(df):
    """
    Создаёт копию датафрейма с замаскированными персональными данными.
    """

    protected_df = df.copy()

    if "user_name" in protected_df.columns:
        protected_df["user_name"] = protected_df["user_name"].apply(mask_name)

    if "phone" in protected_df.columns:
        protected_df["phone"] = protected_df["phone"].apply(mask_phone)

    return protected_df