import logging


LOG_PATH = "app.log"


def setup_logger():
    """
    Настраивает логирование действий пользователя в файл app.log.
    """

    logger = logging.getLogger("review_app_logger")

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(
            LOG_PATH,
            encoding="utf-8"
        )

        formatter = logging.Formatter(
            "%(asctime)s | user_id=%(user_id)s | action=%(action)s | result=%(result)s"
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def log_action(user_id, action, result):
    """
    Записывает действие пользователя в лог.

    Формат записи:
    время | user_id | действие | результат
    """

    logger = setup_logger()

    logger.info(
        "",
        extra={
            "user_id": user_id,
            "action": action,
            "result": result
        }
    )


def read_last_logs(lines_count=50):
    """
    Возвращает последние строки из app.log.
    """

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return lines[-lines_count:]

    except FileNotFoundError:
        return []