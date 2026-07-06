import logging

logger=logging.getLogger()
logger.setLevel(level=logging.DEBUG)

if not logger.handlers:
    filehandler=logging.FileHandler('error.txt',mode='a')
    filehandler.setLevel(logging.ERROR)

    console_handler=logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    filehandler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(console_handler)


