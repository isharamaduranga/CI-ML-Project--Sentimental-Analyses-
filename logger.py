import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename='logs.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
