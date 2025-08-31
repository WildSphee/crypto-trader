import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

is_test_net = False if os.getenv("BINANCE_API_INTERVAL") == "False" else True
trade_interval = os.getenv("BINANCE_API_INTERVAL", "1h")


if not is_test_net:
    print("Running on Mainnet!!!!")
