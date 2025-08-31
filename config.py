import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

isTestnet = True
trade_interval = os.getenv("BINANCE_API_INTERVAL")
