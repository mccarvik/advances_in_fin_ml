import sys
import pandas as pd
sys.path.append("/home/ubuntu/workspace/ml_dev_work")
from utils.db_utils import DBHelper


def get_google():
    df = pd.read_csv("/home/ubuntu/workspace/advances_in_fin_ml/data/Google.csv")
    df.index = pd.DatetimeIndex(df['Date'].values)
    close = df["Close"]
    return close


def get_google_all():
    df = pd.read_csv("/home/ubuntu/workspace/advances_in_fin_ml/data/Google.csv")
    return df


def get_tick(tick):
    """
    Will grab data from the db for the given parameters
    """
    with DBHelper() as dbh:
        dbh.connect()
        where_clause = 'tick = "{}"'.format(tick)
        px_df = dbh.select('eod_px', where=where_clause)
        px_df.index = pd.DatetimeIndex(px_df['date'].values)
        close = px_df["px"]
        return close