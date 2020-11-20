import requests
import pandas as pd
from bs4 import BeautifulSoup




def get_month(month: str) -> pd.DataFrame:
    """
    :param month: monthst oget datafrom basketball-reference
    :return: pandas data frame of values
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_2020_games-{month}.html"
    content = requests.get(url).content
    soup = BeautifulSoup(content)
    tables = soup.findAll("table")
    df = pd.read_html(tables[0].decode(),flavor="bs4")[0]

    return df

def process_month(month_df:pd.DataFrame) -> pd.DataFrame:
    """

    :param month_df: dataframe for month scraped from bb ref
    :return: processed dataframe where 'playoff row' is replaced and playoff tag is added to games
    """
    is_playoffs = month_df.index[month_df["Date"] == "Playoffs"].values.tolist()
    if is_playoffs:
        playoff_row = is_playoffs[0]
        reg_season = month_df.iloc[0:playoff_row]
        reg_season["game_type"] = "Regular Season"
        playoffs = month_df.iloc[playoff_row+1:]
        playoffs["game_type"] = "Playoffs"
        month_df = pd.concat([reg_season,playoffs])

    return month_df

def get_all_months() -> pd.DataFrame:
    """
    get all months data for scores
    :return:
    """
    processed_month_dfs = []
    season_months = ["october", "november", "december", "january", "february", "march", "july", "august", "september"]
    for month in season_months:
        month_df = get_month(month)
        processed_month_df = process_month(month_df)
        processed_month_dfs.append(processed_month_df)
    return pd.concat(processed_month_dfs)


if __name__ == "__main__":
    full_df = get_all_months()
    full_df.to_csv("../sandbox/data.csv",index=False)


