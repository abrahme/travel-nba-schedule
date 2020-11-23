import geocoder
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from math import radians, cos, sin, asin, sqrt


def get_month(month: str) -> pd.DataFrame:
    """
    :param month: monthst oget datafrom basketball-reference
    :return: pandas data frame of values
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_2020_games-{month}.html"
    content = requests.get(url).content
    soup = BeautifulSoup(content)
    tables = soup.findAll("table")
    df = pd.read_html(tables[0].decode(), flavor="bs4")[0]
    df.fillna({"Attend.": 0}, inplace=True)
    return df


def split_playoff(month_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param month_df: dataframe for month scraped from bb ref
    :return: processed dataframe where 'playoff row' is replaced and playoff tag is added to games
    """
    is_playoffs = month_df.index[month_df["Date"] == "Playoffs"].values.tolist()
    if is_playoffs:
        playoff_row = is_playoffs[0]
        reg_season = month_df.iloc[0:playoff_row]
        reg_season["game_type"] = "Regular Season"
        playoffs = month_df.iloc[playoff_row + 1:]
        playoffs["game_type"] = "Playoffs"
        month_df = pd.concat([reg_season, playoffs])
    else:
        month_df["game_type"] = "Regular Season"
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
        processed_month_df = split_playoff(month_df)
        processed_month_dfs.append(processed_month_df)
    return pd.concat(processed_month_dfs)


def haversine_distance(location_i: list, location_j: list) -> float:
    """
    calculates haversine distance between 2 points
    :param location_i: [lat,long]
    :param location_j: [lat,long]
    :return: distance
    """
    lat_1, lon_1 = location_i[0], location_i[1]
    lat_2, lon_2 = location_j[0], location_j[1]
    # convert decimal to radians
    lon_1, lat_1, lon_2, lat_2 = map(radians, [lon_1, lat_1, lon_2, lat_2])
    d_lon = lon_2 - lon_1
    d_lat = lat_2 - lat_1
    a = sin(d_lat / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def calculate_rest(season_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param season_df: calculate the rest each team has since last game
    :return:
    """
    season_df.rename(columns={"PTS": "Visitor PTS", "PTS.1": "Home PTS"}, inplace=True)
    season_df["Date_Clean"] = season_df["Date"].apply(lambda x: datetime.strptime(x.split(",", 1)[-1], ' %b %d, %Y'))
    teams = set(season_df["Visitor/Neutral"].values.tolist() + season_df["Home/Neutral"].values.tolist())
    days_played = {team: pd.NaT for team in teams}
    visitor_rest = []
    home_rest = []
    for index, row in season_df.sort_values(by="Date_Clean").iterrows():
        visitor, home = row["Visitor/Neutral"], row["Home/Neutral"]
        if type(days_played[visitor]) is pd.NaT:
            visitor_rest.append(pd.NaT)
        else:
            visitor_rest.append(row["Date_Clean"] - days_played[visitor])
        days_played[visitor] = row["Date_Clean"]
        if type(days_played[home]) is pd.NaT:
            home_rest.append(pd.NaT)
        else:
            home_rest.append(row["Date_Clean"] - days_played[home])
        days_played[home] = row["Date_Clean"]
    season_df = season_df.sort_values(by="Date_Clean")
    season_df["Home Rest"] = home_rest
    season_df["Home Rest"].fillna(pd.Timedelta("10 days"), inplace=True)
    season_df["Home Rest"] = season_df["Home Rest"].apply(
        lambda x: x.days if x <= pd.Timedelta("10 days") else 10)
    season_df["Visitor Rest"] = visitor_rest
    season_df["Visitor Rest"].fillna(pd.Timedelta("10 days"), inplace=True)
    season_df["Visitor Rest"] = season_df["Visitor Rest"].apply(
        lambda x: x.days if x <= pd.Timedelta("10 days") else 10)
    season_df.drop(columns=["Date"], inplace=True)
    return season_df


def calculate_travel(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates travel from one destination to the next
    :param df: dataframe
    :return: 
    """
    teams = set(df["Visitor/Neutral"].values.tolist() + df["Home/Neutral"].values.tolist())
    locations_played = {team: None for team in teams}
    visitor_rest = []
    home_rest = []
    for index, row in df.sort_values(by="Date_Clean").iterrows():
        visitor, home = row["Visitor/Neutral"], row["Home/Neutral"]
        if locations_played[visitor] is None:
            visitor_rest.append(haversine_distance(row["location_game"],
                                                   row["location_i"] if visitor == row["team_i"] else row[
                                                       "location_j"]))
        else:
            visitor_rest.append(haversine_distance(row["location_game"], locations_played[visitor]))
        locations_played[visitor] = row["location_game"]
        if locations_played[home] is None:
            home_rest.append(0)
        else:
            home_rest.append(haversine_distance(row["location_game"], locations_played[home]))
        locations_played[home] = row["location_game"]

    travel_df = df.sort_values(by="Date_Clean")
    travel_df["Visitor Travel"] = visitor_rest
    travel_df["Home Travel"] = home_rest
    return travel_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: calculate necessary features for model
    :return:
    """
    df["Home PTS"] = df["Home PTS"].apply(lambda x: int(x))
    df["Visitor PTS"] = df["Visitor PTS"].apply(lambda x: int(x))
    df["team_i"] = df.apply(
        lambda x: x["Home/Neutral"] if x["Home PTS"] > x["Visitor PTS"] else x["Visitor/Neutral"], axis=1)
    df["team_j"] = df.apply(
        lambda x: x["Home/Neutral"] if x["Home PTS"] < x["Visitor PTS"] else x["Visitor/Neutral"], axis=1)
    df["margin_ij"] = abs(df["Visitor PTS"] - df["Home PTS"])
    df["rest_i"] = df.apply(
        lambda x: x["Home Rest"] if x["Home PTS"] > x["Visitor PTS"] else x["Visitor Rest"],
        axis=1)
    df["rest_j"] = df.apply(
        lambda x: x["Home Rest"] if x["Home PTS"] < x["Visitor PTS"] else x["Visitor Rest"],
        axis=1)
    # bubble games
    df["home_i"] = df.apply(lambda x: 0 if pd.isnull(x["Attend."]) or (x["Home/Neutral"] != x["team_i"]) else 1,
                            axis=1)
    df["home_j"] = df.apply(lambda x: 0 if pd.isnull(x["Attend."]) or (x["Home/Neutral"] != x["team_j"]) else 1,
                            axis=1)
    return df


def get_teams() -> pd.DataFrame:
    """
    gets team and location
    :return:
    """
    team_json = requests.get("https://www.balldontlie.io/api/v1/teams").json()
    team_df = pd.DataFrame(team_json["data"])
    team_df = team_df.replace("Utah", "Salt Lake City").replace("Golden State", "San Francisco").replace("LA",
                                                                                                         "Los Angeles").replace(
        "LA Clippers", "Los Angeles Clippers")
    team_df["location"] = team_df["city"].apply(lambda x: geocoder.arcgis(x).latlng)
    return team_df


def filter_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    filter out bubble games
    :param df:
    :return:
    """
    return df[(df["game_type"] == "Regular Season") & (df["Attend."].astype(float) > 0)]


if __name__ == "__main__":
    full_df = get_all_months()
    full_df = calculate_rest(full_df)
    full_df = create_features(full_df)
    full_df = filter_games(full_df)
    team_df = get_teams()
    team_df.to_csv("../sandbox/teams.csv", index=False)

    full_df = full_df.merge(team_df[["id", "full_name", "location"]], how="inner",
                            left_on="team_i", right_on="full_name").drop(columns=["full_name"]).rename(
        columns={"location": "location_i", "id": "id_i"})
    full_df = full_df.merge(team_df[["id", "full_name", "location"]], how="inner",
                            left_on="team_j", right_on="full_name").drop(
        columns=["full_name"]).rename(
        columns={"location": "location_j", "id": "id_j"})
    full_df["location_game"] = full_df.apply(lambda x: x["location_i"] if x["home_i"] == 1 else x["location_j"], axis=1)
    full_df = calculate_travel(full_df)
    full_df["travel_i"] = full_df.apply(
        lambda x: x["Home Travel"] if x["Home PTS"] > x["Visitor PTS"] else x["Visitor Travel"],
        axis=1)
    full_df["travel_j"] = full_df.apply(
        lambda x: x["Home Travel"] if x["Home PTS"] < x["Visitor PTS"] else x["Visitor Travel"],
        axis=1)
    full_df.to_csv("../sandbox/data.csv", index=False)
