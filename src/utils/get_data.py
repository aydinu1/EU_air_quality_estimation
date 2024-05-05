import json
import os
import time
import datetime
import pandas as pd
import requests
from geopy.geocoders import Nominatim


def get_capital_coordinates_eu():
    if not os.path.exists(r"list_of_cities.json"):
        print("list_of_cities.json file cannot be found, downloading...")
        eu_capitals = {}

        # List of EU member countries and their capitals
        eu_countries_capitals = {
            "Austria": "Vienna",
            "Belgium": "Brussels",
            "Bulgaria": "Sofia",
            "Croatia": "Zagreb",
            "Cyprus": "Nicosia",
            "Czech Republic": "Prague",
            "Denmark": "Copenhagen",
            "Estonia": "Tallinn",
            "Finland": "Helsinki",
            "France": "Paris",
            "Germany": "Berlin",
            "Greece": "Athens",
            "Hungary": "Budapest",
            "Ireland": "Dublin",
            "Italy": "Rome",
            "Latvia": "Riga",
            "Lithuania": "Vilnius",
            "Luxembourg": "Luxembourg",
            "Malta": "Valletta",
            "Netherlands": "Amsterdam",
            "Poland": "Warsaw",
            "Portugal": "Lisbon",
            "Romania": "Bucharest",
            "Slovakia": "Bratislava",
            "Slovenia": "Ljubljana",
            "Spain": "Madrid",
            "Sweden": "Stockholm",
        }

        geolocator = Nominatim(user_agent="eu_capitals_app")

        for country, capital in eu_countries_capitals.items():
            location = geolocator.geocode(capital, exactly_one=True)
            if location:
                eu_capitals[capital] = {'longitude': location.longitude, 'latitude': location.latitude}
            else:
                print(f"Coordinates not found for {capital}, {country}")

        with open("list_of_cities.json", "w") as fp:
            json.dump(eu_capitals, fp)

    else:
        with open('list_of_cities.json', 'r') as fp:
            eu_capitals = json.load(fp)

    return eu_capitals


def get_pollution_data_api(city_name: str, latitude: str, longitude: float,
                           start_date: str, end_date: str):
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': ["pm2_5"],
        'start_date': start_date,
        'end_date': end_date,
        'timezone': "Europe/Berlin"
    }

    base_url = r"https://air-quality-api.open-meteo.com/v1/air-quality"
    trials = 0
    while True:
        try:
            response = requests.get(base_url, params=params)
            print(f"Downloading pollution data for {city_name}")
            if response.status_code == 429:
                print("rate limiting reached, waiting 10 seconds")
                trials += 1
                if trials == 10:
                    raise ConnectionAbortedError("Max trials reached, breaking")
                time.sleep(10)
            else:
                break
        except requests.exceptions.ConnectionError:
            trials += 1
            if trials == 10:
                raise ConnectionAbortedError("Max trials reached, breaking")
            time.sleep(1)
            response = requests.get(base_url, params=params)
            break
    response_json = response.json()
    res_df = pd.DataFrame(response_json["hourly"])

    res_df["time"] = pd.to_datetime(res_df["time"])
    res_df["city"] = city_name
    res_df["latitude"] = latitude
    res_df["longitude"] = longitude
    return res_df


def get_historical_weather_data_api(city_name: str, latitude: str, longitude: float,
                                    start_date: str, end_date: str):
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': ["temperature_2m", "relativehumidity_2m",
                   "precipitation", "cloudcover", "cloudcover_low",
                   "cloudcover_mid", "cloudcover_high", "windspeed_10m",
                   "winddirection_10m", "windgusts_10m"
                   ],
        'start_date': start_date,
        'end_date': end_date,
        'timezone': "Europe/Berlin"
    }
    base_url = r"https://archive-api.open-meteo.com/v1/archive"
    trials = 0
    while True:
        try:
            response = requests.get(base_url, params=params)
            print(f"Downloading historical weather data for {city_name}")
            if response.status_code == 429:
                print("rate limiting reached, waiting 10 seconds")
                trials += 1
                if trials == 10:
                    raise ConnectionAbortedError("Max trials reached, breaking")
                time.sleep(10)
            else:
                break
        except requests.exceptions.ConnectionError:
            trials += 1
            if trials == 10:
                raise ConnectionAbortedError("Max trials reached, breaking")
            time.sleep(1)
            response = requests.get(base_url, params=params)
            break
    response_json = response.json()
    res_df = pd.DataFrame(response_json["hourly"])

    res_df["time"] = pd.to_datetime(res_df["time"])
    res_df["city"] = city_name
    res_df["latitude"] = latitude
    res_df["longitude"] = longitude
    return res_df


def get_forecast_weather_data_api(city_name: str, latitude: str, longitude: float):
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': ["temperature_2m", "relativehumidity_2m",
                   "precipitation", "cloudcover", "cloudcover_low",
                   "cloudcover_mid", "cloudcover_high", "windspeed_10m",
                   "winddirection_10m", "windgusts_10m"
                   ],
        'timezone': "Europe/Berlin",
        'forecast_days': 2
    }
    base_url = r"https://api.open-meteo.com/v1/forecast"

    try:
        response = requests.get(base_url, params=params)
    except ConnectionError:
        time.sleep(1)
        response = requests.get(base_url, params=params)
    response_json = response.json()
    res_df = pd.DataFrame(response_json["hourly"])

    res_df["time"] = pd.to_datetime(res_df["time"])
    res_df["city"] = city_name
    res_df["latitude"] = latitude
    res_df["longitude"] = longitude
    return res_df


def get_historical_data(start_date=None, end_date=None, resample=0):
    eu_capitals_dict = get_capital_coordinates_eu()

    df_historical = pd.DataFrame()
    if not start_date:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime(format="%Y-%m-%d")
    if not end_date:
        end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(format="%Y-%m-%d")

    if start_date >= end_date:
        raise ValueError("End date must be later than start date")

    for city, coordinates in eu_capitals_dict.items():
        df_pollution = get_pollution_data_api(city,
                                              longitude=coordinates["longitude"],
                                              latitude=coordinates["latitude"],
                                              start_date=start_date,
                                              end_date=end_date)

        df_weather = get_historical_weather_data_api(city,
                                                     longitude=coordinates["longitude"],
                                                     latitude=coordinates["latitude"],
                                                     start_date=start_date,
                                                     end_date=end_date)

        df_merged = df_pollution.merge(df_weather, on=["time", "city", "latitude", "longitude"])

        df_historical = pd.concat([df_merged, df_historical])
        # downsample data for every 6 hours or not
        resample = 0
        if resample:
            df_historical["time"] = pd.to_datetime(df_historical["time"])
            df_historical = df_historical.set_index("time").groupby("city").resample("4H").first()
            df_historical = df_historical.reset_index(level="time",
                                                      names=["time", "city_drop"]).drop(["city_drop"], axis=1)
    return df_historical


def get_forecast_data():
    eu_capitals_dict = get_capital_coordinates_eu()
    df_weather_forecast = pd.DataFrame()
    for city, coordinates in eu_capitals_dict.items():
        df_forecast = get_forecast_weather_data_api(city,
                                                    longitude=coordinates["longitude"],
                                                    latitude=coordinates["latitude"])
        df_weather_forecast = pd.concat([df_weather_forecast, df_forecast])

    time_filter = datetime.datetime.now() + datetime.timedelta(hours=24)
    df_weather_forecast = df_weather_forecast[df_weather_forecast["time"] <= time_filter]

    return df_weather_forecast


if __name__ == "__main__":
    from plotting import plot_air_quality_map
    df_historical = get_historical_data()
    df_weather_forecast = get_forecast_data()

    fig = plot_air_quality_map(df_historical.sort_values(by="time", ascending=True)
                               .groupby(by="city").last()
                               .reset_index())
    fig.show()

