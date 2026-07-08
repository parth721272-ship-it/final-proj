import pandas as pd
import numpy as np
import random

file_name = "covid19_global_data.csv"

try:
    np.random.seed(456)
    random.seed(456)

    n_rows = 18000

    record_ids = [f"COVID_{str(i).zfill(6)}" for i in range(1, n_rows + 1)]
    countries = [
        "USA",
        "India",
        "Brazil",
        "UK",
        "France",
        "Germany",
        "Italy",
        "Spain",
        "Russia",
        "China",
        "Japan",
        "South Korea",
        "Canada",
        "Australia",
        "Mexico",
        "South Africa",
        "Turkey",
        "Iran",
    ]

    usa_states = [
        "California",
        "Texas",
        "Florida",
        "New York",
        "Illinois",
        "Pennsylvania",
        "Ohio",
        "Georgia",
        "North Carolina",
        "Michigan",
    ]
    india_states = [
        "Maharashtra",
        "Tamil Nadu",
        "Kerala",
        "Karnataka",
        "Andhra Pradesh",
        "Uttar Pradesh",
        "Delhi",
        "West Bengal",
        "Rajasthan",
        "Gujarat",
    ]
    brazil_states = [
        "Sao Paulo",
        "Rio de Janeiro",
        "Minas Gerais",
        "Bahia",
        "Parana",
        "Rio Grande do Sul",
        "Pernambuco",
        "Ceara",
        "Santa Catarina",
        "Goias",
    ]
    uk_regions = ["England", "Scotland", "Wales", "Northern Ireland"]
    france_regions = [
        "Ile-de-France",
        "Auvergne-Rhone-Alpes",
        "Provence-Alpes-Cote d'Azur",
        "Hauts-de-France",
        "Grand Est",
        "Occitanie",
        "Pays de la Loire",
        "Brittany",
        "Normandy",
        "Nouvelle-Aquitaine",
    ]
    germany_states = [
        "North Rhine-Westphalia",
        "Bavaria",
        "Baden-Wurttemberg",
        "Lower Saxony",
        "Hesse",
        "Saxony",
        "Rhineland-Palatinate",
        "Berlin",
        "Schleswig-Holstein",
        "Hamburg",
    ]
    italy_regions = [
        "Lombardy",
        "Lazio",
        "Campania",
        "Veneto",
        "Emilia-Romagna",
        "Piedmont",
        "Sicily",
        "Apulia",
        "Tuscany",
        "Calabria",
    ]
    spain_regions = [
        "Madrid",
        "Catalonia",
        "Andalusia",
        "Valencia",
        "Castile and Leon",
        "Basque Country",
        "Castilla-La Mancha",
        "Galicia",
        "Aragon",
        "Murcia",
    ]
    russia_regions = [
        "Moscow",
        "Saint Petersburg",
        "Moscow Oblast",
        "Krasnodar Krai",
        "Sverdlovsk Oblast",
        "Rostov Oblast",
        "Republic of Bashkortostan",
        "Republic of Tatarstan",
        "Chelyabinsk Oblast",
        "Novosibirsk Oblast",
    ]
    china_provinces = [
        "Hubei",
        "Guangdong",
        "Henan",
        "Zhejiang",
        "Hunan",
        "Anhui",
        "Jiangxi",
        "Jiangsu",
        "Chongqing",
        "Sichuan",
    ]
    japan_prefectures = [
        "Tokyo",
        "Osaka",
        "Kanagawa",
        "Aichi",
        "Saitama",
        "Chiba",
        "Hyogo",
        "Hokkaido",
        "Fukuoka",
        "Kyoto",
    ]
    sk_provinces = [
        "Seoul",
        "Busan",
        "Incheon",
        "Daegu",
        "Daejeon",
        "Gwangju",
        "Ulsan",
        "Gyeonggi",
        "Gangwon",
        "Chungcheong",
    ]
    canada_provinces = [
        "Ontario",
        "Quebec",
        "British Columbia",
        "Alberta",
        "Manitoba",
        "Saskatchewan",
        "Nova Scotia",
        "New Brunswick",
        "Newfoundland and Labrador",
        "Prince Edward Island",
    ]
    australia_states = [
        "New South Wales",
        "Victoria",
        "Queensland",
        "Western Australia",
        "South Australia",
        "Tasmania",
        "Australian Capital Territory",
        "Northern Territory",
    ]
    mexico_states = [
        "Mexico City",
        "State of Mexico",
        "Jalisco",
        "Nuevo Leon",
        "Guanajuato",
        "Puebla",
        "Veracruz",
        "Baja California",
        "Chihuahua",
        "Sonora",
    ]
    sa_provinces = [
        "Gauteng",
        "Western Cape",
        "KwaZulu-Natal",
        "Eastern Cape",
        "Free State",
        "Mpumalanga",
        "North West",
        "Limpopo",
        "Northern Cape",
    ]
    turkey_provinces = [
        "Istanbul",
        "Ankara",
        "Izmir",
        "Bursa",
        "Antalya",
        "Konya",
        "Adana",
        "Gaziantep",
        "Kocaeli",
        "Mersin",
    ]
    iran_provinces = [
        "Tehran",
        "Isfahan",
        "Razavi Khorasan",
        "Fars",
        "East Azerbaijan",
        "Mazandaran",
        "Alborz",
        "Kerman",
        "Gilan",
        "Golestan",
    ]

    country_states = {
        "USA": usa_states,
        "India": india_states,
        "Brazil": brazil_states,
        "UK": uk_regions,
        "France": france_regions,
        "Germany": germany_states,
        "Italy": italy_regions,
        "Spain": spain_regions,
        "Russia": russia_regions,
        "China": china_provinces,
        "Japan": japan_prefectures,
        "South Korea": sk_provinces,
        "Canada": canada_provinces,
        "Australia": australia_states,
        "Mexico": mexico_states,
        "South Africa": sa_provinces,
        "Turkey": turkey_provinces,
        "Iran": iran_provinces,
    }

    selected_countries = []
    selected_states = []
    for _ in range(n_rows):
        country = random.choice(countries)
        state = random.choice(country_states[country])
        selected_countries.append(country)
        selected_states.append(state)

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    years = [2020, 2021, 2022, 2023]
    dates = []
    for _ in range(n_rows):
        year = random.choice(years)
        month = random.choice(months)
        day = random.randint(1, 28)
        dates.append(f"{year}-{month:02d}-{day:02d}")

    confirmed_cases = np.random.poisson(500, n_rows)
    confirmed_cases = np.clip(confirmed_cases, 0, 10000)

    deaths = np.round(confirmed_cases * np.random.uniform(0.01, 0.15, n_rows))
    deaths = np.clip(deaths, 0, 2000)

    recovered = np.round(confirmed_cases * np.random.uniform(0.6, 0.95, n_rows))
    recovered = np.clip(recovered, 0, confirmed_cases)

    active_cases = confirmed_cases - deaths - recovered
    active_cases = np.clip(active_cases, 0, confirmed_cases)

    tests_conducted = np.random.poisson(2000, n_rows)
    tests_conducted = np.clip(tests_conducted, confirmed_cases, 50000)

    vaccination_rate = np.random.uniform(10, 95, n_rows)
    vaccination_rate = np.round(vaccination_rate, 1)

    hospitalization_rate = np.random.uniform(2, 20, n_rows)
    hospitalization_rate = np.round(hospitalization_rate, 1)

    icu_cases = np.round(active_cases * hospitalization_rate * 0.1 / 100)
    icu_cases = np.clip(icu_cases, 0, 500)

    df = pd.DataFrame(
        {
            "Record_ID": record_ids,
            "Country": selected_countries,
            "State_Region": selected_states,
            "Date": dates,
            "Confirmed_Cases": confirmed_cases,
            "Deaths": deaths,
            "Recovered": recovered,
            "Active_Cases": active_cases,
            "Tests_Conducted": tests_conducted,
            "Vaccination_Rate": vaccination_rate,
            "Hospitalization_Rate": hospitalization_rate,
            "ICU_Cases": icu_cases,
        }
    )

    duplicate_count = random.randint(100, 200)
    duplicate_indices = random.sample(range(n_rows), duplicate_count)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    numeric_columns = [
        "Confirmed_Cases",
        "Deaths",
        "Recovered",
        "Active_Cases",
        "Tests_Conducted",
        "Vaccination_Rate",
        "Hospitalization_Rate",
        "ICU_Cases",
    ]
    empty_count = random.randint(1000, 3000)
    empty_indices = random.sample(range(len(df)), empty_count)
    for idx in empty_indices:
        col = random.choice(numeric_columns)
        df.at[idx, col] = np.nan

    df = df.sample(frac=1, random_state=456).reset_index(drop=True)
except BaseException as e:
    print(f"An error occurred: {e}")
else:
    df.to_csv(file_name, index=False)
    print(f"Dataset '{file_name}' generated with {len(df)} records.")
finally:
    print("Data generation process completed.")
