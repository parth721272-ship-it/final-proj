import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class CovidDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # 1. Generate Data (runs data_generate.py using os)
    def generate_data(self):
        if os.path.exists("data_generate.py") and not os.path.exists(
            "covid19_global_data.csv"
        ):
            os.system("python data_generate.py")
            print("✅ Data generated successfully.")
        elif not os.path.exists("covid19_global_data.csv"):
            print("❌ data already exists!")
        elif not os.path.exists("data_generate.py"):
            print("❌ data_generate.py not found!")

    # 2. Load Data
    def load_data(self):
        if not os.path.exists(self.file_path):
            print("⚠️ Data file not found. Generating new data...")
            self.generate_data()

        self.df = pd.read_csv(self.file_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        print("✅ Data loaded successfully.")
        print("Shape:", self.df.shape)

    # 3. Basic Info
    def basic_info(self):
        print("\n--- Basic Info ---")
        print(self.df.info())
        print("\nMissing Values:\n", self.df.isna().sum())
        print("\nMissing Values (%):\n", (self.df.isna().sum() / len(self.df)) * 100)

    # 4. Handle Missing Values
    def handle_missing_values(self):
        numeric_cols = [
            "Confirmed_Cases",
            "Deaths",
            "Recovered",
            "Active_Cases",
            "Tests_Conducted",
            "Vaccination_Rate",
            "Hospitalization_Rate",
            "ICU_Cases",
        ]
        while True:
            print("\n--- Handling Missing Values ---")
            print("1. Fill numeric columns with a aggregate function (mean/median)")
            print("2. Drop rows with missing values")
            print("0. exit")
            choice = int(input("Enter your choice: "))

            match choice:
                case 1:
                    func = (
                        input("Enter aggregate function (mean/median): ")
                        .strip()
                        .lower()
                    )
                    if func in ["mean", "median"]:
                        for col in numeric_cols:
                            if func == "mean":
                                self.df[col] = self.df[col].fillna(self.df[col].mean())
                            else:
                                self.df[col] = self.df[col].fillna(
                                    self.df[col].median()
                                )
                        print("✅ Missing values filled using", func)
                    else:
                        print("❌ Invalid function!")
                case 2:
                    self.df.dropna(inplace=True)
                    print("✅ Rows with missing values dropped.")
                case 0:
                    print("No changes made.")
                    print("Exiting...")
                    break
                case _:
                    print("❌ Invalid choice!")

    # 5. All Analysis
    def all_analysis(self):
        if self.df is None:
            print("⚠️ Data not loaded. Please load data first.")
            return
        df = self.df
        print("\n--- All Analysis ---")
        print("Dataset Shape:", df.shape)
        print("\nColumn Data Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isna().sum())

        numeric_cols = [
            "Confirmed_Cases",
            "Deaths",
            "Recovered",
            "Active_Cases",
            "Tests_Conducted",
            "Vaccination_Rate",
            "Hospitalization_Rate",
            "ICU_Cases",
        ]
        print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))

        missing_percent = (df.isna().sum() / len(df)) * 100
        print("\nMissing Values (%):\n", missing_percent)

        country_cases = (
            df.groupby("Country")["Confirmed_Cases"].sum().sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Confirmed Cases:\n", country_cases.head(10))

        country_deaths = (
            df.groupby("Country")["Deaths"].sum().sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Deaths:\n", country_deaths.head(10))

        country_vax = (
            df.groupby("Country")["Vaccination_Rate"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Avg Vaccination Rate:\n", country_vax.head(10))

        time_series = df.groupby("Date")[
            ["Confirmed_Cases", "Deaths", "Recovered", "Active_Cases"]
        ].sum()
        print("\nOverall Time Series (first 10 rows):\n", time_series.head(10))

        # Daily growth rate (Confirmed Cases)
        time_series["Daily_New_Cases"] = time_series["Confirmed_Cases"].diff()
        print(
            "\nDaily New Cases (first 10 rows):\n",
            time_series["Daily_New_Cases"].head(10),
        )

        corr_matrix = df[
            [
                "Confirmed_Cases",
                "Deaths",
                "Recovered",
                "Active_Cases",
                "Tests_Conducted",
                "Vaccination_Rate",
                "Hospitalization_Rate",
                "ICU_Cases",
            ]
        ].corr()
        print("\nCorrelation Matrix:\n", corr_matrix)

        icu_by_country = (
            df.groupby("Country")["ICU_Cases"].mean().sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Avg ICU Cases:\n", icu_by_country.head(10))

        hosp_rate = (
            df.groupby("Country")["Hospitalization_Rate"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Avg Hospitalization Rate:\n", hosp_rate.head(10))

        state_cases = (
            df.groupby(["Country", "State_Region"])["Confirmed_Cases"]
            .sum()
            .sort_values(ascending=False)
        )
        print("\nTop 10 States/Regions by Confirmed Cases:\n", state_cases.head(10))

        df["Death_Rate"] = df["Deaths"] / df["Confirmed_Cases"].replace(0, np.nan)
        df["Recovery_Rate"] = df["Recovered"] / df["Confirmed_Cases"].replace(0, np.nan)

        print("\nGlobal Average Death Rate:", df["Death_Rate"].mean())
        print("Global Average Recovery Rate:", df["Recovery_Rate"].mean())

        # Additional Insights
        print("\n--- Additional Insights ---")
        print("Total Confirmed Cases:", df["Confirmed_Cases"].sum())
        print("Total Deaths:", df["Deaths"].sum())
        print("Total Recovered:", df["Recovered"].sum())
        print("Total Active Cases:", df["Active_Cases"].sum())
        print("Total Tests Conducted:", df["Tests_Conducted"].sum())
        print("Average Vaccination Rate:", df["Vaccination_Rate"].mean())
        print("Average Hospitalization Rate:", df["Hospitalization_Rate"].mean())
        print("Average ICU Cases:", df["ICU_Cases"].mean())
        print("Average Death Rate:", df["Death_Rate"].mean())
        print("Average Recovery Rate:", df["Recovery_Rate"].mean())

    def all_visualizations(self):
        if self.df is None:
            print("⚠️ Data not loaded. Please load data first.")
            return
        df = self.df
        numeric_cols = [
            "Confirmed_Cases",
            "Deaths",
            "Recovered",
            "Active_Cases",
            "Tests_Conducted",
            "Vaccination_Rate",
            "Hospitalization_Rate",
            "ICU_Cases",
        ]
        for col in numeric_cols:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

        for col in numeric_cols:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

        pairs = [("Confirmed_Cases","Deaths"),
                ("Tests_Conducted","Active_Cases"),
                ("Vaccination_Rate","Confirmed_Cases")]

        for x,y in pairs:
            plt.figure(figsize=(6,4))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
            plt.title(f"{x} vs {y}")
            plt.show()

        plt.figure(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        sns.pairplot(df[numeric_cols].dropna().sample(500))  # sample for speed
        plt.show()

        time_group = df.groupby("Date")[["Confirmed_Cases","Deaths","Recovered"]].sum()

        plt.figure(figsize=(12,6))
        time_group.plot()
        plt.title("Cases, Deaths & Recoveries Over Time")
        plt.ylabel("Counts")
        plt.show()

        time_group_rolling = time_group.rolling(7).mean()
        time_group_rolling.plot(figsize=(12,6))
        plt.title("7-Day Rolling Average")
        plt.ylabel("Counts")
        plt.show()

        plt.figure(figsize=(12,6))
        sns.lineplot(data=df.groupby("Date")[["Vaccination_Rate","Confirmed_Cases"]].mean())
        plt.title("Vaccination Rate vs Confirmed Cases Over Time")
        plt.show()

        top_countries = df.groupby("Country")["Confirmed_Cases"].sum().nlargest(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_countries.values, y=top_countries.index)
        plt.title("Top 10 Countries by Confirmed Cases")
        plt.show()

        top5 = df["Country"].value_counts().head(5).index
        plt.figure(figsize=(10,6))
        sns.boxplot(data=df[df["Country"].isin(top5)],
                    x="Country", y="Hospitalization_Rate")
        plt.title("Hospitalization Rate Distribution (Top 5 Countries)")
        plt.xticks(rotation=90)
        plt.show()

        pivot = df.pivot_table(values="ICU_Cases", index="Date", columns="Country", aggfunc="mean")
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot.T, cmap="Reds", cbar_kws={'label': 'ICU Cases'})
        plt.title("ICU Cases Heatmap by Country and Date")
        plt.show()

        sample_countries = df["Country"].value_counts().head(6).index
        g = sns.FacetGrid(df[df["Country"].isin(sample_countries)], col="Country", col_wrap=3, height=3.5)
        g.map_dataframe(sns.lineplot, x="Date", y="Active_Cases")
        g.set_titles("{col_name}")
        g.set_axis_labels("Date","Active Cases")
        g.set_xticklabels(rotation=90)
        plt.show()


# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    file_path = "covid19_global_data.csv"
    analyzer = CovidDataAnalysis(file_path)

    menu = {
        1: ("Generate Data (Run data_generate.py)", analyzer.generate_data),
        2: ("Load Data", analyzer.load_data),
        3: ("Basic Info", analyzer.basic_info),
        4: ("Handle Missing Values", analyzer.handle_missing_values),
        5: ("All Analysis", analyzer.all_analysis),
        6: ("All Visualizations", analyzer.all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- COVID Data Analysis Menu ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            break
        elif choice in menu:
            menu[choice][1]()
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
