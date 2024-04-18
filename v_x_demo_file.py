import pandas as pd


def get_max_coffee_drinker(data):
    df = pd.DataFrame(data)
    max_count = df["coffee_count"].max()
    max_drinker = df.loc[df["coffee_count"] == max_count, "name"].values[0]
    return max_drinker


def main():
    employee_data = {"name": ["Roberto", "Daniele", "Viet"], "coffee_count": [2, 3, 1]}
    top_drinker = get_max_coffee_drinker(employee_data)
    print(f"Person with highest coffee count today: {top_drinker}")


if __name__ == "__main__":
    main()
