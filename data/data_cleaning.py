"""
This is for the intial data cleaning and aggregating to get the full dataset in one file. Through SQL queries on the POS system
(DecorFusion), most of the relevant data has been extracted, however the retail prices and contractor prices by tier are in a separate
Excel file that needs to be merged.

The Excel file has a complex structure:
Case 1: Fixed Pricing
- Trigger: Column M says "Fixed Pricing" (or "Fixed Price").
- Logic: The final retail price is already calculated and sits directly in Column P.

Case 2: Gross Margin
- Trigger: Column M says "Gross Margin".
- Logic: The price must be calculated dynamically.
- Formula: Column J (Standard Cost) / (1 - (Column P (Margin %) / 100)).

Case 3: Item Alias (Reference)
- Trigger: Column M contains another Item Code (e.g., "K2004X-006").
- Logic: The original item doesn't have its own price listed. The code must
  re-search the dataset for this new alias code, find its row, and pull
  the retail price from that alias's Column P.
"""

import pandas as pd

df_sales = pd.read_excel("data/raw_data.xlsx", sheet_name="Sheet")
df_prices = pd.read_excel("data/raw_data.xlsx", sheet_name="Price_Levels")

# Create a dictionary from the Prices sheet for fast lookups
prices_dict = df_prices.set_index("Item").to_dict("index")


def calculate_all_prices(item_code, visited_aliases=None):
    empty_result = pd.Series(
        {
            "Standard Cost": None,
            "Retail Price": None,
            "Contractor 1 Price": None,
            "Contractor 2 Price": None,
            "Contractor 3 Price": None,
        }
    )

    # Prevent Infinite Loops
    if visited_aliases is None:
        visited_aliases = set()
    if item_code in visited_aliases:
        return empty_result
    visited_aliases.add(item_code)

    # Check if the item exists in price list
    if item_code not in prices_dict:
        return empty_result

    # Get the row of data for this item
    row = prices_dict[item_code]

    method = row["Price Method/Same As Item"]
    val_j = row["Standard Cost"]

    price_columns = {
        "Retail Price": "Retail Selling Price",
        "Contractor 1 Price": "Contractor 1 / Designer",
        "Contractor 2 Price": "Contractor 2",
        "Contractor 3 Price": "Contractor 3",
    }
    results = {"Standard Cost": val_j}

    # Rule 1: Fixed Price
    if method == "Fixed Pricing":
        for new_col, excel_col in price_columns.items():
            results[new_col] = row.get(excel_col, None)
        return pd.Series(results)

    # Rule 2: Gross Margin
    elif method == "Gross Margin":
        for new_col, excel_col in price_columns.items():
            val_margin = row.get(excel_col, None)
            try:
                # Calculate: Cost / (1 - (Margin / 100))
                results[new_col] = val_j / (1 - (val_margin / 100))
            except (ZeroDivisionError, TypeError):
                results[new_col] = None
        return pd.Series(results)

    # Rule 3: It's an Alias (Another Item Code)
    else:
        alias_code = method
        # Pass the alias back into this exact function to dig up the actual prices
        return calculate_all_prices(alias_code, visited_aliases)


# Apply the logic to every row in sales data
# Create a new column 'Retail Price' by applying our function to Column D
new_columns = [
    "Standard Cost",
    "Retail Price",
    "Contractor 1 Price",
    "Contractor 2 Price",
    "Contractor 3 Price",
]
df_sales[new_columns] = df_sales["Item No"].apply(calculate_all_prices)

df_sales.to_csv("data/aggregated_sales_data.csv", index=False, encoding="utf-8-sig")
