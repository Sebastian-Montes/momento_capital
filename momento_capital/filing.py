import pandas as pd


def save_dataframes_to_excel(dataframes, sheet_names, file_name):
    if len(dataframes) != len(sheet_names):
        raise ValueError("Each DataFrame must have its own sheet name")
    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
