# Utils: Code Reference: https://github.com/amir-jafari/Capstone/tree/main/Sample_Capstone

# -*- coding: utf-8 -*-
"""
Author: Shikha
Date: 2024-10-24
Version: 1.0
"""

def remove_columns(data, columns_to_remove):
    """
    Remove specified columns (case-insensitive) from the dataset.
    
    :rtype: pd.DataFrame
    :param data: The input DataFrame from which columns need to be removed.
    :param columns_to_remove: List of column names (case-insensitive) to remove from the dataset.
    :return: A DataFrame with the specified columns removed.
    """
    # Normalize column names to lowercase for case-insensitive matching
    data_columns_lower = [col.lower() for col in data.columns]
    
    # Remove columns from the dataset if they exist
    for col in columns_to_remove:
        col_lower = col.lower()
        if col_lower in data_columns_lower:
            col_index = data_columns_lower.index(col_lower)
            data = data.drop(columns=[data.columns[col_index]])

    return data


def func(a: object) -> object:
    """
    :rtype: object

    """
    return  a


def func1(b: object) -> object:
    """
    :rtype: object

    """
    return  b