import pandas as pd

def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset.

    Parameters:
    df (DataFrame): The dataset to preprocess.

    Returns:
    DataFrame: The preprocessed dataset.
    """
    assert 'label_num' in df.columns and 'text' in df.columns
    return df
