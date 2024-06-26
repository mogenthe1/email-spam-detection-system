import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_data, preprocess_data
from model import train_model, evaluate_model
from gui import create_gui

def main():
    global model, vectorizer, conf_matrix, y_test, y_prob

    # ASCII art to be printed once
    ascii_art = r"""
  _     _             _ _                _                       _ _                           _                __  __                                                               _         
 | |__ (_)_ __  _ __ (_) |_ _   _       | |__   ___  _ __  _ __ (_) |_ _   _         __ _  ___| |_        ___  / _|/ _|       _ __ ___  _   _        _ __  _ __ ___  _ __   ___ _ __| |_ _   _ 
 | '_ \| | '_ \| '_ \| | __| | | |      | '_ \ / _ \| '_ \| '_ \| | __| | | |       / _` |/ _ \ __|      / _ \| |_| |_       | '_ ` _ \| | | |      | '_ \| '__/ _ \| '_ \ / _ \ '__| __| | | |
 | | | | | |_) | |_) | | |_| |_| |      | | | | (_) | |_) | |_) | | |_| |_| |      | (_| |  __/ |_      | (_) |  _|  _|      | | | | | | |_| |      | |_) | | | (_) | |_) |  __/ |  | |_| |_| |
 |_| |_|_| .__/| .__/|_|\__|\__, |      |_| |_|\___/| .__/| .__/|_|\__|\__, |       \__, |\___|\__|      \___/|_| |_|        |_| |_| |_|\__, |      | .__/|_|  \___/| .__/ \___|_|   \__|\__, |
         |_|   |_|          |___/                   |_|   |_|          |___/        |___/                                               |___/       |_|             |_|                  |___/ 
    """

    print(ascii_art)

    # Load and preprocess data
    file_path = 'trainingdata/spam_ham_dataset.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

    # Train the model
    model, vectorizer = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report, conf_matrix, y_prob, y_pred = evaluate_model(model, vectorizer, X_test, y_test)

    # Print evaluation results to the console
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Start the GUI
    create_gui(model, vectorizer, conf_matrix, y_test, y_prob)

if __name__ == "__main__":
    main()
