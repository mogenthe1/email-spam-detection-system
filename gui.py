import tkinter as tk
from tkinter import messagebox
from visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from model import classify_email

def create_gui(model, vectorizer, conf_matrix, y_test, y_prob):
    def show_result():
        email_text = entry.get("1.0", tk.END).strip()
        if email_text:
            result = classify_email(model, vectorizer, email_text)
            color = "red" if result == "Spam" else "green"
            result_label.config(text=f"The email is classified as: {result}", fg=color)
        else:
            messagebox.showwarning("Input Error", "Please enter the email text.")

    def show_visualizations():
        plot_confusion_matrix(conf_matrix)
        plot_roc_curve(y_test, y_prob)
        plot_precision_recall_curve(y_test, y_prob)

    window = tk.Tk()
    window.title("Email Spam Classifier")

    frame = tk.Frame(window, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    label = tk.Label(frame, text="Enter the email text below:")
    label.pack()

    global entry, result_label
    entry = tk.Text(frame, width=60, height=15)
    entry.pack(pady=10)

    classify_button = tk.Button(frame, text="Classify Email", command=show_result)
    classify_button.pack(pady=5)

    visualize_button = tk.Button(frame, text="Show Visualizations", command=show_visualizations)
    visualize_button.pack(pady=5)

    result_label = tk.Label(frame, text="", font=("Helvetica", 12))
    result_label.pack(pady=5)

    exit_button = tk.Button(frame, text="Exit", command=window.quit)
    exit_button.pack(pady=5)

    window.mainloop()
