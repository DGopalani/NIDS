import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from NIDS import *

class NIDSGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("NIDS GUI")

        # Create a listbox to display test instances
        self.listbox = tk.Listbox(self.master, selectmode=tk.SINGLE, width=40, height=20)
        self.listbox.pack(pady=10)

        # Populate the listbox with test instances
        for i in range(min(100, len(test_X))):
            self.listbox.insert(tk.END, f"Test Instance {i + 1}")

        # Button to view details of selected test instance
        self.view_details_button = ttk.Button(self.master, text="View Details", command=self.view_details)
        self.view_details_button.pack(pady=5)

    def view_details(self):
        # Get the selected test instance index
        selected_index = self.listbox.curselection()

        if selected_index:
            selected_index = selected_index[0]  # Extracting the index from the tuple

            # Open a new window to display details
            details_window = tk.Toplevel(self.master)
            details_window.title(f"Details - {self.listbox.get(selected_index)}")

            # Display details in the new window
            details_text = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, width=80, height=20)
            details_text.insert(tk.END, f"{self.listbox.get(selected_index)} Details\n")
            details_text.insert(tk.END, f"\n")
            details_text.insert(tk.END, f"Predicted Class: {get_predicted_class(selected_index)}\n")
            details_text.insert(tk.END, f"\n")
            details_text.insert(tk.END, f"Feature Values: \n{get_selected_class_feature_values(selected_index)}\n")
            details_text.insert(tk.END, f"Predicted Probabilities: \n{get_predicted_probabilities(selected_index)}\n")
            
            details_text.pack(pady=10)

            # Button to close the details window
            close_button = ttk.Button(details_window, text="Close", command=details_window.destroy)
            close_button.pack(pady=5)
    def get_selected_index(self):
        # Get the index of the selected test instance in the scrolled text widget
        try:
            selected_index = int(self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST).split()[2]) - 1
            return selected_index if 0 <= selected_index < len(test_X) else -1
        except ValueError:
            return -1

def get_predicted_class(index):
    # Function to get the predicted class for a given test instance index
    if 0 <= index < len(y_test_pred_combined):
        if y_test_pred_combined[index] == 0:
            return "BENIGN"
        elif y_test_pred_combined[index] == 1:
            return "UDPLag"
        elif y_test_pred_combined[index] == 2:
            return "UDP"
        elif y_test_pred_combined[index] == 3:
            return "SYN"
        elif y_test_pred_combined[index] == 4:
            return "WebDDoS"
    return ""

def get_feature_values(index):
    # Function to get the feature values and descriptions for a given test instance index
    if 0 <= index < len(test_df_filtered_to_print):
        feature_values = test_df_filtered_to_print.iloc[index].to_dict()
        descriptions = [f"{feature}: {feature_values.get(feature, 'N/A')}, Description: {feature_descriptions.get(feature, 'N/A')}\n" for feature in feature_values.keys()]
        return "\n".join(descriptions)
    return ""

def get_selected_class_feature_values(index):
        predicted_class = get_predicted_class(index)
        selected_class_features = []

        if predicted_class == "BENIGN":
            selected_class_features = vars_to_keep
            selected_class_descriptions = feature_descriptions
            relevance = {}
        elif predicted_class == "UDPLag":
            selected_class_features = UDPLag.UDPLag_vars_to_keep + [' Timestamp', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port']
            selected_class_descriptions = feature_descriptions
            relevance = UDPLag_description
        elif predicted_class == "UDP":
            selected_class_features = UDP.UDP_vars_to_keep + [' Timestamp', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port']
            selected_class_descriptions = feature_descriptions
            relevance = UDP_description
        elif predicted_class == "SYN":
            selected_class_features = SYN.SYN_vars_to_keep + [' Timestamp', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port']
            selected_class_descriptions = feature_descriptions
            relevance = {}
        elif predicted_class == "WebDDoS":
            selected_class_features = UDPLag.UDPLag_vars_to_keep + [' Timestamp', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port']
            selected_class_descriptions = feature_descriptions
            relevance = {}

        if 0 <= index < len(test_df_filtered_to_print):
            feature_values = test_df_filtered_to_print.iloc[index].to_dict()
            descriptions = [
                f"{feature}: {feature_values.get(feature, 'N/A')}\n"
                f"Description: {selected_class_descriptions.get(feature, 'N/A')}\n"
                f"Relevance: {relevance.get(feature, 'N/A')}\n"
                for feature in selected_class_features
            ]
            return "\n".join(descriptions)
        return ""

def get_predicted_probabilities(index):
    # Function to get the predicted probabilities for a given test instance index
    if 0 <= index < len(probabilities):
        class_labels = ["BENIGN", "UDPLag", "UDP", "SYN", "WebDDoS"]
        class_probabilities = probabilities[index]
        probabilities_text = "\n".join([f"{class_labels[j]} Probability: {prob:.4f}" for j, prob in enumerate(class_probabilities)])
        return probabilities_text
    return ""
    

if __name__ == "__main__":
    root = tk.Tk()
    nidsgui = NIDSGUI(root)
    root.mainloop()
