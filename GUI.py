import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import NIDS

class ModelVisualizationApp:
    def __init__(self, root, test_X, probabilities):
        self.root = root
        self.root.title("Model Visualization")

        self.test_X = test_X
        self.probabilities = probabilities

        self.create_gui()

    def create_gui(self):
        # Create a Matplotlib figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create a button to update the plot
        update_button = tk.Button(self.root, text="Update Plot", command=self.update_plot)
        update_button.pack()

    def update_plot(self):
        # Get the selected instance (for simplicity, let's assume user provides an instance index)
        instance_index = 0  # Replace with your logic to get the selected instance

        # Clear previous plot
        self.ax.clear()

        # Plot predicted probabilities for each class
        classes = range(len(self.probabilities[instance_index]))
        self.ax.bar(classes, self.probabilities[instance_index], tick_label=[f"Class {j}" for j in classes])
        self.ax.set_title(f"Predicted Probabilities - Test Instance {instance_index + 1}")

        # Redraw canvas
        self.canvas.draw()

if __name__ == "__main__":
    # Assuming 'combination' is your machine learning model
    # and 'test_X' and 'probabilities' are your test data and predicted probabilities
    combination = NIDS.combination # Replace with your model
    test_X = NIDS.test_X  # Replace with your test data
    probabilities = NIDS.probabilities  # Replace with your predicted probabilities

    # Create the Tkinter root and the visualization app
    root = tk.Tk()
    app = ModelVisualizationApp(root, test_X, probabilities)

    # Run the Tkinter event loop
    root.mainloop()
