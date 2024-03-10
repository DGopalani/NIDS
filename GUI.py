import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import NIDS

class ModelVisualizationApp:
    def __init__(self, root, test_X, probabilities):
        self.root = root
        self.root.title("Model Visualization")

        self.test_X = test_X
        self.probabilities = probabilities
        self.selected_instance = None

        self.create_gui()

    def create_gui(self):
        # Create a Matplotlib figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create horizontal scale for zooming
        zoom_scale = tk.Scale(self.root, from_=1, to=5, orient=tk.HORIZONTAL, resolution=0.1, label="Zoom", command=self.zoom_function)
        zoom_scale.pack(fill=tk.X)

        # Bind the click event on the plot
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # Initialize the plot
        self.update_plot()

    def update_plot(self):
        # Clear previous plot
        self.ax.clear()

        # Find the class with the highest probability for each instance
        max_class_indices = [np.argmax(prob) for prob in self.probabilities]

        # Plot the instances on the x-axis and the class with the highest probability on the y-axis
        scatter = self.ax.scatter(range(1, len(self.test_X) + 1), max_class_indices, picker=True)
        self.ax.set_title("Class with Highest Probability for Each Test Instance")
        self.ax.set_xlabel("Test Instance")
        self.ax.set_ylabel("Class Index")

        # Store the scatter plot for later reference
        self.scatter = scatter

        # Redraw canvas
        self.canvas.draw()

    def on_pick(self, event):
        # Get the index of the clicked instance
        self.selected_instance = int(event.mouseevent.xdata) - 1

        # Display probabilities for the selected instance
        self.show_probabilities()

    def show_probabilities(self):
        if self.selected_instance is not None:
            # Display probabilities for the selected instance
            probabilities_str = "\n".join([f"Class {j}: {prob:.4f}" for j, prob in enumerate(self.probabilities[self.selected_instance])])
            tk.messagebox.showinfo("Probabilities", f"Probabilities for Test Instance {self.selected_instance + 1}:\n{probabilities_str}")

    def zoom_function(self, zoom_level):
        # Perform zooming based on the scale value
        zoom_factor = float(zoom_level)
        self.ax.set_xlim(self.ax.get_xlim()[0], self.ax.get_xlim()[1] * zoom_factor)

        # Update the scatter plot data positions
        self.scatter.set_offsets(np.column_stack((range(1, len(self.test_X) + 1), [np.argmax(prob) for prob in self.probabilities])))

        # Redraw canvas
        self.canvas.draw()



if __name__ == "__main__":
    # Assuming 'combination' is your machine learning model
    # and 'test_X' and 'probabilities' are your test data and predicted probabilities
    combination = NIDS.combination  # Replace with your model
    test_X = NIDS.test_X  # Replace with your test data
    probabilities = NIDS.probabilities  # Replace with your predicted probabilities

    # Create the Tkinter root and the visualization app
    root = tk.Tk()
    app = ModelVisualizationApp(root, test_X, probabilities)

    # Run the Tkinter event loop
    root.mainloop()
