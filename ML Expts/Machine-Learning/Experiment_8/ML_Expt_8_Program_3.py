import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans

# Load and prepare the data for clustering
df = pd.read_csv(r"C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_8/Placement.csv")
# Strip whitespace from column names
df.columns = df.columns.str.strip()
x = df[['cgpa', 'package']]

# Initialize KMeans with the desired number of clusters
k_mean = KMeans(n_clusters=5, random_state=42)
y_mean = k_mean.fit_predict(x)

# Function to show the entry fields and predict the cluster
def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    result = k_mean.predict([[p1, p2]])
    print("This student belongs to cluster no:", result[0])

    cluster_info = {
        0: "Students with medium CGPA and medium package",
        1: "Students with high CGPA and low package",
        2: "Students with low CGPA and low package",
        3: "Students with low CGPA and high package",
        4: "Students with high CGPA and high package"
    }

    # Clear previous labels if any
    for widget in master.grid_slaves():
        if int(widget.grid_info()["row"]) >= 4:  # Assuming info labels start from row 4
            widget.destroy()

    Label(master, text=cluster_info[result[0]]).grid(row=4)

# Create the main application window
master = Tk()
master.title("Student Placement Segmentation Amaan Shaikh S. 211P052/41")

# Title label
Label(master, text="Student Placement Segmentation using Machine Learning", bg="cyan", fg="black").grid(row=0, columnspan=2)

# Input labels and entries
Label(master, text="CGPA").grid(row=1)
Label(master, text="Package").grid(row=2)
e1 = Entry(master)
e2 = Entry(master)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

# Predict button
Button(master, text='Predict', command=show_entry_fields).grid(row=3)

# Plotting
figure = plt.Figure(figsize=(5, 4), dpi=100)
ax = figure.add_subplot(111)

# Scatter plot for clusters
for i in range(5):  # Since n_clusters=5
    ax.scatter(x.iloc[y_mean == i, 0], x.iloc[y_mean == i, 1], s=100, label=f'Cluster {i}')

# Set labels and title
ax.set_xlabel('CGPA')
ax.set_ylabel('Package')
ax.set_title('CGPA vs Package')
ax.legend()

# Displaying figure in Tkinter
scatter = FigureCanvasTkAgg(figure, master)
scatter.get_tk_widget().grid(row=5, columnspan=2)

# Run the application
master.mainloop()
