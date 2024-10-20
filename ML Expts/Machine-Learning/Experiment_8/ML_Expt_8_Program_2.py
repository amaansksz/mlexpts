import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans

# Load and prepare the data for clustering
df = pd.read_csv(r"C:/Users/amaan/Desktop/ML Expts/Machine-Learning/Experiment_8/Mall_Customers.csv")
x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
k_mean = KMeans(n_clusters=5, random_state=42)
y_mean = k_mean.fit_predict(x)

# Function to show the entry fields and predict the cluster
def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    result = k_mean.predict([[p1, p2]])
    print("This customer belongs to cluster no:", result[0])

    cluster_info = {
        0: "Customer with medium annual income & medium spending score",
        1: "Customer with high annual income & low spending score",
        2: "Customer with low annual income & low spending score",
        3: "Customer with low annual income & high spending score",
        4: "Customer with high annual income & high spending score"
    }

    # Clear previous result labels if any
    for widget in master.grid_slaves():
        if int(widget.grid_info()["row"]) >= 4:  # Assuming info labels start from row 4
            widget.destroy()
    
    Label(master, text=cluster_info[result[0]]).grid(row=4)

# Create the main application window
master = Tk()
master.title("Amaan Shaikh S.Customer Segmentation using Machine Learning")

Label(master, text="Amaan Shaikh S.211P052/41 Customer Segmentation using Machine Learning", bg="cyan", fg="black").grid(row=0, columnspan=2)

# Input labels and entries
Label(master, text="Annual Income (k$)").grid(row=1)
Label(master, text="Spending Score (1-100)").grid(row=2)
e1 = Entry(master)
e2 = Entry(master)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

Button(master, text='Predict', command=show_entry_fields).grid(row=3)

# Plotting the clusters
figure = plt.Figure(figsize=(5, 4), dpi=100)
ax = figure.add_subplot(111)

for i in range(5):  # Since n_clusters=5
    ax.scatter(x.iloc[y_mean == i, 0], x.iloc[y_mean == i, 1], s=100, label=f'Cluster {i}')

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title('Annual Income vs Spending Score')
ax.legend()

# Adding the plot to the Tkinter window
scatter = FigureCanvasTkAgg(figure, master)
scatter.get_tk_widget().grid(row=5, columnspan=2)

# Start the Tkinter main loop
master.mainloop()
