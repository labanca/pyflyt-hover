import pandas as pd
import ast  # For converting string representation of list to actual list

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt



def plot_pos_heatmap(csv_filename):
    # Load your CSV file
    df = pd.read_csv(csv_filename)

    df = df[df['termination']]


    plt.figure(figsize=(10, 8))
    plt.hist2d(df['posx'], df['posy'], bins=(50, 50), cmap='viridis')
    plt.colorbar()

    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Agent Termination Heatmap')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_pos_heatmap('C:/projects/pyflyt-hover/apps/labanca/evals/pyflyt_hover_20231205-163917.csv')

