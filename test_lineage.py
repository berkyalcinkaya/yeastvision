import matplotlib.cm as cm
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def plot_tree(matrix, selected_cells=[]):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = {}
    current_y = 0  # Starting position

    # Bright color list for selected cells
    bright_colors = [
        '#FF0000',  # Bright red
        '#00FF00',  # Bright green
        '#0000FF',  # Bright blue
        '#FF00FF',  # Bright magenta
        '#00FFFF',  # Bright cyan
        '#FFFF00',  # Bright yellow
    ]
    cell_colors = {selected_cells[i]: bright_colors[i % len(bright_colors)] for i in range(len(selected_cells))}

    def plot_lineage(mother, y):
        daughters = matrix[matrix[:, 3] == mother]
        daughters = daughters[daughters[:, 1].argsort()[::-1]]

        mother_color = cell_colors.get(mother, 'grey')

        for daughter in daughters:
            cell_index, birth_frame, death_frame, _ = daughter
            y += 1

            while y in y_positions.values():
                y += 1

            color = mother_color if cell_index in cell_colors or mother_color != 'grey' else 'grey'
            ax.plot([birth_frame, death_frame], [y, y], color=color)
            ax.text(death_frame + 1, y, str(cell_index), color=color, verticalalignment='center',
                    horizontalalignment='left')

            ax.plot([birth_frame, birth_frame], [y_positions[mother], y], color=mother_color)

            y_positions[cell_index] = y
            y = plot_lineage(cell_index, y)

        return y

    no_mothers = matrix[matrix[:, 3] == -1]
    for row in no_mothers:
        cell_index, birth_frame, death_frame, _ = row
        color = cell_colors.get(cell_index, 'grey')
        ax.plot([birth_frame, death_frame], [current_y, current_y], color=color)
        ax.text(death_frame + 1, current_y, str(cell_index), color=color, verticalalignment='center',
                horizontalalignment='left')
        y_positions[cell_index] = current_y
        current_y = plot_lineage(cell_index, current_y) + 1

    ax.set_ylim(-1, current_y + 1)
    ax.set_xlabel('Frame')
    ax.get_yaxis().set_visible(False)
    plt.grid(True)
    plt.tight_layout()
    return fig

class App(QMainWindow):
    def __init__(self, matrix, selected):
        super().__init__()
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI(matrix, selected)

    def initUI(self, matrix, selected):
        self.setWindowTitle('Lineage Tree Plot')
        self.setGeometry(self.left, self.top, self.width, self.height)
        m = plot_tree(matrix, selected_cells=selected)
        canvas = FigureCanvas(m)
        self.setCentralWidget(canvas)
        self.show()

if __name__ == '__main__':
    matrix = np.array([
        [1, 0, 50, -1],
        [2, 0, 48, -1],
        [3, 2, 51, 1],
        [4, 3, 45, 1],
        [5, 4, 52, 2],
        [6, 5, 53, 2],
        [7, 10, 54, 3],
        [8, 15, 50, 3],
        [9, 18, 46, 4],
        [10, 22, 49, 4]
    ])

    selected = [1, 3, 7]

    app = QApplication(sys.argv)
    ex = App(matrix, selected)
    sys.exit(app.exec_())
