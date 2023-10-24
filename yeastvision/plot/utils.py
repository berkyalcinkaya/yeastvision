import matplotlib.pyplot as plt
import math

def arrange_axes(axes_list):
    num_axes = len(axes_list)
    cols = math.ceil(math.sqrt(num_axes))
    rows = math.ceil(num_axes / cols)

    fig = plt.figure()
    for i, ax in enumerate(axes_list):
        ax.set_position([1/cols * (i % cols), 1 - (1/rows) * math.floor(i / cols), 1/cols, 1/rows])
        fig.add_axes(ax)

    return fig





