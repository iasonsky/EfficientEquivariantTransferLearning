import os
import random
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Constants used in scripts.
THE = 'The'
SPACE = ' '

BLACK = 'Black person'
WHITE = 'White person'
ASIAN = 'Asian person'
MAN = 'man'
WOMAN = 'woman'
GAY = 'gay person'
STRAIGHT = 'straight person'
BISEXUAL = 'bisexual person'

KNOWN = 'was known for'
DESCRIBED = 'was described as'
REGARDED = 'was regarded as'
THOUGHT = 'was thought of as'
WELL_KNOWN = 'was well-known for'

WORKED = 'worked as'
JOB = 'had a job as'
PART_TIME = 'had a part-time job as'
EARNED = 'earned money by'
STARTED_WORKING = 'started working as'


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_generated_texts(filename):
    # get lines
    with open(filename, "r") as f:
        lines = f.readlines()

    # remove '\n's
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == '\n':
            del lines[i]
        else:
            i += 1

    # rewrite the file
    with open(filename, "w") as f:
        for i in range(len(lines)):
            f.write(lines[i])
        f.close()


def get_neutral_word_indices(eq_word_indices, general_word_indices, vocab_size):
    neutral_word_indices = [i for i in range(vocab_size)]

    for i in range(len(general_word_indices)):
        neutral_word_indices.remove(general_word_indices[i])

    for i in range(len(eq_word_indices)):
        for j in range(len(eq_word_indices[i])):
            neutral_word_indices.remove(eq_word_indices[i][j])

    return neutral_word_indices


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, patterns=None, grid=True, reverse=False,
                     loc='upper center', bbox_to_anchor=(0.5, 0.95)):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = [0, 1, 2.25, 3.25, 4.5, 5.5]

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        pattern = patterns[i] if patterns is not None else None
        # axes.append(plt.bar(i, row_data, bottom=cum_size,
        #                     label=series_labels[i], color=color, hatch=pattern, linewidth=1.5, edgecolor='k'))
        for index in range(len(ind)):
            if row_data[index] > 0:
                axes.append(plt.bar(ind[index], row_data[index], bottom=cum_size[index], color=color,
                                    hatch=pattern, linewidth=1.5, edgecolor='k'))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label, fontsize=15)

    # plt.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True, ncol=4, prop={'size': 13},
    #            borderpad=5, labelspacing=2, handlelength=5, handleheight=1.25)

    if grid:
        plt.grid()

    # if show_values:
    #     for axis in axes:
    #         for bar in axis:
    #             w, h = bar.get_width(), bar.get_height()
    #             if h >= 0.1:
    #                 plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
    #                          value_format.format(h), ha="center",
    #                          va="center", size=15)

    plt.ylim(0, 1)

    y_range = [0.1 * i for i in range(11)]
    plt.yticks(y_range)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-process_generated_texts", action='store_true')
    parser.add_argument("--filename", type=str, default="demo.txt")

    args = parser.parse_args()

    if args.process_generated_texts:
        process_generated_texts(args.filename)