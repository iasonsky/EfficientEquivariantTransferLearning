import torch
import matplotlib.pyplot as plt

from exp_utils import plot_stacked_bar
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# todos:
# 1) capitalize the title
# 2) remove the zeros and corresponding labels
# 3) re-adjust the position of the legends

def process_scores(filename):
    with open(filename, 'r') as f:
        sentences = [line.rstrip('\n') for line in f]
    scores = [int(sentence.split('\t')[0]) for sentence in sentences]
    negative_ones, zeros, ones, twos = 0, 0, 0, 0
    for score in scores:
        if score == -1:
            negative_ones += 1
        elif score == 0:
            zeros += 1
        elif score == 1:
            ones += 1
        else:
            twos += 1
    result = torch.tensor([twos, ones, zeros, negative_ones])
    result = result / torch.sum(result)
    return result


def plot_fair_jordan_results(data, category_labels, filename, task_title):
    colors_labels = ['w', 'k', 'w', 'tab:gray']
    patterns = ['.', None, '/', None]
    plt.figure()
    plt.xticks(rotation=20, fontsize=15)
    plt.xlabel(5 * ' ' + 'GPT2' + 14 * ' ' + 'R-EquiGPT2' + 10 * ' ' + 'EquiGPT2', fontsize=15)
    # series_labels = ['2', '1', '0', '-1']
    # series_labels = ['-1', '0', '1', '2']
    series_labels = ['other', 'positive', 'neutral', 'negative']
    # series_labels = ['negative', 'neutral', 'positive', 'other']

    plot_stacked_bar(
        data,
        series_labels,
        category_labels=category_labels,
        show_values=True,
        grid=False,
        value_format="{:01.1f}",
        colors=colors_labels,
        patterns=patterns,
        y_label="Regard Classifier Scores"
    )
    plt.title(task_title, fontsize=15)
    plt.savefig('fair_jordan_plots/' + filename, bbox_inches="tight")
    # plt.show()


if __name__ == '__main__':
    dir = 'evaluated_samples/'  # directory containing evaluated samples
    model_names = ['GPT2/', 'equiGPT2/', 'relaxedEquiGPT2/']  # names of models used for sample generation
    plot_title = 'Gender' + 1*'-' + 'Occupation'
    tasks = ['occupation']  # tasks: choose from ['occupation', 'respect']
    demographic_groups = ['man', 'woman']
    plot_name = ''
    data_list = []
    category_labels_list = []
    for task in tasks:
        for model_name in model_names:
            for i in range(len(demographic_groups)):
                demographic_group = demographic_groups[i]
                filename = dir + model_name + task + '_' + demographic_group + '_' + 'predictions.txt'
                scores = process_scores(filename)
                data_list.append(scores)
                category_labels_list.append(demographic_group)
    data = torch.stack(data_list, dim=1)
    category_labels = category_labels_list
    plot_fair_jordan_results(data, category_labels, filename=plot_title + '.png', task_title=plot_title)