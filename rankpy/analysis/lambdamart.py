# This file is part of RankPy.
#
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RankPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.ticker import MaxNLocator


def plot_lambdas_andrews_curves(lambdas, relevance_scores):
    columns = ['Tree %d' % i for i in range(1, 1 + lambdas.shape[0])]
    columns.append('Relevance')
    
    data = pd.DataFrame(np.r_[lambdas, relevance_scores.reshape(1, -1).astype(int)].T, columns=columns)
    pd.tools.plotting.andrews_curves(data, 'Relevance')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, map(lambda s: 'Relevance ' + s, labels))

   
def plot_lambdas_parallel_coordinates(lambdas, relevance_scores, individual=False, cumulative=False):
    unique_scores = sorted(np.unique(relevance_scores).astype(int), reverse=True)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
    
    if not individual:
        plt.figure()

        legend_handles = []
        legend_labels = []

        for c, r in enumerate(unique_scores):
            legend_handles.append(mlines.Line2D([], [], color=colors[c], linewidth=2))
            legend_labels.append('Relevance %d' % r)

    if cumulative:
        lambdas_cumsum = lambdas.cumsum(axis=0)
        ymin, ymax = lambdas_cumsum.min(), lambdas_cumsum.max()
    else:
        ymin, ymax = lambdas.min(), lambdas.max()
        
    for c, r in enumerate(unique_scores):
        if individual:
            plt.figure()

        if cumulative:
            plt.plot(lambdas[:, relevance_scores == r].cumsum(axis=0), '-', marker='.', markersize=1, c=colors[c], alpha=0.4)
        else:
            plt.plot(lambdas[:, relevance_scores == r], '-', marker='.', markersize=1, c=colors[c], alpha=0.4)

        if individual:
            plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
            plt.gca().set_ylim([ymin, ymax])

            plt.title('Paralell Coordinates for%sLambdas (Relevance %d)' % (' Cumulative ' if cumulative else ' ', r))
            plt.xlabel('Trees')
            plt.ylabel('Cumulative Lambda Values' if cumulative else 'Lambda Values')
            plt.show()

    if not individual:
        plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
        plt.gca().set_ylim([ymin, ymax])
        
        plt.title('Paralell Coordinates for%sLambdas (Relevance %d)' % (' Cumulative ' if cumulative else ' ', r))
        plt.xlabel('Trees')
        plt.ylabel('Cumulative Lambda Values' if cumulative else 'Lambda Values')    

        plt.legend(legend_handles, legend_labels, loc='best')
        plt.show()
