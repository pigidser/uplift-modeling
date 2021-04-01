import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modeling import Model 
from analysis import AnalysisMetrics

class Comparison(object):
    
    def __init__(self, models):
        # Check 'models' variable
        if not isinstance(models, list):
            raise Exception("Models should be list of models!")
        for model in models:
            if not isinstance(model, Model):
                raise Exception("Models should be list of models!")
        self.models = models
        
    def print_metrics(self):
        for model in self.models:
            analysis_metrics = AnalysisMetrics(model=model)
            print(analysis_metrics.overalls.shape)
        
    def plot_metric_by_clients(self, metric_name, ylabel, title):
        
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        df = pd.concat([self.overalls, self.accounts], axis=1)
        
        columns_m = list()
        columns_h = list()
        columns_m_h = list()
        columns_f_m_h = list()
        labels = list()
        for column in df.columns:
            if metric_name + '_m' in column:
                columns_m.append(column)
                labels.append(column[:-len(metric_name)-4])
            elif metric_name + '_h' in column:
                columns_h.append(column)
            elif metric_name + '__m' in column:
                columns_m_h.append(column)
            elif metric_name + '_f_m' in column:
                columns_f_m_h.append(column)

        df_m = df[columns_m]
        df_h = df[columns_h]
        df_m_h = df[columns_m_h]
        df_f_m_h = df[columns_f_m_h]

        model = df_m.squeeze().values
        human = df_h.squeeze().values
        model_human = df_m_h.squeeze().values
        future_model_human = df_f_m_h.squeeze().values
        
        x = np.arange(len(labels))  # the label locations
        width = 0.22  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width - width/2, model, width, label='Model vs Fact')
        rects2 = ax.bar(x - width/2, human, width, label='Human vs Fact')
        rects3 = ax.bar(x + width/2, model_human, width, label='Model vs Human (Historic)')
        rects4 = ax.bar(x + width + width/2, future_model_human, width, label='Model vs Human (Future)')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation = 45)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        fig.tight_layout()

        plt.show()

