import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import colors
from IPython.display import set_matplotlib_formats, display, Latex
set_matplotlib_formats('svg','pdf')

cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def plot_data(lda, X, y, y_pred, name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
    
    test_size = int(X.size / 4)
    test_vec1_x, test_vec1_y = X[:test_size].T
    test_vec2_x, test_vec2_y = X[test_size + 1:].T
    ax1.plot(test_vec1_x, test_vec1_y, 'r.', test_vec2_x, test_vec2_y, 'b.')
    
    tp = (y == y_pred)
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    
    ax2.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    ax2.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')
    
    
    ax2.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    ax2.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')
    
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    ax2.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0, shading='auto')
    ax2.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    ax2.plot(lda.means_[0][0], lda.means_[0][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    ax2.plot(lda.means_[1][0], lda.means_[1][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    
    fig.savefig(name)
    #plt.suptitle("Test results")
    #plt.show()
    
    
def display_confusion_matrix(cf_matrix, name):
    group_names = ['True 1','False 2','False 1','True 2']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    #accuracy  = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    #stats_text = "\n\nAccuracy = {:0.3f}".format(accuracy)
    
    hmap = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    lbs = [1, 2]
    hmap.set_xticklabels(lbs)
    hmap.set_yticklabels(lbs)
    #hmap.set_xlabel(stats_text, fontweight='bold')
    plt.yticks(rotation=0)
    #plt.xlabel(stats_text, fontweight='bold')
    plt.show()
    hmap.get_figure().savefig(name)
    
def calculate_metrics(cf_matrix):
    TN = cf_matrix[0][0]
    FN = cf_matrix[1][0]
    TP = cf_matrix[1][1]
    FP = cf_matrix[0][1]
    A = (TN + TP) / (TN + TP + FN + FP)
    #P = TP / (TP + FP)
    #R = TP / (TP + FN)
    #F1 = 2 * P * R / (P + R)
    df = pd.DataFrame(dict(metrics=['Accuracy'], 
                      value = [A]))
    #display(df)
    return df

def print_metrics_to_file(df, name):
    ans = df.to_latex(index=False, column_format= '|l|c|')
    ans = ans.replace('\\bottomrule','').replace('\\midrule', '').replace("toprule","hline").replace("\\\n", "\\ \hline\n")
    with open(name,'w') as f:
        f.write(ans)