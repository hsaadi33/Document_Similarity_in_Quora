import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

def plot_bar(df,target_col,title,x_range,colors):
    total_vals = len(df[target_col])
    plt.bar([0,1],[len(df[df[target_col]==0])/total_vals*100,len(df[df[target_col]==1])/total_vals*100], color=colors)
    plt.xticks(x_range, x_range)
    plt.ylabel('%')
    plt.title("Percentage of duplicates");


def plot_boxplots(df,features,label,num_rows):
    """Plots boxplots for each useful feature for all GICS sectors
    Args:
        df: Dataframe containing all generated features with stocks tickers and GICS sectors. (dataframe)
        label: A column to group by. (str)
        features: A group of chosen features. (str list)
        num_rows: Number of rows in subplots. (int)
    """
    plt.style.use('default')
    fig, axs = plt.subplots(ncols=3, nrows = num_rows, figsize=(12,20),constrained_layout=True)
    axs = axs.flatten()
    i = 0
    for feature in features:
        df_temp = df.groupby(label)[feature].apply(list)
        ax = sns.boxplot(data=df_temp, width=.3,orient="v",showmeans=True,
                         meanprops={"marker": "o", "markeredgecolor": "yellow","markersize": "5"}, 
                         whis=[0,100], ax = axs[i])
        ax.set_xticklabels(df_temp.index)
        ax.grid(which='both', axis='both')
        ax.set_title(feature)
        ax.tick_params(axis='y', pad= -2)
        ax.yaxis.labelpad = -2
        ax.tick_params(axis='both', which='major', labelsize=10)
        i += 1
        
    fig.delaxes(axs[-1])
    plt.show()
    

def plot_correlation_matrix(df,features,dimensions):
    """ Plots correlation matrix heat map 
    Args:
        df: A dataframe the contains the numerical features. (dataframe)
        features: Numerical features. (str list)
        dimensions: Dimensions of the plot. (int tuple)
    """
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=dimensions)
    sns.heatmap(corr,annot = True)
    plt.show()
    
def plot_calibration_curves(model,X_test,y_test,y_col,dup_classes):
    """Plots calibration curves for one vs rest classification problem
    Args:
        model: Model to plot calibration curves for.
        X_test: Features test set.
        y_test: Label test set after applying label encoder.
        dup_classes: A list of zero and one indicating duplicity. (str list)
    """
    classes_num = len(dup_classes)
    plt.style.use('default')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    for dup_class in range(classes_num):
        prob_pos = model.predict_proba(X_test)[:, dup_class]
        y_test_plot = [int(dup_class==num) for num in y_test[y_col].values.tolist()]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_plot, prob_pos, n_bins=5)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label = f"{dup_class}")
        plt.ylabel("The proportion of samples whose class is the positive class")
        plt.xlabel("The mean predicted probability in each bin")
    
    plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right')
    plt.show() 
    
    
def plot_roc_curve(model,X_test,y_test):
    preds = model.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()