
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import sklearn.learning_curve as curves
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit, train_test_split
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

# latex parameter
font = {
    'family': 'serif', 
    'serif': ['Computer Modern Roman'],
    'weight' : 'regular',
    'size'   : 18
    }

plt.rc('font', **font)
plt.rc('text', usetex=True)
# plt.style.use('classic')

color_map = 'viridis'

class vizualization:
    
    def __init__(self, df):
        self.df = df
    
    def check_null(self):
        print(self.df.isnull().any().any())
        
    def check_missing_value(self, missing=None, start=2, end=39):
        df_copy = self.df
        if missing == None:
            df_copy = df_copy.replace('NaN', np.NaN)
        else:
            df_copy = df_copy.replace(-1, np.NaN)
        
        msno.matrix(self.df_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))
    
    def check_data_balance(self, target_column=None):

        if isinstance(target_column, str):  
            x = self.df[target_column].value_counts().index.values
            y = self.df[target_column].value_counts().values
        elif isinstance(target_column, int):
            if target_column == -1:
                x = self.df.iloc[:, -1].value_counts().index.values
                y = self.df.iloc[:, -1].value_counts().values
                print(x,y)
            else:
                x = self.df.iloc[:, target_column].value_counts().index.values
                y = self.df.iloc[:, target_column].value_counts().values
        elif isinstance(target_column, None):
            print('please provide the target feature name or index')
        
        
        total = sum(y)
        labels = ['{} percent'.format(np.round(val*100/total), 2) for i, val in enumerate(y)]
        
        
        plt.figure(figsize=(10, 5))
        plt.bar(x, y, align='center', color= '#AA0000', alpha=0.80)
        plt.xticks([0, 1])
        plt.xlabel('Classes')
        plt.ylabel('Number of data')

        for i, val in enumerate(y):
            x_cord = x[i]
            y_cord = val * 0.5
            plt.text(x_cord, y_cord, labels[i], ha='center', va='bottom')

        plt.show()
    
    def check_datatype(self):
        print(self.df.dtypes)
        
    def check_row_columns(self):
        rows = train.shape[0]
        columns = train.shape[1]
        print("The train dataset contains {0} rows and {1} columns".format(rows, columns))

    def check_correlation(self):
        colormap = plt.cm.cubehelix_r
        plt.figure(figsize=(16,12))
        plt.title('Pearson correlation of continuous features', y=1.05, size=15)
        sns.heatmap(self.df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        
    def check_pca(self, pca1=1, pca2=2, target):
        from sklearn.decomposition import PCA
        
        features = self.df.drop([target], axis = 1)
        pcamat = PCA(n_components=2).fit_transform(Xmat)

        plt.figure()
        plt.scatter(pcamat[:X_train1.shape[0],0],pcamat[:X_train1.shape[0],1], c='b', label='targ 0', alpha=0.1)
        plt.scatter(pcamat[X_train1.shape[0]:,0],pcamat[X_train1.shape[0]:,1],c='r', label='targ 1', alpha=0.1)
        plt.legend()
        plt.title('PC space')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()
        print("PCA done")
        
        
    def check_model_learning(self, X, y, depth_arr):
        """ Calculates the performance of several models with varying sizes of training data.
            The learning and testing scores for each model are then plotted. """

        # Create 10 cross-validation sets for training and testing
        cv = ShuffleSplit(X.shape[0], n_iter = 100, test_size = 0.2, random_state = 0)

        # Generate the training set sizes increasing by 50
        train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

        # Create the figure window
        fig = pl.figure(figsize=(10,7))

        # Create three different models based on max_depth
        for k, depth in enumerate(depth_arr):

            # Create a Decision tree regressor at max_depth = depth
            regressor = RandomForestClassifier(criterion='entropy', max_depth = depth, min_samples_split=20,               class_weight='balanced')

            # Calculate the training and testing scores
            sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
                cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

            # Find the mean and standard deviation for smoothing
            train_std = np.std(train_scores, axis = 1)
            train_mean = np.mean(train_scores, axis = 1)
            test_std = np.std(test_scores, axis = 1)
            test_mean = np.mean(test_scores, axis = 1)

            # Subplot the learning curve 
            pl.subplot(3, 3, k+1)
            pl.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
            pl.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
            # ax.fill_between(sizes, train_mean - train_std, \
            #     train_mean + train_std, alpha = 0.15, color = 'r')
            # ax.fill_between(sizes, test_mean - test_std, \
            #     test_mean + test_std, alpha = 0.15, color = 'g')

            # Labels
            pl.title('max_depth = %s'%(depth))
            pl.xlabel('Number of Training Points')
            pl.ylabel('Score')
            pl.xlim([0, X.shape[0]*0.8])
            pl.ylim([-0.05, 1.05])

        # Visual aesthetics
        pl.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
        pl.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
        pl.tight_layout()
        pl.savefig('learning_performance.eps')
        pl.show()


    def check_model_complexity(self, X, y):
        """ Calculates the performance of the model as model complexity increases.
            The learning and testing errors rates are then plotted. """

        # Create 10 cross-validation sets for training and testing
        cv = ShuffleSplit(X.shape[0], n_iter = 100, test_size = 0.2, random_state = 0)

        # Vary the max_depth parameter from 1 to 10
        max_depth = np.arange(1,11)

        # Calculate the training and testing scores

        train_scores, test_scores = curves.validation_curve(RandomForestClassifier(criterion='entropy', min_samples_split=20), X, y, \
            param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'accuracy')

        # Find the mean and standard deviation for smoothing
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the validation curve
        pl.figure(figsize=(7, 5))
        pl.title('Decision Tree Regressor Complexity Performance')
        pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
        pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
        # pl.fill_between(max_depth, train_mean - train_std, \
        #     train_mean + train_std, alpha = 0.15, color = 'r')
        # pl.fill_between(max_depth, test_mean - test_std, \
        #     test_mean + test_std, alpha = 0.15, color = 'g')

        # Visual aesthetics
        pl.legend(loc = 'lower right')
        pl.xlabel('Maximum Depth')
        pl.ylabel('Score')
        pl.ylim([-0.05,1.05])
        pl.savefig('validation_curve.eps')
        pl.show()


    def PredictTrials(self, X, y, fitter, data):
        """ Performs trials of fitting and predicting data. """

        # Store the predicted prices
        prices = []

        for k in range(10):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, \
                test_size = 0.2, random_state = k)

            # Fit the data
            reg = fitter(X_train, y_train)

            # Make a prediction
            pred = reg.predict([data[0]])[0]
            prices.append(pred)

            # Result
            print("Trial {}: ${:,.2f}".format(k+1, pred))

        # Display price range
        print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))