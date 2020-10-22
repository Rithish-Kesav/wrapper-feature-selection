import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


class ForwardSearch():
    """sklearn compatible implementation of the forward-search feature selection algorithm.

    Args:
        mode (str): what kind of foward-search to perform. If equal to 'first', add first random feature
        that improves performance in each iteration. If equal to 'maximize', add feature that maximizes the
        improvement of performance in each iteration.
        eval_alg (object): prediction algorithm used to evaluate the subsets.
        cv (object): cross-validation generator or an iterable used to evaluate the feature subset.
        verbose (bool): if True, print progress.

    Attributes:
        mode (str): what kind of foward-search to perform.
        eval_alg (object): prediction algorithm used to evaluate the subsets.
        cv (object): cross-validation generator or an iterable used to evaluate the feature subset.
        selected_features (list): list for storing indices of selected features.
        verbose (bool): if True, print progress.
    """

    def __init__(self, mode="first", eval_alg=RandomForestClassifier(), 
            cv=StratifiedKFold(n_splits=10, shuffle=True), verbose=False):

        # Check mode argument.
        if mode not in {'first', 'maximize'}:
            raise ValueError("mode should either be equal to 'first' or 'maximize'")
        self.mode = mode

        # Set evaluation algorithm.
        self.eval_alg = eval_alg

        # Set cross-validation generator or an iterable.
        self.cv = cv

        # Initialize list for storing selected features.
        self.selected_features = []
        
        # Set verbose flag.
        self.verbose = verbose


    def fit(self, data, target):
        """
        Perform feature selection using the forward search algorithm.

        Args:
            data (numpy.ndarray): array of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # If adding random feature that increases performance.
        if self.mode == 'first':
                
            # Initialize array for the dataset with selected features.
            data_new = np.empty((data.shape[0], 0), dtype=data.dtype)
                
            # Initialize performance of previous feature subset.
            acc_prev = 0.0
            
            # While there is improvement, add features.
            improvement = True
            while improvement:

                # Enumerate features and shuffle indices.
                feat_enum = np.arange(data.shape[1]) 
                np.random.shuffle(feat_enum)

                # Go over features one-by-one. Add to set if it increases performance.
                for idx, feature_idx in enumerate(feat_enum):
                    
                    # If verbose flag true, print progress.
                    if self.verbose:
                        print(f"Evaluating feature {idx}")
                        print(f"Current feature subset: {self.selected_features}")

                    # Add feature to dataset.
                    data_new = np.hstack((data_new, data[:, feature_idx:feature_idx+1]))

                    # Evaluate performance.
                    acc_new = np.mean(cross_val_score(self.eval_alg, data_new, target, cv=self.cv, n_jobs=-1))

                    # If performance better, keep feature in set and delete it from original dataset.
                    if acc_new > acc_prev:
                        data = np.delete(data, feature_idx, axis=1)
                        
                        # Update performance of previous feature subset.
                        acc_prev = acc_new

                        # Append feature index to list of selected features.
                        self.selected_features.append(feature_idx)
                        break
                    else:
                        # Remove added feature.
                        data_new = data_new[:, :-1]

                else:
                    # If went over all features without finding one that
                    # improves the performance, stop.
                    improvement = False

        if self.mode == 'maximize':
            
            # Initialize array for the dataset with selected features.
            data_new = np.empty((data.shape[0], 0), dtype=data.dtype) 

            # Initialize performance of previous feature subset.
            acc_prev = 0.0

            # Initialize performance of current best new feature and its index.
            acc_max_nxt = 0.0
            to_add_idx = -1
            
            # While there is improvement, add features.
            improvement = True
            while improvement:
                
                # Go over features.
                for feature_idx in np.arange(data.shape[1]):
                    
                    # If verbose flag true, print progress.
                    if self.verbose:
                        print(f"Evaluating feature {feature_idx}/{data.shape[1]}")
                        print(f"Current feature subset: {self.selected_features}")
                    
                    # Add feature to dataset.
                    data_new = np.hstack((data_new, data[:, feature_idx:feature_idx+1]))
                    
                    # Evaluate performance.
                    acc_new = np.mean(cross_val_score(self.eval_alg, data_new, target, cv=self.cv, n_jobs=-1))
                    
                    # Check if increase in performance maximal of currently tried features.
                    if acc_new >= acc_max_nxt:
                        acc_max_nxt = acc_new
                        to_add_idx = feature_idx
                    
                    # Remove added feature.
                    data_new = np.delete(data_new, data_new.shape[1]-1, axis=1)

                
                # If added feature that yielded best performance improved performance
                # over previous feature subset.
                if acc_max_nxt > acc_prev:
                    
                    # Update performance of previous feature subset.
                    acc_prev = acc_max_nxt

                    # Add best feature and delete it from full feature set.
                    data_new = np.hstack((data_new, data[:, to_add_idx:to_add_idx+1]))
                    data = np.delete(data, to_add_idx, axis=1)
                        
                    # Append feature index to list of selected features.
                    self.selected_features.append(to_add_idx)

                    # Reset performance of current best feature and its index.
                    acc_max_nxt = 0.0
                    to_add_idx = -1
                else:
                    improvement = False


    def transform(self, data):
        """
        Perform feature selection using selected features.

        Args:
            data (numpy.ndarray): array of data samples on which to perform feature selection.

        Returns:
            (numpy.ndarray): result of performing feature selection.
        """

        if not self.selected_features:
            raise NotFittedError("This ForwardSearch instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        else:
            return data[:, self.selected_features]

