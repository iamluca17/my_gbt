import numpy as np

import pandas as pd

class Node():

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, efficiency_metric=None, 
                 value=None, class_probabilities=None, class_logits=None):

        #these parameters are important if the node is decision node 
        self.feature_index = feature_index #each node compares splits for one set of features
        self.threshold = threshold #threshold defines splits and is adjusted to obtain best split - this is the actual learning part
        self.left = left
        self.right = right
        self.efficiency_metric = efficiency_metric #measured by entropy reduction i.e. how much the split reduced entropy is how much information has been gained

        #self.value is used if node is leaf node
        self.value = value

        self.class_probabilities = class_probabilities
        self.class_logits = class_logits



class Decision_Tree():

    def __init__(self, min_samples=2, max_depth=2, mode="classify", thresh_quantile_opt=False, histogram_binning_opt=True, generate_logit_parameterization=False):

        self.root = None

        self.min_samples = min_samples

        #too big of a max_depth can cause overfitting
        self.max_depth = max_depth

        if mode == "classify" or mode == "regression":
            self.mode = mode
        else:
            raise Exception("Invalid mode! Model supports classification or regression functionality only!")
        
        self.thresh_quantile_opt = thresh_quantile_opt

        if self.thresh_quantile_opt == False:
            self.histogram_binning_opt = histogram_binning_opt
        else:
            self.histogram_binning_opt = False

        self.generate_logit_parameterization = generate_logit_parameterization if mode == "classify" else False

        self.class_labels = None
        self.class_labels_index_map = None

    @property
    def get_mode(self):

        if self.mode:
            return self.mode

    def build_tree(self, dataset:pd.DataFrame, curr_depth=0):

        if self.mode=="classify":

            self.class_labels = np.unique(np.array(dataset[:, -1]))  
            self.class_labels_index_map = {label: i for i, label in enumerate(self.class_labels)}

            return self.build_classifier_tree(dataset, curr_depth)
        elif self.mode=="regression":
            return self.build_regression_tree(dataset, curr_depth)


    def build_classifier_tree(self, dataset, curr_depth=0):

        #note dataset is a a pandas dataframe

        features = dataset[:, :-1]
        targets = dataset[:,-1]
        num_samples, num_features = np.shape(features)

        if num_samples >= self.min_samples and curr_depth<= self.max_depth:

            #generate best split
            best_split = self.gen_best_split(dataset, num_samples, num_features, criteria="information gain")

            '''
            if information gain value is less then 0 
            there is no information gain because the data is pure 
            i.e it's of the same class so no more need for splitting
            '''
            if best_split["efficiency_metric"] is not None and best_split["efficiency_metric"] > 0: 
                
                #recursion for left child node
                left_child = self.build_classifier_tree(best_split["left_child_dataset"], curr_depth+1)
                #recursion for right child node
                right_child = self.build_classifier_tree(best_split["right_child_dataset"], curr_depth+1)

                #return node in case node is decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_child, right_child, best_split["efficiency_metric"])

        leaf_value = self.calculate_leaf_value(targets)

        if self.generate_logit_parameterization:
            class_probabilities = self.calculate_class_probabilities(targets)

            epsilon = 1e-8
            class_logits = {cls: np.log((prob + epsilon) / (1 - prob + epsilon)) for cls, prob in class_probabilities.items()}

            return Node(value=leaf_value, class_probabilities=class_probabilities, class_logits=class_logits)
        else:
            return Node(value=leaf_value)
    
    def build_regression_tree(self, dataset, curr_depth=0):

        #note dataset is a a pandas dataframe

        features = dataset[:, :-1]
        targets = dataset[:,-1]
        num_samples, num_features = np.shape(features)

        if num_samples >= self.min_samples and curr_depth<= self.max_depth:

            #generate best split
            best_split = self.gen_best_split(dataset, num_samples, num_features, criteria="variance reduction")

            '''
            if information gain value is less then 0 
            there is no information gain because the data is pure 
            i.e it's of the same class so no more need for splitting
            '''
            if best_split["efficiency_metric"] is not None and best_split["efficiency_metric"] > 0: 
                
                #recursion for left child node
                left_child = self.build_regression_tree(best_split["left_child_dataset"], curr_depth+1)
                #recursion for right child node
                right_child = self.build_regression_tree(best_split["right_child_dataset"], curr_depth+1)

                #return node in case node is decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_child, right_child, best_split["efficiency_metric"])

        leaf_value = self.calculate_leaf_value(targets)

        #return node in case node is leaf node
        return Node(value=leaf_value)

    def calculate_leaf_value(self, leaf_dataset):
        
        if self.mode == "classify":
            #this calculates the leaf value in case of target values of categorical importance
            leaf_dataset = list(leaf_dataset)
            return max(leaf_dataset, key=leaf_dataset.count)
        elif self.mode == "regression":
            return np.mean(leaf_dataset)

    def calculate_class_probabilities(self, leaf_dataset):
        # Convert leaf_dataset to a NumPy array if it isn't already.
            leaf_dataset = np.array(leaf_dataset)

            # Count occurrences of each label in the leaf dataset
            unique, counts = np.unique(leaf_dataset, return_counts=True)
            total = len(leaf_dataset)

            # Initialize probabilities dictionary with all class labels set to 0
            probabilities = {label: 0.0 for label in self.class_labels}

            # Assign computed probabilities to the present class labels
            for label, count in zip(unique, counts):
                probabilities[label] = count / total

            return probabilities


    def gen_best_split(self, dataset, num_samples, num_features, criteria="information gain", bins=256):
        
        

        best_split = {
            "feature_index": None,
            "threshold": None,
            "left_child_dataset": None,
            "right_child_dataset": None,
            "efficiency_metric": None
        }
        #max_info_gain is first initialized at -infinity
        max_metric_val = -float("inf")
        
        '''
        Check all possible thresholds and feature combinations 
        by looping through all feature sets and all thresholds in each feature set
        '''

        bin_edges = {}

        for feature_index in range(num_features):

            feature_values = dataset[:, feature_index]
            #possible thresholds can have values only from the feature set at the current feature index
            possible_thresholds  = np.unique(feature_values)

            if self.histogram_binning_opt:
                
                if feature_index not in bin_edges:
                    # Compute histogram bin edges only once per feature
                    bin_edges[feature_index] = np.quantile(feature_values, np.linspace(0, 1, bins + 1))
                
                # Use bin midpoints as possible thresholds
                bin_midpoints = (bin_edges[feature_index][:-1] + bin_edges[feature_index][1:]) / 2
                possible_thresholds = bin_midpoints
            
            elif self.thresh_quantile_opt:
                possible_thresholds = np.quantile(possible_thresholds, np.linspace(0, 1, 20))


            for threshold in possible_thresholds:

                #split according to threshold
                left_child_dataset, right_child_dataset = self.split(dataset, feature_index, threshold)

                if left_child_dataset is None or right_child_dataset is None:
                    continue

                if len(left_child_dataset) > 0 and len(right_child_dataset) > 0:

                    #get the outcome targets for each sample
                    parent_outcome_target = dataset[:, -1]
                    left_child_outcome_target = left_child_dataset[:, -1]
                    right_child_outcome_target = right_child_dataset[:, -1]

                    if criteria=="information gain":
                        split_efficiency_metric = self.info_gain(parent_outcome_target, left_child_outcome_target, right_child_outcome_target, "entropy")
                    elif criteria=="variance reduction":
                        split_efficiency_metric = self.variance_reduction(parent_outcome_target, left_child_outcome_target, right_child_outcome_target)

                    #compare split metric
                    if split_efficiency_metric > max_metric_val:

                        #save best slit information
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_child_dataset"] = left_child_dataset
                        best_split["right_child_dataset"] = right_child_dataset
                        best_split["efficiency_metric"] = split_efficiency_metric

                        max_metric_val = split_efficiency_metric
                    

        return best_split


    def split(self, dataset, feature_index, threshold):

        sub_threshold_dataset = np.array([row for row in dataset if row[feature_index] <= threshold])
        supra_threshold_dataset = np.array([row for row in dataset if row[feature_index] > threshold])

        if len(sub_threshold_dataset) == 0 or len(supra_threshold_dataset) == 0:
            return None, None  # Return None to indicate no split was made

        return sub_threshold_dataset, supra_threshold_dataset


    #compute information gain for from parent to children after split as entropy or gini index difference
    def info_gain(self, parent_dataset_outcomes, left_child_dataset_outcomes, right_child_dataset_outcomes, mode="entropy"):

        weight_left = len(left_child_dataset_outcomes) / len(parent_dataset_outcomes)
        weight_right = len(right_child_dataset_outcomes) / len(parent_dataset_outcomes)

        if mode=="entropy":
            information_gain = self.entropy(parent_dataset_outcomes) - (weight_left*self.entropy(left_child_dataset_outcomes) + weight_right*self.entropy(right_child_dataset_outcomes))
        elif mode=="gini":
            information_gain = self.gini_index(parent_dataset_outcomes) - (weight_left*self.gini_index(left_child_dataset_outcomes) + weight_right*self.gini_index(right_child_dataset_outcomes))
        
        return information_gain

    def variance_reduction(self, parent_dataset_outcomes, left_child_dataset_outcomes, right_child_dataset_outcomes):

        weight_left = len(left_child_dataset_outcomes) / len(parent_dataset_outcomes)
        weight_right = len(right_child_dataset_outcomes) / len(parent_dataset_outcomes)

        variance_reduction = np.var(parent_dataset_outcomes, ddof=0) - (
        weight_left * np.var(left_child_dataset_outcomes, ddof=0) + 
        weight_right * np.var(right_child_dataset_outcomes, ddof=0)
        )

        return variance_reduction
    

    #compute entropy
    '''
    Entropy Formula:
     
    H(S) = - Σ p(c) * log2(p(c))
    
    Where:
        H(S) is the entropy of the set S,
        p(c) is the probability of class c in the set S,
        log2(p(c)) is the binary logarithm of the probability of class c.
        -log2(p(c)) calculates numbers of bits describing the probability of class c
    '''
    def entropy(self, dataset_outcomes):

        class_labels = np.unique(dataset_outcomes)

        entropy = 0
        for class_label in class_labels:
            p_class = len(dataset_outcomes[dataset_outcomes==class_label])/len(dataset_outcomes)
            entropy += -p_class * np.log2(p_class)

        return entropy

    '''
    gini index is computed faster than entropy which uses logarithms so it's an alternative for optimizing computation time
    '''
    def gini_index(self, dataset_outcomes):

        class_labels = np.unique(dataset_outcomes)
        gini = 0

        for class_label in class_labels:
            p_class = len(dataset_outcomes[dataset_outcomes==class_label])/len(dataset_outcomes)
            gini += p_class**2
        
        return 1 - gini

    def variance(self, dataset_outcomes):

        dataset_outcomes = np.array(dataset_outcomes)

        if len(dataset_outcomes) == 0:
            return 0
        
        outcome_mean = self.sample_mean(dataset_outcomes)
        variance = sum((dataset_outcomes - outcome_mean)**2) / len(dataset_outcomes)
        
        return variance


    def sample_mean(self, sample_values)->float:

        sample_values = np.array(sample_values)

        if len(sample_values) > 0:
            return sum(sample_values) / len(sample_values)
        
        return 0

    #trains the model
    def fit(self, X, Y):

        dataset = np.concatenate((X,Y), axis = 1)
        if isinstance(dataset, pd.DataFrame):
            dataset = np.array(dataset)
            
        self.root = self.build_tree(dataset)

    #generates outcome vector from feature matrix
    def predict_from_dataset(self, X, output_type=None):
        
        predictions = np.array([self.make_predictions(x, self.root, output_type) for x in X])

        return predictions
    
    #generates an out come for a sample features vector
    def make_predictions(self, x, tree:Node, output_type=None):

        if self.generate_logit_parameterization and output_type == "logit":
            if tree.class_logits:
                
                sample_logits_array = np.zeros(len(self.class_labels))
                for label, logit in tree.class_logits.items():
                    sample_logits_array[self.class_labels_index_map[label]] = logit

                return sample_logits_array
            
        elif self.generate_logit_parameterization and output_type=="probabilities":
            if tree.class_probabilities:
                return tree.class_probabilities
        else:
            if tree.value is not None:
                return tree.value
        
        if x[tree.feature_index] > tree.threshold:
            return self.make_predictions(x, tree.right, output_type=output_type)
        elif x[tree.feature_index] <= tree.threshold:
            return self.make_predictions(x, tree.left, output_type=output_type)


    #prints conceptual representaion of decision nodes and leaf nodes
    def print_tree(self, tree:Node=None, depth=0):

        if not tree:
            tree = self.root

        indent = "   " * depth

        if tree.value is not None:
            print(f"{indent}Leaf: {tree.value}")
        else:
            print(f"{indent}Node at depth {depth}: if x{tree.feature_index} <= {tree.threshold}:     (Metric = {tree.efficiency_metric})")
            self.print_tree(tree.left, depth+1)
            print(f"{indent}Node at depth {depth}: else if x{tree.feature_index} > {tree.threshold}:     (Metric = {tree.efficiency_metric})")
            self.print_tree(tree.right, depth+1)



if __name__ == "__main__":

    col_names = ['variance', 'skewness', 'curtosis', 'entropy', 'type']
    banknote_dataset = pd.read_csv("D:\dev\Pred_&_ML_tutorial_model_building\src\\resources\data_banknote_authentication\data_banknote_authentication.txt",
                               skiprows=1, delimiter=",", header=None, names=col_names)

    banknote_dataset = banknote_dataset.sample(frac=1).reset_index(drop=True)

    from sklearn.model_selection import train_test_split

    features = banknote_dataset.iloc[:, :-1].values
    targets = banknote_dataset.iloc[:, -1].values.reshape(-1,1)

    TRAIN_DATASET_FEATURES, TEST_DATASET_FEATURES, TRAIN_DATASET_OUTCOMES, TEST_DATASET_OUTCOMES = train_test_split(features, targets, test_size=.2,random_state=42)

    
    classifier = Decision_Tree(min_samples=3, max_depth=5, mode="classify", generate_logit_parameterization=True)

    classifier.fit(TRAIN_DATASET_FEATURES, TRAIN_DATASET_OUTCOMES)
    classifier.print_tree()
