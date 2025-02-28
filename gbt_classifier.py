import numpy as np
import decision_tree as dt  # Assuming this contains a decision tree class

import pandas as pd

class My_GBT_CLS:
    def __init__(self, loss_function="", n_estimators=100, learning_rate=1, **learner_constructor_kwargs):
        self.learner_constructor_args = learner_constructor_kwargs
        self.loss_function = loss_function
        self.n_estimators = n_estimators


        self.initial_classifier = None
        self.gd_models_with_alphas = []  # Store (model, alpha) pairs

        self.learning_rate = learning_rate

        self.loss_function="log-loss"

    def softmax(self, logits_matrix):
        """
        Compute softmax probabilities from an array of logits.

        Parameters:
            logits (np.array): A 2D NumPy array of raw logits, where each row corresponds to the logits for a sample.

        Returns:
            np.array: A 2D NumPy array of probabilities, same shape as logits.
        """
        # Subtract the maximum logit for numerical stability (across each row)
        shifted_logits = logits_matrix - np.max(logits_matrix, axis=1, keepdims=True)
        
        # Exponentiate the shifted logits
        exp_logits = np.exp(shifted_logits)
        
        # Compute probabilities by normalizing across each row
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probabilities

    
    def one_hot_encode(self, labels, class_labels):
        """
        Convert a 1D array of class labels into a one-hot encoded 2D matrix.
        
        Parameters:
            labels (np.array): 1D NumPy array of class labels.
            class_labels (np.array): 1D array of all possible class labels.
        
        Returns:
            np.array: 2D NumPy array where each row is a one-hot encoded vector.
        """
        num_classes = len(class_labels)
        one_hot_matrix = np.zeros((len(labels), num_classes))
        
        for i, label in enumerate(labels):
            class_index = np.where(class_labels == label)[0][0]  # Find index of the label
            one_hot_matrix[i, class_index] = 1  # Set the corresponding position to 1
        
        return one_hot_matrix

    def gradient(self, predictions, one_hot_targets):
        
            if self.loss_function == "log-loss":
                if predictions.shape != one_hot_targets.shape:
                    raise ValueError(f"Shape mismatch: predictions shape {predictions.shape} != one_hot_targets shape {one_hot_targets.shape}")
                
                return predictions - one_hot_targets  # Element-wise subtraction
            else:
                raise ValueError("Unsupported loss function")


    def gen_gradient_fitted_learner(self, X, gradient_vector):
        """ Fit a weak learner to the negative gradient (residuals) """
        learner = dt.Decision_Tree(max_depth=15, mode="regression", thresh_quantile_opt=False, histogram_binning_opt=False)
        learner.fit(X, gradient_vector)
        return learner

    def fit(self, X, Y):
        """ Train the gradient boosting model """
        classification_tree = dt.Decision_Tree(max_depth=9, mode="classify", generate_logit_parameterization=True)
        classification_tree.fit(X=X, Y=Y)
        
        self.initial_classifier = classification_tree

        n_samples = len(Y)
        class_labels = np.unique(np.array(Y))
        number_of_class_labels = len(class_labels)

        one_hot_outcomes_matrix = self.one_hot_encode(Y, class_labels)

        predictions = classification_tree.predict_from_dataset(X=X, output_type="logit")

        for t in range(self.n_estimators):

            predicted_class_prob_distrib = self.softmax(predictions)

            gradient = self.gradient(predicted_class_prob_distrib, one_hot_outcomes_matrix)

            gradient = -gradient #make sure we're doing gradient descent not ascent p.s. i learned the hard way:)))))))))))

            for k in range(number_of_class_labels):

                per_class_gradient_vector = gradient[:,k] #here it's a row vector 1d list
                per_class_gradient_vector_transposed = per_class_gradient_vector.reshape(-1, 1) #now it's a column vector 2d array

                learner = self.gen_gradient_fitted_learner(X, per_class_gradient_vector_transposed)

                h_t = learner.predict_from_dataset(X)
                h_t = np.array(h_t) #it's a 1d list row vector
                h_t_transposed = h_t.reshape(-1,1)

                # Compute optimal alpha_t
                numerator = h_t @ per_class_gradient_vector_transposed
                denominator = h_t @ h_t_transposed
                alpha_t = numerator / denominator if denominator != 0 else 0  # Avoid div by zero

                #Generate row selector permutation matrix

                P_t_k = np.zeros((1, number_of_class_labels))  # Row vector with n columns
                P_t_k[0, k] = 1

                alpha_t_matrix = P_t_k * alpha_t

                # Store (learner, alpha)
                self.gd_models_with_alphas.append((learner, alpha_t_matrix))

                # Update predictions

                predictions += self.learning_rate * (h_t_transposed @ alpha_t_matrix)

    def predict(self, X):
        """ Predict using the trained model """
        predictions = self.initial_classifier.predict_from_dataset(X=X, output_type='logit')
        for model, alpha_t_matrix in self.gd_models_with_alphas:
            prediction =  np.array(model.predict_from_dataset(X))
            prediction_transposed = prediction.reshape(-1,1)

            predictions += prediction_transposed @ alpha_t_matrix

        predictions = self.softmax(predictions)
        # Get the index of the max probability for each row
        max_indices = np.argmax(predictions, axis=1)  

        # Map indices back to class labels using the dictionary
        index_to_class = {v: k for k, v in self.initial_classifier.class_labels_index_map.items()}
        predicted_labels = np.array([index_to_class[idx] for idx in max_indices]) #it's in the same order as predictions

        return predicted_labels 

if __name__ == "__main__":

    pass
    
    