# the FRANe algorithm -> skrlj, petkovic & primozic, 2020

import numpy as np
import tqdm
import logging
from scipy.spatial.distance import pdist, squareform
from .thresholds import threshold_dict
from .helpers import normalize, convert_metrices

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# constants
VERBOSE_LEVEL_NOTHING = 0
VERBOSE_LEVEL_BAR = 1
VERBOSE_LEVEL_VERBOSE = 2


def set_value(value, default):
    return default if value is None else value


class FRANe:
    def __init__(self,
                 iterations=100,
                 metric='correlation',
                 min_edge_threshold=1.0,
                 threshold_function='geomspace',
                 page_rank_decay=0.85,
                 page_rank_iterations=1000,
                 page_rank_no_weights=False,
                 save_all_scores=False):
        """
        Initialize FRANe object.

        Parameters
        ----------

            iterations int
                number of iterations

            metric : func or str
                Metric to calculate distance

            threshold_function - func or str  TODO ab šlo to stran?
            threshold_function(min : float, max : float, iterations : int, distances : ndarray, sorted_indexes : ndarray) 
                Should return ndarray of thresholds for FRANe network. TODO dopiš

            min_edge_threshold int TODO a naj gre stran?

            page_rank_decay float
                Page rank decay factor

            page_rank_iterations int
                Number of iterations for solving the fixed point of page rank iteration

            page_rank_no_weights bool
                If True, the weights of the edges are ignored (i.e., all equal 1) and the standard
                page rank score is computed. Otherwise, the transition probabilities in the formula
                are proportional to the weights.

            page_rank_iterations int
                Maximal number of iterations when solving the page rank equation.

            save_all_scores bool
                If set to True, the scores from every fit_page_rank iteration are stored.
                Otherwise, only the best (according to the max span heuristic) are.
        """

        # check if metric ot threshold function is known
        if metric in convert_metrices.keys():
            metric = convert_metrices[metric]

        if threshold_function in threshold_dict.keys():
            threshold_function = threshold_dict[threshold_function]

        # set variables
        self.iterations = iterations
        self.distance_metric = metric
        self.min_edge_threshold = min_edge_threshold
        self.threshold_function = threshold_function
        self.page_rank_decay = page_rank_decay
        self.page_rank_no_weights = page_rank_no_weights
        self.page_rank_iterations = page_rank_iterations
        self.save_all_scores = save_all_scores
        self.feature_importances_ = None
        self.meta_data = None

    @staticmethod
    def ranking_quality_heuristic(importance_scores):
        '''      
        Returns
        ---------

        Quality of the scores.
        '''
        eps = 10**-10
        modified = [max(score, eps) for score in importance_scores]
        modified.sort(reverse=True)
        if len(importance_scores) <= 2:
            return modified[0] / modified[-1]

        else:

            # median of the top and bottom three scores
            return modified[1] / modified[-2]

    def fit(self, data, transpose=True, verbose=0):
        """
        Computes feature ranking.

        Parameters
        ----------sorted
        data np.ndarray
            Data matrix, data[i, j] is the value of j-th feature for i-th example if transpose,
            and is the value of i-th feature for j-th example if not transpose

        transpose bool
            Set this to True if and only if the columns in the data correspond to features
            (and rows to the examples).

        verbose int
            Logging level\n
            VERBOSE_LEVEL_NOTHING - no output\n
            VERBOSE_LEVEL_BAR - show iteration progress bar\n
            VERBOSE_LEVEL_VERBOSE - logs results of every iteration\n
            other - no output

        Returns
        -------
            self
        """
        return self.fit_page_rank(data, transpose=transpose, verbose=verbose)

    def fit_page_rank(self, data, transpose=True, verbose=0):
        """

        Parameters
        ----------
        data np.ndarray
            Data matrix, data[i, j] is the value of j-th feature for i-th example if transpose,
            and is the value of i-th feature for j-th example if not transpose

        transpose bool
            Set this to True if and only if the columns in the data correspond to features
            (and rows to the examples).

        verbose int
            Logging level\n
            VERBOSE_LEVEL_NOTHING - no output\n
            VERBOSE_LEVEL_BAR - show iteration progress bar\n
            VERBOSE_LEVEL_VERBOSE - logs results of every iteration\n
            other - no output

        Returns
        -------
            self
        """
        if transpose:
            data = np.transpose(data)
        n_features, n_examples = data.shape

        # normalize
        for i in range(n_features):
            data[i] = normalize(data[i], self.distance_metric)

            if self.distance_metric == convert_metrices["correlation"]:
                i_min = np.min(data[i])
                i_max = np.max(data[i])
                if i_min == i_max:
                    # something that is not correlated to anything
                    data[i] = np.random.rand(n_examples)

        if verbose in [VERBOSE_LEVEL_VERBOSE, VERBOSE_LEVEL_BAR] :
            logger.info("Calculating distances..")
        # calculates distances between features
        distances = squareform(pdist(data, self.distance_metric))
        if np.isnan(distances).any():
            raise ValueError("Distances must not be nan!")

        if np.min(distances, axis=None) < 0.0:
            raise ValueError("Distances must be non-negative")

        distances_copy = distances.copy()
        sorted_indexes = np.argsort(distances, axis=None)
        d_max = np.max(distances, axis=None)

        # trick for non zero min
        distances_copy[distances_copy == 0.0] = d_max + 1.0
        d_min = np.min(distances_copy, axis=None)
        thresholds = self.threshold_function(d_min, d_max, self.iterations,
                                             distances, sorted_indexes)

        # FRANe iterations
        solutions = []
        for threshold in tqdm.tqdm(thresholds, total=len(thresholds), colour='green', disable=(not verbose==VERBOSE_LEVEL_BAR)):
            distances_copy = distances.copy()

            # Weights on the edges: d_max - distance
            # To ignore the edges with distance > threshold:
            distances_copy[distances > threshold] = d_max

            if self.page_rank_no_weights:

                # Make everything 1, so that all the weights are also (zero or) one
                distances_copy[distances_copy <= threshold] = 1.0
            weights = d_max - distances_copy

            # ignore self-similarities of the node
            np.fill_diagonal(weights, 0.0)
            degrees = np.sum(weights, axis=0)
            n_edges = np.sum(weights > 0.0)
            not_enough_edges = n_edges < self.min_edge_threshold * n_features

            if not_enough_edges and not self.save_all_scores:
                continue
            if verbose == VERBOSE_LEVEL_VERBOSE:
                logger.info(
                    f"Generated a |G| = {n_features} and |E| = {n_edges} graph.")
            degrees[degrees == 0.0] = 1.0  # does not matter what

            # define matrix and vector from the page rank iteration
            matrix = self.page_rank_decay * (weights / degrees)
            del weights
            vector = (1 - self.page_rank_decay) / \
                n_features  # effectively a vector

            solution = np.ones(n_features) / n_features
            solution = solution / np.sum(solution)
            iterations = 0
            eps = 10**-10
            converged = False

            while iterations < self.page_rank_iterations:
                iterations += 1
                previous = solution.copy()
                solution = matrix.dot(solution)
                solution = solution + vector
                if np.max(np.abs(solution - previous)) < eps:
                    converged = True
                    break
            if verbose == VERBOSE_LEVEL_VERBOSE:
                if converged:
                    logger.info(
                        f"Procedure has converged after {iterations} iterations.")
                else:
                    logger.warning(
                        f"Procedure has not converged after {iterations} iterations."
                    )

            quality = FRANe.ranking_quality_heuristic(solution)
            solutions.append((quality, threshold, not_enough_edges, solution))

        if self.save_all_scores:

            self.feature_importances_ = [
                solution[-1] for solution in solutions
            ]

            self.meta_data = [(solution[0], solution[2])
                              for solution in solutions]
        else:

            # we want as large spread as possible
            solutions.sort(key=lambda triplet: -triplet[0])
            if not solutions:
                if verbose == VERBOSE_LEVEL_VERBOSE:
                    logger.error("No feature rankings!")
                self.feature_importances_ = np.ones(n_features)

            else:
                self.feature_importances_ = solutions[0][-1]

        return self
