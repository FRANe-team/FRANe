# the FRANe algorithm -> skrlj, petkovic & primozic, 2020

import numpy as np
import tqdm
import logging
from scipy.spatial.distance import pdist, squareform
from .thresholds import threshold_dict
from .helpers import normalize, scipy_metrices

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# constants
VERBOSE_LEVEL_NOTHING = 0
VERBOSE_LEVEL_BAR = 1
VERBOSE_LEVEL_VERBOSE = 2


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
                number of different edge-weight thresholds

            metric : func or str
                Metric (or its name for the predefined ones) to calculate distance. If given as a function f,
                it should have the signature
                f(x1: np.ndarray, x2: np.ndarry) --> float
                where x1 and x2 both have the shape (m, ) where m is the number of examples, i.e., x1 and x2 are
                feature vectors.

            threshold_function : func or str
                Function (or its name for the predefined ones) that calculates different edge-weight thresholds.
                If given as a function f, it should have the signature
                f(
                    min_value: float,
                    max_value: float,
                    iterations: int,
                    distances: np.ndarray
                ) -> np.ndarray
                where
                - distances is a 2D array of shape (n, n), where n is the number of features in the data.
                - the returned array is an 1D of shape (iterations, ) and contains the thresholds used by FRANE.

            min_edge_threshold int
                The minimal average degree

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
        if isinstance(metric, str):
            if metric in scipy_metrices.keys():
                metric = scipy_metrices[metric]
            else:
                raise ValueError(
                    f"Unknown metric ({metric}). "
                    f"Use one of the following: {list(scipy_metrices.keys())} "
                    f"or define your own metric as specified in the docstring."
                )
        if isinstance(threshold_function, str):
            if threshold_function in threshold_dict.keys():
                threshold_function = threshold_dict[threshold_function]
            else:
                raise ValueError(
                    f"Unknown metric ({threshold_function}). "
                    f"Use one of the following: {list(threshold_dict.keys())} "
                    f"or define your own metric as specified in the docstring."
                )

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
        """
        Parameters
        ----------
            importance_scores : np.ndarray or list
                The feature importance scores.

        Returns
        ---------

        Quality of the scores, based on their spread.
        """
        eps = 10 ** -10
        modified = [max(score, eps) for score in importance_scores]
        modified.sort(reverse=True)
        if len(importance_scores) <= 2:
            return modified[0] / modified[-1]
        else:
            # median of the top and bottom three scores
            return modified[1] / modified[-2]

    def fit(self, data, transpose=True, verbose=VERBOSE_LEVEL_NOTHING):
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

        if transpose:
            data = np.transpose(data)
        n_features, n_examples = data.shape

        # normalize
        for i in range(n_features):
            data[i] = normalize(data[i], self.distance_metric)

            if self.distance_metric == scipy_metrices["correlation"]:
                # take care of constant features
                i_min = np.min(data[i])
                i_max = np.max(data[i])
                if i_min == i_max:
                    # something that is not correlated to anything
                    data[i] = np.random.rand(n_examples)

        if verbose in [VERBOSE_LEVEL_VERBOSE, VERBOSE_LEVEL_BAR]:
            logger.info("Calculating distances..")
        # calculates distances between features
        distances = squareform(pdist(data, self.distance_metric))
        if np.isnan(distances).any():
            raise ValueError("Distances must not be nan!")

        if np.min(distances, axis=None) < 0.0:
            raise ValueError("Distances must be non-negative")

        distances_copy = distances.copy()
        d_max = np.max(distances, axis=None)

        # trick for non zero min
        distances_copy[distances_copy == 0.0] = d_max + 1.0
        d_min = np.min(distances_copy, axis=None)
        thresholds = self.threshold_function(d_min, d_max, self.iterations, distances)

        # FRANe iterations
        solutions = []
        show_progress = verbose == VERBOSE_LEVEL_BAR
        show_message = verbose == VERBOSE_LEVEL_VERBOSE
        for threshold in tqdm.tqdm(thresholds, total=len(thresholds), disable=(not show_progress)):
            if show_message:
                logger.info(f"Starting iteration with threshold {threshold}.")
            distances_copy = distances.copy()

            # Weights on the edges: d_max - distance
            # To ignore the edges with distance > threshold:
            distances_copy[distances > threshold] = d_max

            if self.page_rank_no_weights:
                # Change edge weights to 0 or 1
                distances_copy[distances_copy <= threshold] = 1.0
            weights = d_max - distances_copy

            # ignore self-similarities of the nodes
            np.fill_diagonal(weights, 0.0)
            degrees = np.sum(weights, axis=0)
            n_edges = np.sum(weights > 0.0)
            not_enough_edges = n_edges < self.min_edge_threshold * n_features

            if not_enough_edges and not self.save_all_scores:
                continue
            if show_message:
                logger.info(
                    f"Generated a graph (V, E), with |V| = {n_features} and |E| = {n_edges}."
                )
            degrees[degrees == 0.0] = 1.0  # does not matter what

            # define matrix and vector from the page rank iteration
            matrix = self.page_rank_decay * (weights / degrees)
            del weights
            vector = (1 - self.page_rank_decay) / n_features  # effectively a vector
            solution = np.ones(n_features) / n_features
            solution = solution / np.sum(solution)
            iterations = 0
            eps = 10 ** -10
            converged = False
            while iterations < self.page_rank_iterations:
                iterations += 1
                previous = solution.copy()
                solution = matrix.dot(solution)
                solution = solution + vector
                if np.max(np.abs(solution - previous)) < eps:
                    converged = True
                    break
            if show_message:
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
            self.meta_data = [
                (solution[0], solution[2]) for solution in solutions
            ]
        else:
            # we want as large spread as possible
            solutions.sort(key=lambda triplet: -triplet[0])
            if not solutions:
                if verbose != VERBOSE_LEVEL_NOTHING:
                    logger.error("No feature rankings!")
                self.feature_importances_ = np.ones(n_features)
            else:
                self.feature_importances_ = solutions[0][-1]
        return self
