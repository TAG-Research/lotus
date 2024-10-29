import numpy as np

import lotus


def importance_sampling(
    proxy_scores: list[float],
    sample_percentage: float,
) -> tuple[list[int], list[float]]:
    """Uses importance sampling and returns the list of indices from which to learn cascade thresholds."""

    w = np.sqrt(proxy_scores)
    is_weight = lotus.settings.is_weight
    w = is_weight * w / np.sum(w) + (1 - is_weight) * np.ones((len(proxy_scores))) / len(proxy_scores)
    indices = np.arange(len(proxy_scores))
    sample_size = (int) (sample_percentage * len(proxy_scores))
    sample_indices = np.random.choice(indices, sample_size, p=w)
    correction_factors = (1/len(proxy_scores)) / w

    return sample_indices, correction_factors

def calibrate_llm_logprobs(true_probs: list[float]) -> list[float]:
    """Transforms true probabilities to calibrate LLM proxies."""
    num_quantiles = lotus.settings.num_helper_lm_calibration_quantiles
    quantile_values = np.percentile(true_probs, np.linspace(0, 100, num_quantiles + 1))
    true_probs = ((np.digitize(true_probs, quantile_values) - 1) / num_quantiles)
    true_probs = np.clip(true_probs, 0, 1)
    return true_probs

def learn_cascade_thresholds(
    proxy_scores: list[float],
    oracle_outputs: list[float],
    sample_correction_factors: list[float],
    recall_target: float,
    precision_target: float,
    delta: float
) -> tuple[tuple[float, float], int]:
    """Learns cascade thresholds given targets and proxy scores, 
    oracle outputs over the sample, and correction factors for the
    sample."""

    def UB(mean, std_dev, s, delta):
        return mean + (std_dev / (s ** 0.5)) * ((2 * np.log(1 / delta)) ** 0.5)

    def LB(mean, std_dev, s, delta):
        return mean - (std_dev / (s ** 0.5)) * ((2 * np.log(1 / delta)) ** 0.5)

    def recall(pos_threshold: float, neg_threshold: float, sorted_pairs) -> bool:
        helper_accepted = [x for x in sorted_pairs if x[0] >= pos_threshold or x[0] <= neg_threshold]
        sent_to_oracle = [x for x in sorted_pairs if x[0] < pos_threshold and x[0] > neg_threshold]
        total_correct = sum(pair[1] * pair[2] for pair in sorted_pairs)
        recall = (sum(1 for x in helper_accepted if x[0] >= pos_threshold and x[1]) + sum(x[1] * x[2] for x in sent_to_oracle)) / total_correct
        return recall

    def precision(pos_threshold: float, neg_threshold: float, sorted_pairs) -> bool:
        helper_accepted = [x for x in sorted_pairs if x[0] >= pos_threshold or x[0] <= neg_threshold]
        sent_to_oracle = [x for x in sorted_pairs if pos_threshold > x[0] > neg_threshold]
        oracle_positive = sum(x[1] for x in sent_to_oracle)
        true_positives = sum(1 for x in helper_accepted if x[0] >= pos_threshold and x[1]) + oracle_positive
        predicted_positives = sum(1 for x in helper_accepted if x[0] >= pos_threshold) + oracle_positive
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        return precision

    # Pair helper model probabilities with helper correctness and oracle answer
    paired_data = list(zip(proxy_scores, oracle_outputs, sample_correction_factors))
    sorted_pairs = sorted(paired_data, key=lambda x: x[0], reverse=True)
    sample_size = len(sorted_pairs)

    best_combination = (1,0) # initial tau_+, tau_-

    # Find tau_negative based on recall
    tau_neg_0 = max(x[0] for x in sorted_pairs[::-1] if recall(best_combination[0], x[0], sorted_pairs) >= recall_target)
    best_combination = (best_combination[0], tau_neg_0)

    # Do a statistical correction to get a new target recall
    Z1 = [int(x[1]) * x[2] for x in sorted_pairs if x[0] >= best_combination[1]]
    Z2 = [int(x[1]) * x[2] for x in sorted_pairs if x[0] < best_combination[1]]

    mean_z1 = np.mean(Z1) if Z1 else 0
    std_z1 = np.std(Z1) if Z1 else 0
    mean_z2 = np.mean(Z2) if Z2 else 0
    std_z2 = np.std(Z2) if Z2 else 0

    corrected_recall_target = UB(mean_z1, std_z1, sample_size, delta/2)/(UB(mean_z1, std_z1, sample_size, delta/2) + LB(mean_z2, std_z2, sample_size, delta/2))
    corrected_recall_target = min(1, corrected_recall_target)
    tau_neg_prime = max(x[0] for x in sorted_pairs[::-1] if recall(best_combination[0], x[0], sorted_pairs) >= corrected_recall_target)
    best_combination = (best_combination[0], tau_neg_prime)

    # Do a statistical correction to get a target satisfying precision
    candidate_thresholds = [1]
    for pair in sorted_pairs:
        possible_threshold = pair[0]
        Z = [int(x[1]) for x in sorted_pairs if x[0] >= possible_threshold]
        mean_z = np.mean(Z) if Z else 0
        std_z = np.std(Z) if Z else 0
        p_l = LB(mean_z, std_z, len(Z), delta/len(sorted_pairs))
        if p_l > precision_target:
            candidate_thresholds.append(possible_threshold)

    best_combination = (max(best_combination[1], min(candidate_thresholds)), best_combination[1])
    oracle_calls = sum(1 for x in proxy_scores if best_combination[0] > x > best_combination[1])

    no_correction_sorted_pairs = [tup[:2] + (1,) for tup in sorted_pairs]
    lotus.logger.info(f"Sample recall: {recall(best_combination[0], best_combination[1], no_correction_sorted_pairs)}")
    lotus.logger.info(f"Sample precision: {precision(best_combination[0], best_combination[1], sorted_pairs)}")

    return best_combination, oracle_calls

def calibrate_sem_sim_join(true_score: list[float]) -> list[float]:
    true_score = np.clip(true_score, 0, 1)
    return true_score