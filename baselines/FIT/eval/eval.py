import csv
from collections import defaultdict
from sklearn.metrics import ndcg_score


# write a function to read csv file
def read_csv_file(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[1:]

def collect_morph(data):
    morph = defaultdict(list)
    for line in data:
        source = line[0]
        morph[source].append({'prediction': line[1], 'target': line[2], 'loss': line[3]})

    sorted_morph = morph.copy()
    unsorted_morph = morph.copy()
    for key in sorted_morph:
        sorted_morph[key] = sorted(sorted_morph[key], key=lambda x: x['loss'])
    return sorted_morph, unsorted_morph

def calc_precision_k(morph, k):
    correct = 0
    total = 0
    for key in morph:
        for i in range(k):
            if morph[key][i]['prediction'] == morph[key][i]['target']:
                correct += 1
        total += k
    print('precision@{}: {}'.format(k, correct / total))
    return correct / total


def mean_reciprocal_rank(y_true, y_pred):
    """
    Calculate the mean reciprocal rank (MRR) for a given set of true and predicted values.
    
    Args:
    - y_true: list of tuples (category, rank) representing the true values, where rank is an integer from 1 to n.
    - y_pred: list of tuples (category, rank) representing the predicted values, where rank is an integer from 1 to n.
    
    Returns:
    - mrr: mean reciprocal rank (MRR) score.
    """
    
    # Create a dictionary to map categories to their ranks in y_true
    true_ranks = dict(y_true)
    
    # Initialize a variable to store the sum of reciprocal ranks
    sum_ranks = 0
    
    # Loop over the predicted values
    for category, rank in y_pred:
        # If the category is in y_true, add the reciprocal of its rank to the sum
        if category in true_ranks:
            sum_ranks += 1 / true_ranks[category]
    
    # Calculate the MRR score as the average of the reciprocal ranks
    mrr = sum_ranks / len(y_pred) if len(y_pred) > 0 else 0
    
    return mrr


def convert_order_into_rank(pred_order, true_order):
    pred_order_indices = {}
    true_order_indices = {}
    for key in true_order.keys():
        # only keep loss 
        true_loss = [i['loss'] for i in true_order[key]]
        pred_loss = [i['loss'] for i in pred_order[key]]
        mapping = {v: i for i, v in enumerate(true_loss)}
        true_order_indices[key] = [mapping[i] for i in true_loss]
        pred_order_indices[key] = [mapping[i] for i in pred_loss]
    return pred_order_indices, true_order_indices


if __name__ == '__main__':
    data = read_csv_file('./neural_baseline/test_res.test.csv')
    sorted_morph, unsorted_morph = collect_morph(data)
    precision_k = calc_precision_k(sorted_morph, 1)

    rank_pred, rank_true = convert_order_into_rank(sorted_morph, unsorted_morph)
    mrr = {}
    ndcg = {}
    for key in rank_pred:
        mrr[key] = mean_reciprocal_rank(rank_true[key], rank_pred[key])
        ndcg[key] = ndcg_score(rank_true[key], rank_pred[key], k=1)
    
    mrr = sum(mrr.values()) / len(mrr)
    ndcg = sum(ndcg.values()) / len(ndcg)
    print('mrr: {}'.format(mrr))
    print('ndcg: {}'.format(ndcg))