import os
import numpy as np
import pandas as pd
import re
import argparse


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data


#这个函数的作用是生成校准标签集。



#计算s(x)分数
def compute_partial_label_scores(val_data_1, val_data_2):
    S_PLL = 0
    noise_label = val_data_1[1]
    score_1 = val_data_1[2]
    score_2 = val_data_2[2]
    for j in range(len(score_1)):
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if score_2[j] >= min(max(score_1), score_1[noise_label]):
        #if score_2[j] >= max(score_1):
        #if score_2[j] >= score_1[noise_label]:
            S_PLL += score_2 [j]
    return S_PLL


#分位数
def compute_quantile(S_PLL_scores, epsilon):
    n = len(S_PLL_scores)
    quantile_index = int(np.ceil((n + 1) * (1 - epsilon)))
    Q_PLL = np.sort(S_PLL_scores)[quantile_index - 1]
    return Q_PLL

#取出比分位数高的样本
def generate_prediction_set(test_data_1, test_data_2, Q_PLL):
    score = test_data_2[2]
    noise_label = test_data_2[1]
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_score = min(max(score),score[noise_label])  # Use the maximum score from candidate labels
    #y_score = max(score)  # Use the maximum score from candidate labels
    #y_score = score[noise_label]  # Use the maximum score from candidate labels
    score = np.array(score, dtype=float)  # list -> numpy array
    filtered_indices = np.where(score >= y_score)[0]  # Get the original indices of the filtered scores
    filtered_scores = score[filtered_indices]  # Get the filtered scores based on the original indices

    sorted_indices = np.argsort(filtered_scores)  # Sort indices of the filtered scores in ascending order
    sorted_filtered_scores = filtered_scores[sorted_indices]
    sorted_filtered_indices = filtered_indices[
        sorted_indices]  # Get the sorted original indices based on the filtered scores

    # Compute cumulative sum starting with the highest scores included
    cumulative_sum = np.cumsum(sorted_filtered_scores[::-1])  # Start with all scores summed in descending order
    C_epsilon_PLL = list(sorted_filtered_indices[::-1])  # Reverse to match cumulative sum order

    while len(cumulative_sum) > 1 and cumulative_sum[-1] > Q_PLL:
        cumulative_sum = cumulative_sum[:-1]
        C_epsilon_PLL = C_epsilon_PLL[:-1]

    return C_epsilon_PLL


def compute_accuracy(test_data2, prediction_sets):
    correct_predictions = 0
    total_predictions = len(test_data2)

    for i, (y_true, noise_label, score) in enumerate(test_data2):
        if y_true in prediction_sets[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def CP_PLL_algorithm(val_data1, val_data2, test_data1, test_data2, epsilon):
    S_PLL_scores = []
    # Compute partial label scores for validation data
    for i in range(len(val_data1)):
        val_data_1 = val_data1[i]
        val_data_2 = val_data2[i]
        S_PLL_scores.append(compute_partial_label_scores(val_data_1, val_data_2))
    # Compute quantile value
    Q_PLL = compute_quantile(S_PLL_scores, epsilon)

    prediction_sets = []
    for i in range(len(test_data2)):
        test_data_2 = test_data2[i]
        test_data_1 = test_data1[i]
        prediction_sets.append(generate_prediction_set(test_data_1, test_data_2, Q_PLL))
    # Compute quantile value

    # Compute accuracy
    accuracy = compute_accuracy(test_data2, prediction_sets)

    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])

    return Q_PLL, avg_set_size, accuracy


def process_datasets(base_path, net, epsilon, partial_rate=0.1):
    results = []
    # scores1 & scores2 路径
    scores1_val = os.path.join(base_path, "scores1", "val", f"{net}_scores.npy")
    scores1_test = os.path.join(base_path, "scores1", "test", f"{net}_scores.npy")
    scores2_val = os.path.join(base_path, "scores2", "val", f"{net}_scores.npy")
    scores2_test = os.path.join(base_path, "scores2", "test", f"{net}_scores.npy")

    # 加载数据
    val_data1 = load_data(scores1_val)
    test_data1 = load_data(scores1_test)
    val_data2 = load_data(scores2_val)
    test_data2 = load_data(scores2_test)

    Q_PLL, avg_set_size, accuracy = CP_PLL_algorithm(val_data1, val_data2, test_data1, test_data2, epsilon)

    results.append({
        #'Test File': corresponding_test_file,
        'Quantile Value': Q_PLL,
        'Average Prediction Set Size': avg_set_size,
        'Accuracy': accuracy
    })

    # Write results to Excel file
    output_dir = os.path.join(base_path, 'evaluation_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_output = pd.DataFrame(results)
    output_file_path = os.path.join(output_dir, 'evaluation_results.xlsx')
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False)

    print(f"All results saved to '{output_file_path}'.")


def main():
    parser = argparse.ArgumentParser(description='Process datasets for CP-PLL algorithm.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Specify the base path containing the files to process.')
    parser.add_argument('--net', type=str, required=True,
                        help='Specify the base path containing the files to process.')
    parser.add_argument('--epsilon', type=float, required=True, help='Specify the epsilon value for the algorithm.')
    parser.add_argument('--partial_rate', type=float, default=0.1,
                         help='Specify the partial rate for generating candidate labels.')

    args = parser.parse_args()

    process_datasets(args.base_path, args.net, args.epsilon, args.partial_rate)


if __name__ == '__main__':
    main()
