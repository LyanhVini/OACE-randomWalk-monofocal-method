"""
Validação dos modelos    
"""
from main import *

def summarize_best_average_worst(oace_metrics_per_iteration):
    all_results = []

    # Percorre todas as iterações e coleta as informações relevantes
    for iteration, metrics_dict in oace_metrics_per_iteration.items():
        all_results.append({
            'Iteration': iteration,
            'Model Name': metrics_dict["model_name"],
            'A': metrics_dict["A"],
            'C': metrics_dict["C"],
            'Score': metrics_dict["Score"],
            'Solution': metrics_dict["solution"]
        })
    
    # Ordena os resultados pelo Score
    all_results = sorted(all_results, key=lambda x: x['Score'])

    # Identifica o melhor, a média e o pior
    best_result = all_results[-1]  # Último da lista (maior Score)
    worst_result = all_results[0]  # Primeiro da lista (menor Score)
    
    # Calcula o resultado mediano
    mid_index = len(all_results) // 2
    average_result = all_results[mid_index] if len(all_results) % 2 != 0 else {
        'Iteration': 'Average of two middle iterations',
        'Model Name': 'Average Model',
        'A': (all_results[mid_index-1]['A'] + all_results[mid_index]['A']) / 2,
        'C': (all_results[mid_index-1]['C'] + all_results[mid_index]['C']) / 2,
        'Score': (all_results[mid_index-1]['Score'] + all_results[mid_index]['Score']) / 2,
        'Solution': {
            'lr': (all_results[mid_index-1]['Solution']['lr'] + all_results[mid_index]['Solution']['lr']) / 2,
            'model_index': 'Average'
        }
    }
    
    # Cria um dicionário para retornar os resultados do melhor, médio e pior
    summary = {
        'Best Score': {
            'Iteration': best_result['Iteration'],
            'Model Name': best_result['Model Name'],
            'A': best_result['A'],
            'C': best_result['C'],
            'Score': best_result['Score'],
            'Solution': best_result['Solution']
        },
        'Average Score': {
            'Iteration': average_result['Iteration'],
            'Model Name': average_result['Model Name'],
            'A': average_result['A'],
            'C': average_result['C'],
            'Score': average_result['Score'],
            'Solution': average_result['Solution']
        },
        'Worst Score': {
            'Iteration': worst_result['Iteration'],
            'Model Name': worst_result['Model Name'],
            'A': worst_result['A'],
            'C': worst_result['C'],
            'Score': worst_result['Score'],
            'Solution': worst_result['Solution']
        }
    }
    
    return summary

def rank_scores(oace_metrics_per_iteration):
    ranked_results = []

    # Percorre todas as iterações e extrai as informações relevantes
    for iteration, metrics_dict in oace_metrics_per_iteration.items():
        ranked_results.append({
            'Iteration': iteration,
            'Model Name': metrics_dict["model_name"],
            'A': metrics_dict["A"],
            'C': metrics_dict["C"],
            'Score': metrics_dict["Score"],
            'Solution': metrics_dict["solution"]
        })

    # Reorganiza os resultados do melhor Score para o pior
    ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)

    # Converte a lista de resultados classificados em um dicionário organizado por rank
    ranked_dict = {rank + 1: result for rank, result in enumerate(ranked_results)}
    
    return ranked_dict

def plot_convergence(oace_metrics_per_iteration):
    pass