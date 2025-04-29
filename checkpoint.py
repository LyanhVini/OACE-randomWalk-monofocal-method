
from hyperparameter_optimization import *
checkpoint_path = 'checkpoint.pkl'

checkpoint = load_checkpoint(checkpoint_path)
if checkpoint:
    current_round = checkpoint['current_round']
    iteration = checkpoint['iteration']
    best_model_overall = checkpoint['best_model_overall']
    best_score_overall = checkpoint['best_score_overall']
    best_solution_overall = checkpoint['best_solution_overall']
    metrics_per_iteration = checkpoint['metrics_per_iteration']  
    oace_metrics_per_iteration = checkpoint['oace_metrics_per_iteration']  
    maximos_a = checkpoint['max_assertiveness_max']  
    minimos_a = checkpoint['max_assertiveness_min']  
    
metricas_a = ["precision", "accuracy", "recall"]
print(current_round)
print(iteration)
print(metrics_per_iteration)
print(oace_metrics_per_iteration)
print(maximos_a)
print(minimos_a)

