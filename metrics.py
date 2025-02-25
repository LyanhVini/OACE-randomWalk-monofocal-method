"""
Implementação das métricas e funções para cálculo do método OACE
"""
import numpy as np
from models import *
import numpy as np
from pyDecision.algorithm import ahp_method

metrics_a = ["precision", "accuracy", "recall"]
metrics_c = ["mtp", "tpi", "ms"]
epsilon = 1e-5 # Parâmetro da padronização para evitar divisão por zero

##################
### AHP Method ###
##################

weight_derivation = 'geometric' # 'mean'; 'geometric' or 'max_eigen'

dataset = np.array([# Dataset for assertiveness metrics (P -> A -> R)
  #P      A      R
  [1  ,   5,     7   ],   #P
  [1/5,   1,     3   ],   #A
  [1/7,   1/3,   1   ],   #R
])
dataset = np.array([ # Dataset for cost metrics (MTP -> TPI -> MS)
  #MTP    TPI    MS
  [  1,     5,   7   ],   #MTP
  [1/5,     1,   3   ],   #TPI
  [1/7,   1/3,   1   ],   #MS
])
# Call AHP Function
weights, rc = ahp_method(dataset, wd = weight_derivation)
w1, w2, w3 = weights[0], weights[1], weights[2]

wa = [w1, w2, w3] # Precision, Acuraccy, Recall
wc = [w1, w2, w3] # MTP, TPI, MS

### OACE Method ###

def get_max_min_metrics(metrics_dict):
    """Calcula os máximos e mínimos das métricas de custo resultante do aquecimento dos modelos"""
    metricas_custo = ['mtp', 'tpi', 'ms']
    
    max_custo = {metrica: float('-inf') for metrica in metricas_custo}
    min_custo = {metrica: float('inf') for metrica in metricas_custo}

    for model_name, metrics in metrics_dict.items():
        for metrica in metricas_custo:
            valor = metrics[metrica]
            if valor > max_custo[metrica]:
                max_custo[metrica] = valor
            if valor < min_custo[metrica]:
                min_custo[metrica] = valor

    # Convertendo os dicionários de máximos e mínimos em listas simples
    max_custo_list = [max_custo[metrica] for metrica in metricas_custo]
    min_custo_list = [min_custo[metrica] for metrica in metricas_custo]

    return max_custo_list, min_custo_list

def calculo_maximum_minimum(metrics_list, metricas_a):
    """Calcula os máximos e mínimos das métricas de assertividade obtida a cada iteração"""
    valores_a = {metrica: [] for metrica in metricas_a}
    
    # Iterar sobre as métricas recebidas em cada iteração e armazenar valores nas listas
    for metrics in metrics_list:
        for metrica in metricas_a:
            valores_a[metrica].append(float(metrics['assertividade'][metrica]))
    
    # Usar numpy para calcular os valores máximos e mínimos
    max_metrics_a = [float(np.max(valores_a[metrica])) for metrica in metricas_a]
    min_metrics_a = [float(np.min(valores_a[metrica])) for metrica in metricas_a]
    
    return max_metrics_a, min_metrics_a

def N(value, max_value, min_value):
    """Função de normalização min-max, com clamp para garantir valores entre 0 e 1 e logs para valores fora do intervalo."""
    if max_value is None or min_value is None or max_value == min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, normalized))  # Clampa entre 0 e 1

def N_cost(value, max_value, min_value):
    if max_value is None or min_value is None or max_value == min_value:
        return 0.0
    normalized = (max_value - value) / (max_value - min_value)
    return max(0.0, min(1.0, normalized))  

def A(metrics, wa, metricas_a, maximos_a, minimos_a):
    """Calcula a assertividade normalizada."""
    a_i = [metrics['assertividade'][metrica] for metrica in metricas_a]
    normalized_values = [N(a, maximo, minimo) for a, maximo, minimo in zip(a_i, maximos_a, minimos_a)]
    return sum([n * w for n, w in zip(normalized_values, wa)])

def C(metrics, wc, metricas_c, maximos_c, minimos_c):
    """Calcula o custo normalizado."""
    c_i = [metrics['custo'][metrica] for metrica in metricas_c]
    normalized_values = [N_cost(c, maximo, minimo) for c, maximo, minimo in zip(c_i, maximos_c, minimos_c)]
    return sum([n * w for n, w in zip(normalized_values, wc)])

def F_score(lambda_, assertiveness, cost, max_score, min_score):
    """Calcula o score do método e normaliza."""
    score = lambda_ * assertiveness + (1 - lambda_) * cost
    return N(score, max_score, min_score)

def calculo_maximos_minimos(accumulated_metrics, metricas, tipo):
    """Calcula os máximos e mínimos para normalização das métricas."""
    if not accumulated_metrics:
        return [0.0] * len(metricas), [0.0] * len(metricas)  # Valores padrão para evitar erros iniciais

    valores = {metrica: [] for metrica in metricas}
    for metrics in accumulated_metrics:
        for metrica in metricas:
            valores[metrica].append(float(metrics[tipo][metrica]))

    maximos = [np.max(valores[metrica]) for metrica in metricas]
    minimos = [np.min(valores[metrica]) for metrica in metricas]
    return maximos, minimos

def calcular_metricas_oace(iteration_metrics, accumulated_metrics, lambda_, wa, wc, metricas_a, metricas_c, maximos_a=None, minimos_a=None, maximos_c=None, minimos_c=None):
    print("\n🔎 INICIANDO A AVALIAÇÃO DO OACE PARA ITERAÇÃO ATUAL")
    print("iteration_metrics: ", iteration_metrics)
    print("accumulated_metrics: ", accumulated_metrics)

    if not iteration_metrics:
        return None
    if maximos_a is None or minimos_a is None:
        maximos_a, minimos_a = calculo_maximos_minimos(accumulated_metrics, metricas_a, "assertividade")
        
    print("maximos_a: ", maximos_a)
    print("minimos_a: ", minimos_a)
    print("maximos_c (fixos do aquecimento): ", maximos_c)
    print("minimos_c (fixos do aquecimento): ", minimos_c)

    max_score = lambda_ * 1.0 + (1 - lambda_) * 1.0  # Máximo teórico
    min_score = lambda_ * 0.0 + (1 - lambda_) * 0.0  # Mínimo teórico

    assertividade = A(iteration_metrics, wa, metricas_a, maximos_a, minimos_a)
    print("🔹 A (Assertividade Normalizada): ", assertividade)
    custo = C(iteration_metrics, wc, metricas_c, maximos_c, minimos_c)
    print("🔹 C (Custo Normalizado): ", custo)
    score = F_score(lambda_, assertividade, custo, max_score, min_score)
    print("🔹 S (Score OACE): ", score)

    return {
        "model_name": iteration_metrics["model_name"],
        "A": assertividade,
        "C": custo,
        "Score": score,
        "solution": iteration_metrics["solution"]
    }