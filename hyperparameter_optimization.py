"""
Implementação do Random Walk para otimização dos hiperparâmetros.
- Random Walk deve caminhar aleatoriamente por lr e na seleção do modelo a ser treinado a cada iteração
- A cada iteração deve ser armazenado as informações do modelo, hiperparametrização, métricas, score \sphi, score de assertividade e custo.
- As listas oace_metrics_per_iteraction e metrics_per_iteraction para indexar os dicionários por iteração e não por arquitetura
- Ao final, deve ser retornado os 5 melhores treinamentos durante todas a iterações
"""
import pickle
import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from metrics import *
from torchvision.models.inception import InceptionOutputs
import torch.nn.functional as F
from datasets import *
from models import *
from sklearn.metrics import accuracy_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_solution(models):
    """ Gera uma solução inicial para o Random Walk, em termos de lr e do modelo a ser usado"""
    lr = random.uniform(1e-4, 1e-2)
    model_index = random.randint(0, len(models) - 1)
    print("model: ", model_index)
    return [lr, model_index]

def random_walk_step(solution, models, step_size=0.1):
    """Executa um passo do Random Walk para explorar novas soluções"""
    new_solution = solution.copy()
    new_solution[0] = min(max(new_solution[0] + random.uniform(-step_size * new_solution[0], step_size * new_solution[0]), 1e-4), 1e-2) 
    new_solution[1] = random.randint(0, len(models) - 1) 
    return new_solution

def train_models(model, trainLoader, validLoader, criterion, optimizer, epochs=30, early_stopping_rounds=5):
    """Treina o modelo e monitora a perda de validação para aplicar early stopping."""
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)

            # Redimensionamento dinâmico se necessário
            if isinstance(model, models.Inception3):
                data = F.interpolate(data, size=(299, 299), mode='bilinear', align_corners=False)

            optimizer.zero_grad()
            output = model(data)

            # Verifica se a saída é do tipo InceptionOutputs
            if isinstance(output, InceptionOutputs):
                output = output.logits  # Usa apenas a saída principal

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        with torch.no_grad():
            for data, target in validLoader:
                data, target = data.to(device), target.to(device)

                 # Redimensionamento dinâmico se necessário
                if isinstance(model, models.Inception3):
                    data = F.interpolate(data, size=(299, 299), mode='bilinear', align_corners=False)

                output = model(data)
                # Verifica se a saída é do tipo InceptionOutputs
                if isinstance(output, InceptionOutputs):
                    output = output.logits  # Usa apenas a saída principal
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss/len(trainLoader.dataset)
        valid_loss = valid_loss/len(validLoader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f"Early stopping after {epoch} epochs due to no improvement.")
            break

    return model

def evaluate_solution(model, trainLoader, testLoader, validLoader, criterion, optimizer, dataset_name):
    """
    - Avalia uma solução de modelo treinado medindo sua precisão, acurácia, recall, tempo de inferência, tamanho e número de parâmetros.
    - Esta função treina o modelo, realiza inferências no conjunto de testes e calcula várias métricas de desempenho, incluindo assertividade e custo computacional.
    """
    model = train_models(model, trainLoader, validLoader, criterion, optimizer)
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for inputs, target in testLoader:
            inputs, target = inputs.to(device), target.to(device)

            # Redimensionamento dinâmico se necessário
            if isinstance(model, models.Inception3):
                inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)

            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            inference_times.append(inference_time)

    # Defina o valor de 'average' com base no tipo de dataset
    if dataset_name == "Chest X-Ray":
        average_type = 'binary'
    else:
        average_type = 'macro'

    precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)# No dataset char x-ray average binary deve ser utilizada, para o restanto, micro ou average

    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters()) 
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

    return float(precision), accuracy, float(recall), avg_inference_time, model_size, num_params

def optimize_hyperparameters(models_list, trainloader, testloader, validLoader, classes, lbd, wa, wc, dataset_name, checkpoint_path, num_rounds=3, iterations_per_round=10, save_checkpoint_every=5):
    
    initial_seeds = [100, 600, 1100]  # Seeds fixas, pode ser [50, 150, 200], [100, 600, 1100] ou [1000, 2500, 4000]

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

        print(f"🔄 Continuação do treinamento a partir da Rodada {current_round + 1}, Iteração {iteration + 1}...")
    else:
        current_round = 0
        iteration = 0  
        best_model_overall = None
        best_score_overall = float('-inf')
        best_solution_overall = None
        metrics_per_iteration = {}  # Dicionário para todas as iterações
        oace_metrics_per_iteration = {}  # Dicionário para todas as métricas OACE
        maximos_a, minimos_a = None, None  # Inicia como None para recalcular ou carregar do checkpoint

    with open('warm_up_metrics.pkl', 'rb') as f:
        warm_up_metrics = pickle.load(f)

    maximos_c, minimos_c = get_max_min_metrics(warm_up_metrics)

    accumulated_metrics = []  # Para calcular máximos/mínimos de assertividade
    total_iterations = num_rounds * iterations_per_round  # Total de iterações globais

    for round_idx in range(current_round, num_rounds):
        print(f"\n🔄 Iniciando Rodada {round_idx + 1}/{num_rounds} 🔄")
        base_seed = initial_seeds[round_idx] 
        best_model = None
        best_score = float('-inf')
        best_solution = None if not checkpoint or round_idx > current_round else best_solution_overall

        # Reinicia a solução se for um novo round após o checkpoint
        if round_idx > current_round or not checkpoint:
            solution = generate_solution(models_list)
        else:
            solution = best_solution_overall

        # Itera sobre as iterações desta rodada, incrementando a seed
        for local_iteration in range(iterations_per_round):
            global_iteration = round_idx * iterations_per_round + local_iteration  # Iteração global
            if global_iteration <= iteration and checkpoint:  # Pula iterações já processadas
                continue

            current_seed = base_seed + local_iteration  # Incrementa a seed dentro do round
            random.seed(current_seed)

            print(f"▶️ Iteration {global_iteration + 1}/{total_iterations} (Round {round_idx + 1}): model {solution[1]} e lr {solution[0]}")
            print(f"🔀 Random Seed Atual: {current_seed}")

            model_name, Model = models_list[solution[1]]
            model = Model(num_classes=len(classes)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=solution[0])
            criterion = nn.CrossEntropyLoss()

            precision, accuracy, recall, avg_inference_time, model_size, num_params = evaluate_solution(
                model, trainloader, testloader, validLoader, criterion, optimizer, dataset_name)

            current_metrics = {
                "model_name": model_name,
                "assertividade": {"precision": precision, "accuracy": accuracy, "recall": recall},
                "custo": {"mtp": num_params, "tpi": avg_inference_time, "ms": model_size},
                "solution": {"lr": solution[0], "model_index": solution[1]}
            }

            accumulated_metrics.append(current_metrics)
            metrics_per_iteration[global_iteration] = current_metrics

            # Calcular OACE usando os custos fixos do aquecimento e os máximos/mínimos de assertividade carregados ou atualizados
            oace_metrics = calcular_metricas_oace(
                current_metrics,
                accumulated_metrics,
                lbd, wa, wc,
                ["precision", "accuracy", "recall"],
                ["mtp", "tpi", "ms"],
                maximos_a, minimos_a,
                maximos_c, minimos_c
            )
            oace_metrics_per_iteration[global_iteration] = oace_metrics

            print("Resultado do OACE por Iteração: ", oace_metrics)
            score = oace_metrics["Score"]

            # Atualizar o melhor modelo globalmente (sobre todas as iterações)
            if score > best_score_overall:
                best_score_overall = score
                best_model_overall = model_name
                best_solution_overall = solution.copy()

            solution = random_walk_step(solution, models_list)

            # Salvar checkpoint a cada 'save_checkpoint_every' iterações globais
            if (global_iteration + 1) % save_checkpoint_every == 0 or global_iteration == total_iterations - 1:
                save_checkpoint(
                    round_idx, global_iteration, best_model_overall, best_score_overall, best_solution_overall,
                    metrics_per_iteration, oace_metrics_per_iteration, maximos_a, minimos_a, checkpoint_path
                )

        # Atualizar máximos e mínimos de assertividade ao final de cada round
        current_max_a, current_min_a = calculo_maximos_minimos(accumulated_metrics, ["precision", "accuracy", "recall"], "assertividade")
        if maximos_a is None or minimos_a is None:
            maximos_a, minimos_a = current_max_a, current_min_a
        else:
            for i in range(len(current_max_a)):
                maximos_a[i] = max(maximos_a[i], current_max_a[i])
                minimos_a[i] = min(minimos_a[i], current_min_a[i])

    return best_model_overall, best_score_overall, best_solution_overall, metrics_per_iteration, oace_metrics_per_iteration

def warm_calculate_metrics(model, dataloader, device):
    """Calcula as métricas de custo no aquecimento para um modelo dado"""
    model.eval()
    total_inference_time = 0.0

    inference_times = []

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            _, preds = torch.max(outputs, 1)
            inference_times.append(inference_time)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    num_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024**2)  # Size in MB

    return num_params, avg_inference_time, model_size

def warm_up_models(models, dataloader, device):
    metrics = {}
    for model_name, get_model_func in models:
        model = get_model_func().to(device)
        num_params, avg_inference_time, model_size = warm_calculate_metrics(model, dataloader, device)
        metrics[model_name] = {
            'mtp': num_params,
            'tpi': avg_inference_time,
            'ms': model_size
        }
    return metrics

def save_checkpoint(current_round, iteration, best_model_overall, best_score_overall, best_solution_overall,
                    metrics_per_iteration, oace_metrics_per_iteration, max_assertiveness_max, max_assertiveness_min, checkpoint_path):
    """Salva o estado atual do treinamento para continuar depois."""
    checkpoint_data = {
        'current_round': current_round,
        'iteration': iteration,
        'best_model_overall': best_model_overall,
        'best_score_overall': best_score_overall,
        'best_solution_overall': best_solution_overall,
        'metrics_per_iteration': metrics_per_iteration,  # Armazena todas as iterações
        'oace_metrics_per_iteration': oace_metrics_per_iteration,  # Armazena todas as métricas OACE
        'max_assertiveness_max': max_assertiveness_max,
        'max_assertiveness_min': max_assertiveness_min
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"✅ Checkpoint salvo na Rodada {current_round + 1}, Iteração {iteration + 1}.")

def load_checkpoint(checkpoint_path):
    """Carrega o checkpoint, se existir."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    else:
        return None