import pickle
import os

def adjust_checkpoint(checkpoint_path="checkpoint.pkl", new_maximos_a=[0.9003294748121103, 0.8979, 0.8978999999999999], 
                     new_minimos_a=[0.03154491629236423, 0.1001, 0.1001]):
    """
    Ajusta o checkpoint.pkl removendo iterações com chaves >= 50 e atualizando maximos_a e minimos_a.
    
    Args:
        checkpoint_path (str): Caminho do arquivo checkpoint.pkl.
        new_maximos_a (list): Novos valores para maximos_a.
        new_minimos_a (list): Novos valores para minimos_a.
    
    Returns:
        dict: Checkpoint ajustado.
    """
    # Verifica se o arquivo existe
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"O arquivo {checkpoint_path} não foi encontrado.")

    # Carrega o checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Remove iterações com chaves >= 50 do metrics_per_iteration
    if 'metrics_per_iteration' in checkpoint:
        checkpoint['metrics_per_iteration'] = {
            key: value for key, value in checkpoint['metrics_per_iteration'].items()
            if key < 50
        }

    # Remove iterações com chaves >= 50 do oace_metrics_per_iteration, se existir
    if 'oace_metrics_per_iteration' in checkpoint:
        checkpoint['oace_metrics_per_iteration'] = {
            key: value for key, value in checkpoint['oace_metrics_per_iteration'].items()
            if key < 50
        }

    # Atualiza current_round e iteration para refletir as iterações restantes
    if 'metrics_per_iteration' in checkpoint and checkpoint['metrics_per_iteration']:
        max_iteration = max(checkpoint['metrics_per_iteration'].keys())
        checkpoint['current_round'] = 0  # Reinicia a rodada, assumindo que as iterações < 50 são da rodada 0
        checkpoint['iteration'] = max_iteration
    else:
        checkpoint['current_round'] = 0
        checkpoint['iteration'] = -1  # Nenhuma iteração restante, reinicia do zero

    # Atualiza maximos_a e minimos_a
    checkpoint['max_assertiveness_max'] = new_maximos_a
    checkpoint['max_assertiveness_min'] = new_minimos_a

    # Salva o checkpoint ajustado
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    return checkpoint

# Executa a função
if __name__ == "__main__":
    try:
        adjusted_checkpoint = adjust_checkpoint()
        print("Checkpoint ajustado com sucesso!")
        print(f"Novo checkpoint: {adjusted_checkpoint}")
    except Exception as e:
        print(f"Erro ao ajustar o checkpoint: {str(e)}")