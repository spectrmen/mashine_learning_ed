import numpy as np  # type: ignore
import psutil  # type: ignore


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024**3)


def linear_decay(
    init_val: float, final_val: float, cur_step: int, total_steps: int
) -> float:
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) + final_val * cur_step) / total_steps


def smoothen(values, kernel_size: int = 25):
    """
    Сглаживание 1D-последовательности средним по скользящему окну.
    Используется для красивых графиков TD-loss / reward без зашумлённости.

    Принимает list/np.ndarray/list of torch.Tensor (CPU/CUDA).
    """
    # Если внутри лежат torch-тензоры (например, grad_norm с GPU) —
    # конвертим в питоновские float, иначе np.asarray падает на CUDA.
    if len(values) > 0 and hasattr(values[0], "detach"):
        values = [float(v.detach().cpu()) for v in values]
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    k = min(kernel_size, len(values))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(values, kernel, mode="valid")


def play_and_log_episode(env, agent, gamma: float = 0.99, t_max: int = 10_000):
    """
    Прогоняет один эпизод greedy-политикой и логгирует статистики каждого шага.
    Используется для сравнения V_agent vs V_mc (Monte-Carlo state values).

    Возвращает dict со списками: states, qvalues, actions, rewards, v_agent, v_mc.
    """
    states, qvalues_list, actions, rewards = [], [], [], []
    s, _ = env.reset()
    for _ in range(t_max):
        qvalues = agent.get_qvalues(np.asarray([s]))[0]  # (n_actions,)
        a = int(np.argmax(qvalues))

        states.append(np.asarray(s))
        qvalues_list.append(qvalues)
        actions.append(a)

        s, r, terminated, truncated, _ = env.step(a)
        rewards.append(r)
        if terminated or truncated:
            break

    states = np.array(states)
    qvalues_arr = np.array(qvalues_list)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)

    # V_agent[t] = max_a Q(s_t, a)
    v_agent = qvalues_arr.max(axis=-1)

    # V_mc[t] = sum_{t'>=t} gamma^{t'-t} * r[t']  — реальный дисконтированный возврат
    v_mc = np.zeros_like(rewards)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        last = rewards[t] + gamma * last
        v_mc[t] = last

    return {
        "states": states,
        "qvalues": qvalues_arr,
        "actions": actions,
        "rewards": rewards,
        "v_agent": v_agent,
        "v_mc": v_mc,
    }
