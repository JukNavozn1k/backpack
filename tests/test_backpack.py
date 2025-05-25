import pytest
import pulp
from backpack import backpack

def solve_with_pulp(weights, values, capacity):
    """
    Возвращает (max_value, selected_indices) через целочисленное решение PuLP.
    Обрабатывает случай пустого списка.
    """
    n = len(weights)
    if n == 0:
        return 0, []

    # Определяем модель
    model = pulp.LpProblem("Knapsack", pulp.LpMaximize)
    # Переменные x0…x{n-1} ∈ {0,1}
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    # Целевая функция
    model += pulp.lpSum(values[i] * x[i] for i in range(n))
    # Ограничение по весу
    model += pulp.lpSum(weights[i] * x[i] for i in range(n)) <= capacity
    # Решаем
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    # Собираем результат
    selected = [i for i in range(n) if pulp.value(x[i]) == 1]
    max_value = pulp.value(model.objective) or 0
    return int(max_value), selected

# Параметры для сравнения с PuLP
@pytest.mark.parametrize("weights, values, capacity", [
    ([], [], 10),                            # пустой рюкзак
    ([5], [10], 5),                          # один предмет помещается
    ([6], [10], 5),                          # один предмет не помещается
    ([2,3,4,5], [3,4,5,6], 5),               # несколько предметов
    ([1,4,3,1], [1500,3000,2000,200], 4),    # более общий случай
    ([2,2,2], [1,2,3], 6),                   # граничный случай: ровно впритык
])
def test_backpack_against_pulp(weights, values, capacity):
    max_bp, sel_bp, _ = backpack(weights, values, capacity)
    max_pulp, sel_pulp = solve_with_pulp(weights, values, capacity)

    assert max_bp == max_pulp, (
        f"Max value mismatch: backpack={max_bp}, pulp={max_pulp} "
        f"for weights={weights}, values={values}, capacity={capacity}"
    )
    # сравниваем множества индексов (порядок неважен)
    assert set(sel_bp) == set(sel_pulp), (
        f"Selected items mismatch: backpack={sel_bp}, pulp={sel_pulp} "
        f"for weights={weights}, values={values}, capacity={capacity}"
    )

# Параметры для проверки истории
@pytest.mark.parametrize("weights, values, capacity", [
    ([1,2,3], [1,2,3], 4),
    ([2,3,5], [3,4,6], 7),
    ([1,1,1,1], [1,1,1,1], 2),
])
def test_history_non_decreasing(weights, values, capacity):
    _, _, history = backpack(weights, values, capacity)
    # Для каждого шага i проверяем, что таблица не уменьшилась
    for i in range(1, len(history)):
        prev_dp = history[i-1]
        cur_dp  = history[i]
        for row_prev, row_cur in zip(prev_dp, cur_dp):
            for prev_val, cur_val in zip(row_prev, row_cur):
                assert cur_val >= prev_val, (
                    "DP values must be non-decreasing when adding items"
                )
