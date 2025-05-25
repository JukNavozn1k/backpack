import pytest
import pulp
from backpack import backpack
import random

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

def brute_force_knapsack(weights, values, capacity):
    """
    Тривиальный перебор всех подмножеств для n <= 15.
    Возвращает (max_value, selected_indices).
    """
    n = len(weights)
    best_value = 0
    best_set = []
    for mask in range(1 << n):
        total_w = 0
        total_v = 0
        subset = []
        for i in range(n):
            if mask & (1 << i):
                total_w += weights[i]
                total_v += values[i]
        if total_w <= capacity and total_v > best_value:
            best_value = total_v
            # сохранить в порядке возрастания индексов
            best_set = [i for i in range(n) if mask & (1 << i)]
    return best_value, best_set

@pytest.mark.parametrize("weights, values, capacity", [
    # нулёвая вместимость — всегда 0, [],
    ([1,2,3], [10,20,30], 0),
    # предметы нулевого веса — можно взять все
    ([0,0,1], [5,6,7], 1),
    # предметы нулевой ценности — бесполезны
    ([1,2,3], [0,0,0], 5),
    # одинаковые веса, разные ценности
    ([2,2,2,2], [1,5,3,4], 4),
    # одинаковые ценности, разные веса
    ([1,2,3,4], [10,10,10,10], 5),
])
def test_special_edge_cases(weights, values, capacity):
    max_bp, sel_bp, history = backpack(weights, values, capacity)
    max_bf, sel_bf = brute_force_knapsack(weights, values, capacity)

    # верное значение
    assert max_bp == max_bf, f"Expected max {max_bf}, got {max_bp}"
    # корректная выборка
    assert set(sel_bp) == set(sel_bf), f"Expected items {sel_bf}, got {sel_bp}"
    # суммарный вес <= capacity
    total_w = sum(weights[i] for i in sel_bp)
    assert total_w <= capacity, f"Total weight {total_w} exceeds capacity {capacity}"
    # история нужной длины
    assert len(history) == len(weights) + 1

def test_history_shapes_and_initial_state():
    weights = [3,1]
    values = [4,2]
    capacity = 5
    _, _, history = backpack(weights, values, capacity)

    # Каждая таблица — размер (n+1)×(capacity+1)
    for dp in history:
        assert len(dp) == len(weights) + 1
        for row in dp:
            assert len(row) == capacity + 1

    # Начальное состояние — все нули
    assert all(v == 0 for row in history[0] for v in row)

def test_random_small_instances():
    random.seed(42)
    for _ in range(10):
        n = random.randint(1, 6)
        weights = [random.randint(0, 5) for _ in range(n)]
        values  = [random.randint(0, 10) for _ in range(n)]
        capacity = random.randint(0, 15)

        max_bp, sel_bp, _ = backpack(weights, values, capacity)
        max_bf, sel_bf = brute_force_knapsack(weights, values, capacity)

        assert max_bp == max_bf, (
            f"Random test failed max: weights={weights}, values={values}, "
            f"cap={capacity}, got {max_bp}, expected {max_bf}"
        )
        assert set(sel_bp) == set(sel_bf), (
            f"Random test failed sel: weights={weights}, values={values}, "
            f"cap={capacity}, got {sel_bp}, expected {sel_bf}"
        )


@pytest.mark.parametrize("weights, values, capacity, expected_exception", [
    # несовпадение длины списков
    ([1, 2], [3], 5, ValueError),
    # отрицательная вместимость
    ([1, 2, 3], [10, 20, 30], -1, ValueError),
    # отрицательный вес
    ([-1, 2], [10, 20], 5, ValueError),
    # отрицательная ценность
    ([1, 2], [10, -5], 5, ValueError),
    # нецелочисленные значения
    ([1.5, 2], [10, 20], 5, TypeError),
    ([1, 2], [10, '20'], 5, TypeError),
    ('not a list', [10, 20], 5, TypeError),
    ([1, 2], [10, 20], 'five', TypeError),
])
def test_backpack_validation_errors(weights, values, capacity, expected_exception):
    with pytest.raises(expected_exception):
        backpack(weights, values, capacity)