"""
Lab 2 template
"""

from copy import deepcopy
import random
import time
import matplotlib.pyplot as plt

def read_incidence_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the incidence matrix of a given graph
    """
    with open(filename, 'r', encoding='utf-8') as file:
        file_content = file.readlines()
    file_content = file_content[1:-1]
    matrix = []
    matrix_size = 0
    for line in file_content:
        line = line.removesuffix('\n').removesuffix(';')
        start, end = line.split('->')
        matrix_size = max(matrix_size, int(start), int(end))
    matrix_size += 1
    matrix = []
    for _ in range(matrix_size):
        matrix.append([])
        for _ in range(len(file_content)):
            matrix[-1].append(0)
    for index, line in enumerate(file_content):
        line = line.removesuffix('\n').removesuffix(';')
        start, end = line.split('->')
        start = int(start)
        end = int(end)
        matrix[start][index] = -1
        matrix[end][index] = 1
    return matrix


def read_adjacency_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the adjacency matrix of a given graph
    """
    with open(filename, 'r', encoding='utf-8') as file:
        file_content = file.readlines()
    file_content = file_content[1:-1]
    matrix = []
    matrix_size = 0
    for line in file_content:
        line = line.removesuffix('\n').removesuffix(';')
        start, end = line.split('->')
        matrix_size = max(matrix_size, int(start), int(end))
    matrix_size += 1
    matrix = []
    for _ in range(matrix_size):
        matrix.append([])
        for _ in range(matrix_size):
            matrix[-1].append(0)
    for line in file_content:
        line = line.removesuffix('\n').removesuffix(';')
        start, end = line.split('->')
        start = int(start)
        end = int(end)
        matrix[start][end] = 1
    return matrix


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict[int, list[int]]: the adjacency dict of a given graph
    """
    with open(filename, 'r', encoding='utf-8') as file:
        file_content = file.readlines()
    file_content = file_content[1:-1]
    adjacency_dict = {}
    for line in file_content:
        line = line.removesuffix('\n').removesuffix(';')
        start, end = line.split('->')
        start = int(start)
        end = int(end)
        adjacency_dict.setdefault(start, []).append(end)
    adjacency_dict = dict(sorted(adjacency_dict.items(), key=lambda x: x[0]))
    return adjacency_dict


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    graph = dict(sorted(graph.items(), key=lambda x: x[0]))
    visited = [start]
    not_visited = set(graph.keys())
    not_visited.discard(start)
    stack = [start]
    current_location = start
    possible_ways = graph.copy()
    while not_visited:
        if current_location not in stack:
            stack.append(current_location)
        for key in possible_ways.keys():
            for elem in visited:
                if elem in possible_ways[key]:
                    possible_ways[key].remove(elem)
        while not possible_ways[current_location]:
            stack.pop()
            current_location = stack[-1]
        if not_visited:
            next_move = min(possible_ways[current_location])
            not_visited.discard(next_move)
            visited.append(next_move)
        current_location = next_move
    return visited


def iterative_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    dict_graph = {}
    for index_i, line in enumerate(graph):
        for index_j, elem in enumerate(line):
            dict_graph.setdefault(index_i, [])
            if elem:
                dict_graph.setdefault(index_i, []).append(index_j)
    dict_graph = dict(sorted(dict_graph.items(), key=lambda x: x[0]))
    visited = [start]
    not_visited = set(dict_graph.keys())
    not_visited.discard(start)
    stack = [start]
    current_location = start
    possible_ways = dict_graph.copy()
    while not_visited:
        if current_location not in stack:
            stack.append(current_location)
        for key in possible_ways.keys():
            for elem in visited:
                if elem in possible_ways[key]:
                    possible_ways[key].remove(elem)
        while not possible_ways[current_location]:
            stack.pop()
            current_location = stack[-1]
        if not_visited:
            next_move = min(possible_ways[current_location])
            not_visited.discard(next_move)
            visited.append(next_move)
        current_location = next_move
    return visited


def recursive_adjacency_dict_dfs(graph: dict[int,list[int]], start: int, visited=None) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    if visited is None:
        visited = list()
    visited.append(start)
    for neighbor in sorted(graph[start], key=lambda x: x):
        if neighbor not in visited:
            recursive_adjacency_dict_dfs(graph, neighbor, visited)
    return visited


def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int, visited=None) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    dict_graph = {}
    for index_i, line in enumerate(graph):
        for index_j, elem in enumerate(line):
            dict_graph.setdefault(index_i, [])
            if elem:
                dict_graph.setdefault(index_i, []).append(index_j)
    return recursive_adjacency_dict_dfs(dict_graph, start, visited)


def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    graph = dict(sorted(graph.items(), key=lambda x: x[0]))
    visited = [start]
    next_layer = []
    current_layer = [start]
    while current_layer:
        for peak in current_layer:
            for val in graph[peak]:
                if val not in visited:
                    next_layer.append(val)
                    visited.append(val)
        current_layer = next_layer[::]
        next_layer.clear()
    return visited


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    dict_graph = {}
    for index_i, line in enumerate(graph):
        for index_j, elem in enumerate(line):
            dict_graph.setdefault(index_i, [])
            if elem:
                dict_graph.setdefault(index_i, []).append(index_j)
    graph = dict_graph
    graph = dict(sorted(graph.items(), key=lambda x: x[0]))
    visited = [start]
    next_layer = []
    current_layer = [start]
    while current_layer:
        for peak in current_layer:
            for val in graph[peak]:
                if val not in visited:
                    next_layer.append(val)
                    visited.append(val)
        current_layer = next_layer[::]
        next_layer.clear()
    return visited


def bfs(graph: dict[int, list[int]], start: int) -> dict:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns dict: key: distance drom start point to value(points)
    """
    graph = dict(sorted(graph.items(), key=lambda x: x[0]))
    visited = [start]
    vist = [[start]]
    next_layer = []
    current_layer = [start]
    while current_layer:
        vist.append([])
        for peak in current_layer:
            for val in graph[peak]:
                if val not in visited:
                    next_layer.append(val)
                    visited.append(val)
                    vist[-1].append(val)
        current_layer = next_layer[::]
        next_layer.clear()
    if not vist[-1]:
        vist.pop()
    visit = {}
    for index, elem in enumerate(vist):
        visit.setdefault(index, elem)
    return visit

def adjacency_matrix_radius(graph: list[list[int]]) -> int:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    1
    """
    dict_graph = {}
    for index_i, line in enumerate(graph):
        for index_j, elem in enumerate(line):
            if elem:
                dict_graph.setdefault(index_i, []).append(index_j)
    radius = 0
    start = True
    for peak in dict_graph:
        ecc = max(bfs(dict_graph, peak).keys())
        if start:
            start = False
            radius = ecc
        radius = min(radius, ecc)
    return radius


def adjacency_dict_radius(graph: dict[int, list[int]]) -> int:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1]})
    1
    """
    radius = 0
    start = True
    for peak in graph.keys():
        ecc = max(bfs(graph, peak).keys())
        if start:
            start = False
            radius = ecc
        radius = min(radius, ecc)
    return radius


def find_cycles(graph: dict[int, list[int]]) -> list[list[int]]:
    """
    Знаходить усі прості орієнтовані цикли в графі.
    :param graph: список суміжності орієнтованого графа
    :return: список циклів; кожен цикл — це список вершин,
    де перша вершина повторюється в кінці
    >>> find_cycles({0: [1], 1: [0]})
    [[0, 1, 0]]
    """
    cycles = []
    seen_cycles = set()
    vertices = sorted(graph.keys())
    def normalize_cycle(cycle: list[int]) -> tuple[int, ...]:
        """
        Перебирає цикл так шоб найменша вершина була першою
        для усунення дублікатів
        """
        core = cycle[:-1]
        min_v = min(core)
        while core[0] != min_v:
            core.insert(0, core[-1])
            core.pop()
        return tuple(core)
    def dfs(current: int, start: int, path: list[int], on_path: set[int]) -> None:
        """
        dfs який будує шляхи і цикли що повертаються у start.
        """
        for neighbor in graph.get(current, []):
            if neighbor == start:
                cycle = path + [start]
                key = normalize_cycle(cycle)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(deepcopy(cycle))
            elif neighbor > start and neighbor not in on_path:
                on_path.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, start, path, on_path)
                path.pop()
                on_path.remove(neighbor)
    for start in vertices:
        dfs(start, start, [start], {start})
    return cycles

def generate_random_digraph(n: int, p: float, path: str) -> dict[int, list[int]]:
    """
    Генерує орієнтований граф з n вершин.
    Для кожної пари додаємо ребро з імовірністю p.
    """
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < p:
                graph[i].append(j)
    with open(path, 'w', encoding='utf-8') as file:
        file.write('digraph sample_input {\n')
        for key, value_lis in graph.items():
            for value in value_lis:
                file.write(f'\t{key} -> {value};\n')
        file.write('}')
    return graph

def compare_functions(size: list, output_file='test_lab.dot', input_file='input.dot'):
    """
    Порівнює швидкість дії функцій на різному об'ємі даних.
    """
    read_incidence_matrix_times = []
    read_adjacency_matrix_times = []
    read_adjacency_dict_times = []

    iterative_adjacency_dict_dfs_times = []
    iterative_adjacency_matrix_dfs_times = []
    recursive_adjacency_dict_dfs_times = []
    recursive_adjacency_matrix_dfs_times = []

    iterative_adjacency_dict_bfs_times = []
    iterative_adjacency_matrix_bfs_times = []

    adjacency_matrix_radius_times = []
    adjacency_dict_radius_times = []

    find_cycles_times = []
    for n in size:
        graph = generate_random_digraph(n, 0.57, output_file)
        #порівняння зчитання
        start = time.perf_counter()
        read_incidence_matrix(input_file)
        end = time.perf_counter()
        read_incidence_matrix_times.append(end - start)

        start = time.perf_counter()
        adjency_graph = read_adjacency_matrix(input_file)
        end = time.perf_counter()
        read_adjacency_matrix_times.append(end - start)

        start = time.perf_counter()
        read_adjacency_dict(input_file)
        end = time.perf_counter()
        read_adjacency_dict_times.append(end - start)

        #порівняння dfs
        start = time.perf_counter()
        iterative_adjacency_dict_dfs(graph, 0)
        end = time.perf_counter()
        iterative_adjacency_dict_dfs_times.append(end - start)

        start = time.perf_counter()
        iterative_adjacency_matrix_dfs(adjency_graph, 0)
        end = time.perf_counter()
        iterative_adjacency_matrix_dfs_times.append(end - start)

        start = time.perf_counter()
        recursive_adjacency_dict_dfs(graph, 0)
        end = time.perf_counter()
        recursive_adjacency_dict_dfs_times.append(end - start)

        start = time.perf_counter()
        recursive_adjacency_matrix_dfs(adjency_graph, 0)
        end = time.perf_counter()
        recursive_adjacency_matrix_dfs_times.append(end - start)

        #порівняння bfs
        start = time.perf_counter()
        iterative_adjacency_dict_bfs(graph, 0)
        end = time.perf_counter()
        iterative_adjacency_dict_bfs_times.append(end - start)

        start = time.perf_counter()
        iterative_adjacency_matrix_bfs(adjency_graph, 0)
        end = time.perf_counter()
        iterative_adjacency_matrix_bfs_times.append(end - start)

        #порівняння знаходження радіусу
        start = time.perf_counter()
        adjacency_matrix_radius(adjency_graph)
        end = time.perf_counter()
        adjacency_matrix_radius_times.append(end - start)

        start = time.perf_counter()
        adjacency_dict_radius(graph)
        end = time.perf_counter()
        adjacency_dict_radius_times.append(end - start)

        #знаходження усіх циклів
        start = time.perf_counter()
        find_cycles(graph)
        end = time.perf_counter()
        find_cycles_times.append(end - start)
    plt.plot(size, read_incidence_matrix_times, label='Читання \
матриці і запис у матрицю інцидентності')
    plt.plot(size, read_adjacency_matrix_times, label='Читання \
матриці і запис у матрицю суміжності')
    plt.plot(size, read_adjacency_dict_times, label='Читання матриці і запис у словник')

    plt.xlabel('Розмір вхідних даних (n)')
    plt.ylabel('Час роботи (секунди)')
    plt.title('Порівняння часу роботи алгоритмів по зчитуванню файлів')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(size, iterative_adjacency_dict_dfs_times, label='\
Знаходження dfs з словника ітеративно')
    plt.plot(size, iterative_adjacency_matrix_dfs_times, label='\
Знаходження dfs з матриці ітеративно')
    plt.plot(size, recursive_adjacency_dict_dfs_times, label='\
Знаходження dfs з словника рекурсивно')
    plt.plot(size, recursive_adjacency_matrix_dfs_times, label='\
Знаходження dfs з матриці рекурсивно')

    plt.xlabel('Розмір вхідних даних (n)')
    plt.ylabel('Час роботи (секунди)')
    plt.title('Порівняння часу роботи алгоритмів по знаходженню dfs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(size, iterative_adjacency_dict_bfs_times, label='\
Знаходження bfs з словника ітеративно')
    plt.plot(size, iterative_adjacency_matrix_bfs_times, label='\
Знаходження bfs з матриці ітеративно')

    plt.xlabel('Розмір вхідних даних (n)')
    plt.ylabel('Час роботи (секунди)')
    plt.title('Порівняння часу роботи алгоритмів по знаходженню bfs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(size, adjacency_matrix_radius_times, label='Знаходження радіусу з матриці')
    plt.plot(size, adjacency_dict_radius_times, label='Знаходження радіусу з словника')

    plt.xlabel('Розмір вхідних даних (n)')
    plt.ylabel('Час роботи (секунди)')
    plt.title('Порівняння часу роботи алгоритмів по знаходженню bfs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(size, find_cycles_times, label='Знаходження усіх циклів у матриці')

    plt.xlabel('Розмір вхідних даних (n)')
    plt.ylabel('Час роботи (секунди)')
    plt.title('Порівняння часу роботи алгоритмів по знаходженню всіх циклів')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Функція запускає весь код.
    """
    compare_list = [x for x in range(10, 101, 10)]
    compare_functions(compare_list)

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
    main()
