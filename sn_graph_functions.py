import pandas as pd
import networkx as nx
import numpy as np
from operator import itemgetter
import random


def create_initial_graph(path):
    data = pd.read_csv(path, sep=',', header=None, names=['id1', 'id2', 'time', 'intensity'])

    # Создадим граф
    initial_graph = nx.DiGraph()

    # Сформируем список вершин
    id1_list = list(data['id1'])
    id2_list = list(data['id2'])
    ids = list(set(id1_list + id2_list))
    initial_graph.add_nodes_from(ids)

    # Сформируем список ребер
    edges = list(zip(id1_list, id2_list))
    initial_graph.add_edges_from(edges)

    attrs = {}
    time_list = data['time']
    intensity_list = data['intensity']
    edges_attributes = list(zip(id1_list, id2_list, time_list, intensity_list))
    for pair in edges_attributes:
        attrs[(pair[0], pair[1])] = {'time': pair[2], 'intensity': pair[3]}
    nx.set_edge_attributes(initial_graph, attrs)

    return initial_graph


def edge_is_mutual(a, b, g):
    if g.has_edge(b, a):
        return 1
    else:
        return 0


def friends_of_friends(u, g):
    # ищем друзей друзей (из результата убираем непосредственно друзей указанного пользователя)
    # (для направленного графа neighbors() ищет соседей по исходящим ребрам)
    result = set()
    u_neighbors = set(g.neighbors(u))
    for node in u_neighbors:
        result = result.union(set(g.neighbors(node)))
    result = result.difference(u_neighbors)
    result.discard(u)
    return result


def common_friends(u, v, g):
    # ищем общих друзей для двух пользователей
    # (исключая самих этих пользователей, если есть связь от u к v)
    u_neighbors = set(g.neighbors(u))
    v_neighbors = set(g.neighbors(v))
    result = set(u_neighbors.intersection(v_neighbors))
    if g.has_edge(u, v):
        result.discard(u)
        result.discard(v)
    return result


def common_friends_count(u, v, g):
    # считаем общих друзей для двух пользователей
    # (исключая самих этих пользователей, если есть связь от u к v)
    result = common_friends(u, v, g)
    return len(result)


def user_common_friends_counts(u, g):
    result = []
    candidates = friends_of_friends(u, g)
    candidates.discard(u)
    for candidate in candidates:
        result.append((candidate, common_friends_count(u, candidate, g)))
    return result


def recommend_by_common_friends_count(u, g, count):
    friends_counts = user_common_friends_counts(u, g)
    friends_counts_sorted = sorted(friends_counts, key=itemgetter(1), reverse=True)
    result = [el[0] for el in friends_counts_sorted[:count]]
    return result


def k_most_popular_users(graph, k=5):
    return [val[0] for val in sorted(graph.in_degree(), key=itemgetter(1), reverse=True)][:k]


def create_train_test(initial_graph, min_friends_for_test=5, test_users_size=0.2, negative_samples_per_user=20):
    # словарь исходящих связей для пользователей
    graph_friends_count = initial_graph.out_degree()
    # выберем только тех, у кого больше k+1 друзей (исходящих ребер)
    users_more_k_friends = list(set([k for (k, v) in graph_friends_count if v > min_friends_for_test + 1]))
    users_count_for_test = int(len(users_more_k_friends) * test_users_size)
    # окончательно выбираем пользователей для теста
    users_for_test = random.sample(users_more_k_friends, users_count_for_test)
    # и выбираем их друзей, которых уберем из трейна и положим в тест
    test_edges_positive = []
    for user in users_for_test:
        u_edges = initial_graph.out_edges(user)

        u_edges_times = list([initial_graph[u][v]['time'] for (u, v) in u_edges])
        u_edges_sorted = sorted([x for x, _ in zip(u_edges, u_edges_times)], key=itemgetter(1), reverse=False)
        test_edges_positive = list(test_edges_positive + u_edges_sorted[:min_friends_for_test])

    test_edges_positive_inv = list([(v, u) for u, v in test_edges_positive if initial_graph.has_edge(v, u)])
    test_edges_positive = test_edges_positive + test_edges_positive_inv

    train_graph = initial_graph.copy()
    train_graph.remove_edges_from(test_edges_positive)

    # Сформируем набор негативных примеров
    train_edges_negative = []
    # # j = 0
    # при этом не будем использовать тех пользователей, которых мы отобрали для теста
    # и тех, у кого нет друзей
    users_no_friends = set([k for (k, v) in graph_friends_count if v < 1])
    users_to_exclude = set(users_no_friends).union(set(users_for_test))
    nodes_list = set(initial_graph.nodes()).difference(users_to_exclude)
    for u in nodes_list:
        i = 0
        candidates = set(friends_of_friends(u, initial_graph))
        # кандидаты - не соседи
        candidates = candidates.intersection(set(nx.non_neighbors(initial_graph, u)))
        # уберем лишних
        candidates = candidates.difference(users_to_exclude)

        # добавим подписчиков
        candidates = candidates.union(set(initial_graph.predecessors(u)))
        candidates = candidates.difference(set(initial_graph.successors(u)))

        if negative_samples_per_user < len(candidates):
            candidates_samples = random.sample(candidates, negative_samples_per_user)
        else:
            candidates_samples = candidates

        for v in candidates_samples:

            train_edges_negative.append((u, v))
            i += 1

        # j += 1
        # if j % 1000 == 0:
        #     print(j)
    return train_graph, train_edges_negative, test_edges_positive
##########################################################


def common_friends_mean_sum_intensity(u, v, g):
    # для двух людей ищем общих друзей, и считаем по ним среднюю интенсивность взаимодействия
    mean_intensity = 0.0
    sum_intensity = 0.0
    n = 0
    friends = common_friends(u, v, g)
    friends.add(u)
    friends.add(v)
    subg = nx.subgraph(g, friends)
    intensity = nx.get_edge_attributes(subg, 'intensity')
    if intensity:
        intensity = list(intensity.values())
        n = len(intensity)
        intensity = np.sum(intensity)
        if g.has_edge(u, v):
            intensity = intensity - subg.edges[u, v]['intensity']
            n = n - 1
        if n == 0:
            mean_intensity = 0.0
        else:
            mean_intensity = 1.0 * intensity / n
        sum_intensity = intensity
    return mean_intensity, sum_intensity


def common_friends_mean_sum_time(u, v, g):
    # для двух людей ищем общих друзей, и считаем по ним среднюю интенсивность взаимодействия
    result = 0
    mean_time = 0.0
    sum_time = 0.0
    n = 0
    friends = common_friends(u, v, g)
    friends.add(u)
    friends.add(v)
    subg = nx.subgraph(g, friends)
    time = nx.get_edge_attributes(subg, 'time')
    if not time:
        mean_time = 0.0
    else:
        time = list(time.values())
        n = len(time)
        time = np.sum(time)
        if g.has_edge(u, v):
            time = time - subg.edges[u, v]['time']
            n = n - 1
        if n == 0:
            mean_time = 0.0
        else:
            mean_time = 1.0 * time / n
        sum_time = time
    return mean_time, sum_time


def AdamicAdar(u, v, undirected_g):
    edges_removed = list([])
    edge_removed_attrs = {}
    if undirected_g.has_edge(u, v):
        edges_removed.append((u, v))
        edge_removed_attrs = undirected_g[u][v]
        undirected_g.remove_edges_from(edges_removed)

    aa = nx.adamic_adar_index(undirected_g, [(u, v)])
    aa = np.mean([el[2] for el in aa])


    if edges_removed:
        undirected_g.add_edges_from(edges_removed)
        undirected_g[u][v]['intensity'] = edge_removed_attrs['intensity']
        undirected_g[u][v]['time'] = edge_removed_attrs['time']

    return aa


def ResourceAllocation(u, v, undirected_g):
    edges_removed = list([])
    edge_removed_attrs = {}
    if undirected_g.has_edge(u, v):
        edges_removed.append((u, v))
        edge_removed_attrs = undirected_g[u][v]
        undirected_g.remove_edges_from(edges_removed)

    rai = nx.resource_allocation_index(undirected_g, [(u, v)])
    rai = np.mean([el[2] for el in rai])

    if edges_removed:
        undirected_g.add_edges_from(edges_removed)
        undirected_g[u][v]['intensity'] = edge_removed_attrs['intensity']
        undirected_g[u][v]['time'] = edge_removed_attrs['time']

    return rai


def JaccardCoefficent(u, v, undirected_g):
    edges_removed = list([])
    edge_removed_attrs = {}
    if undirected_g.has_edge(u, v):
        edges_removed.append((u, v))
        edge_removed_attrs = undirected_g[u][v]
        undirected_g.remove_edges_from(edges_removed)

    jc = nx.jaccard_coefficient(undirected_g, [(u, v)])
    jc = np.mean([el[2] for el in jc])

    if edges_removed:
        undirected_g.add_edges_from(edges_removed)
        undirected_g[u][v]['intensity'] = edge_removed_attrs['intensity']
        undirected_g[u][v]['time'] = edge_removed_attrs['time']

    return jc


def PreferentialAttachment(u, v, undirected_g):
    edges_removed = list([])
    edge_removed_attrs = {}
    if undirected_g.has_edge(u, v):
        edges_removed.append((u, v))
        edge_removed_attrs = undirected_g[u][v]
        undirected_g.remove_edges_from(edges_removed)

    pa = nx.jaccard_coefficient(undirected_g, [(u, v)])
    pa = np.mean([el[2] for el in pa])

    if edges_removed:
        undirected_g.add_edges_from(edges_removed)
        undirected_g[u][v]['intensity'] = edge_removed_attrs['intensity']
        undirected_g[u][v]['time'] = edge_removed_attrs['time']

    return pa


def ShortestPathLength(u, v, g, weighted = False):
    edges_removed = list([])
    edge_removed_attrs_uv = {}
    edge_removed_attrs_vu = {}
    if g.has_edge(u, v):
        edges_removed.append((u, v))
        edge_removed_attrs_uv = g[u][v]
    if g.has_edge(v, u):
        edges_removed.append((v, u))
        edge_removed_attrs_vu = g[v][u]
    if edges_removed:
        g.remove_edges_from(edges_removed)

    try:
        if weighted:
            shpl = nx.shortest_path_length(g, u, v, 'intensity')
        else:
            shpl = nx.shortest_path_length(g, u, v)
    except:
        shpl = 0

    if edges_removed:
        g.add_edges_from(edges_removed)
        if g.has_edge(u, v):
            g[u][v]['intensity'] = edge_removed_attrs_uv['intensity']
            g[u][v]['time'] = edge_removed_attrs_uv['time']
        if g.has_edge(v, u):
            g[v][u]['intensity'] = edge_removed_attrs_vu['intensity']
            g[v][u]['time'] = edge_removed_attrs_vu['time']

    return shpl

