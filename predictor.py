import networkx as nx
import numpy as np
from operator import itemgetter
import pickle
import sn_graph_functions as gf


def save_to_pickle(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_from_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data


def user_in_graph(user_identifier, graph):
    if user_identifier in graph.nodes():
        return True
    else:
        return False


def create_features(edges, g, undir_g, add_answers=False, initial_graph=None):

    edges = list(edges)

    # feature_common_friends_count = np.array([gf.common_friends_count(e[0], e[1], g) for e in edges])
    feature_common_friends_mean_sum_intensity = [gf.common_friends_mean_sum_intensity(e[0], e[1], g) for e in edges]
    feature_common_friends_mean_sum_time = [gf.common_friends_mean_sum_time(e[0], e[1], g) for e in edges]

    feature_common_friends_mean_intensity = [val[0] for val in feature_common_friends_mean_sum_intensity]
    feature_common_friends_sum_intensity = [val[1] for val in feature_common_friends_mean_sum_intensity]
    feature_common_friends_mean_time = [val[0] for val in feature_common_friends_mean_sum_time]
    # feature_common_friends_sum_time = [val[1] for val in feature_common_friends_mean_sum_time]

    # feature_ShortestPathLength = np.array([gf.ShortestPathLength(e[0], e[1], g, weighted = False) for e in edges])
    # feature_ShortestPathLengthWeighted = np.array([gf.ShortestPathLength(e[0], e[1], g, weighted = True) for e in edges])

    feature_AdamicAdar = np.array([gf.AdamicAdar(e[0], e[1], undir_g) for e in edges])
    feature_ResourceAllocation = np.array([gf.ResourceAllocation(e[0], e[1], undir_g) for e in edges])
    feature_JaccardCoefficent = np.array([gf.JaccardCoefficent(e[0], e[1], undir_g) for e in edges])
    feature_PreferentialAttachment = np.array([gf.PreferentialAttachment(e[0], e[1], undir_g) for e in edges])

    feature_has_back_link = np.array([gf.edge_is_mutual(e[0], e[1], g) for e in edges])

    data = np.column_stack((np.array(feature_common_friends_mean_intensity).T,
                            np.array(feature_common_friends_sum_intensity).T,
                            np.array(feature_common_friends_mean_time).T,
                            #np.array(feature_common_friends_sum_time).T,
                            #np.array(feature_common_friends_count).T,
                            np.array(feature_AdamicAdar).T,
                            np.array(feature_ResourceAllocation).T,
                            np.array(feature_JaccardCoefficent).T,
                            np.array(feature_PreferentialAttachment).T,
                            # np.array(feature_ShortestPathLength).T,
                            # np.array(feature_ShortestPathLengthWeighted).T
                            # np.array(feature_has_back_link).T
                            ))

    if add_answers:
        if initial_graph is None:
            raise ValueError('Initial graph not specified')

        ids1 = np.array([e[0] for e in edges])
        ids2 = np.array([e[1] for e in edges])
        answers = np.array([initial_graph.has_edge(e[0], e[1]) for e in edges])

        data = np.column_stack((data,
                                np.array(ids1).T,
                                np.array(ids2).T,
                                np.array(answers).T
                                ))

    return data


def select_candidates(user_identifier, graph, k=15, type=1):

    if type == 1:
        candidates = set(gf.friends_of_friends(user_identifier, graph))
    else:
        candidates = set(gf.recommend_by_common_friends_count(u=user_identifier, g=graph, count=k))

    # добавим подписчиков
    candidates = candidates.union(set(graph.predecessors(user_identifier)))
    candidates = candidates.difference(set(graph.successors(user_identifier)))

    return candidates


def recommend(user_identifier, graph, undir_graph, estimator, scaler, k=5, pre_selection_type=2, pre_selection_k=30):

    candidates = select_candidates(user_identifier, graph, k=pre_selection_k, type=pre_selection_type)
    edges = []
    for candidate in candidates:
        edges.append((user_identifier, candidate))

    data = create_features(edges=edges, g=graph, undir_g=undir_graph)
    if data.shape[0] > 1:

        data = scaler.transform(data)

        predictions_proba = estimator.predict_proba(data)
        predictions_proba = predictions_proba[:, list(estimator.classes_).index(1)]

        recommendation = list([x for _, x in sorted(zip(predictions_proba, candidates),
                                                      key=itemgetter(0), reverse=True)])

        recommendation = recommendation[:k]
    else:
        recommendation = gf.k_most_popular_users(graph, k=k)

    if len(recommendation) < k:
        recommendation = recommendation + gf.k_most_popular_users(graph, k=(k-len(recommendation)))

    return recommendation


if __name__ == '__main__':

    model = load_from_pickle('data/model_rf')
    clf = model['estimator']
    scaler = model['scaler']

    train_graph = nx.read_gpickle("data/train_graph")
    undir_graph = train_graph.to_undirected()

    print("Type 'exit' to exit the program")

    while True:

        user_id = input('Enter user id: ')
        if user_id == 'exit':
            break

        try:
            user_id = int(user_id)
        except:
            print("Invalid number for user_id. It must be an integer.")
            continue

        if not user_in_graph(user_id, train_graph):
            print("Error! Social network doesn't contain this user id. Try another id.")
            continue

        recommendations = recommend(user_identifier=user_id,
                                    graph=train_graph, undir_graph=undir_graph,
                                    estimator=clf, scaler=scaler)

        print(' '.join(map(str, recommendations)))




