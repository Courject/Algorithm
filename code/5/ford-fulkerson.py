#!/usr/bin/python3
# Ford-Fulkerson Algorithm
# Author: HongXin
# 2016.12.13

import networkx as nx
import numpy as np

class FordFulkerson:
    def __init__(self, graph, s, t):
        self.__source = s
        self.__target = t
        if type(graph) is nx.classes.digraph.DiGraph:
            self.__graph = graph
        else:
            self.__graph = nx.DiGraph()
            self.__graph.add_weighted_edges_from(graph)
        self.flow = nx.DiGraph()
        self.f = 0
        self.__step = 0

    def get_max_flow(self, mid=True):
        # mid - True for print intermediate process of flow, False for not.
        while True:
            path = self.__get_a_path()
            if path is None:
                break
            bottleneck = self.__bottleneck(path)
            self.__construct_residual_graph(path, bottleneck)
            self.__update_flow(path, bottleneck)
            if mid:
                self.__print_flow()
        return self.f

    def __get_a_path(self):
        # Return a random path from source to target, None for no available path.
        # also can use dijkstra algorithm here to get a shortest path
        # the method which I used here really get a random path from source to target
        paths = list(nx.all_simple_paths(self.__graph, self.__source, self.__target))
        return None if len(paths) == 0 else paths[0]

    def __bottleneck(self, path_nodes):
        # Find the minimum edge of given path
        return min(map(lambda i: self.__graph.get_edge_data(path_nodes[i], path_nodes[i + 1])['weight'],
                       range(len(path_nodes) - 1)))

    def __construct_residual_graph(self, path_nodes, bottleneck):
        # Construct residual graph.
        # Update(decrease) capacity first and update(increase or add new) backward edges.
        for i in range(len(path_nodes) - 1):
            start, end = path_nodes[i], path_nodes[i + 1]
            self.__graph[start][end]['weight'] -= bottleneck
            if self.__graph[start][end]['weight'] == 0:
                self.__graph.remove_edge(start, end)
            if self.__graph.has_edge(end, start):
                self.__graph[end][start]['weight'] += bottleneck
            else:
                self.__graph.add_edge(end, start, weight=bottleneck)

    def __update_flow(self, path_nodes, bottleneck):
        self.f += bottleneck
        # Update flow.
        for i in range(len(path_nodes) - 1):
            start, end = path_nodes[i], path_nodes[i + 1]
            if self.flow.has_edge(start, end):
                self.flow[start][end]['weight'] += bottleneck
            elif self.flow.has_edge(end, start):
                remain = self.flow[end][start]['weight'] - bottleneck
                if remain > 0:
                    self.flow[end][start]['weight'] = remain
                else:
                    self.flow.remove_edge(end, start)
                    if remain < 0:
                        self.flow.add_edge(start, end, weight=-remain)
            else:
                self.flow.add_edge(start, end, weight=bottleneck)

    def __print_flow(self):
        # Example: Step 0 : 1 -> [('s', 'u', {'weight': 1}), ('u', 'v', {'weight': 1}), ('v', 't', {'weight': 1})]
        if self.__step == 0:
            print('Intermediate Process: (max flow & detail):')
        print('Step', self.__step, ':', self.f, '->', self.flow.edges(data=True))
        self.__step += 1


# if __name__ == '__main__':
#     edge_list = [('s', 'u', 1), ('s', 'v', 1), ('u', 'v', 1), ('u', 't', 1), ('v', 't', 1)]
#     ford = FordFulkerson(edge_list, 's', 't')
#     print('Max Flow:', ford.get_max_flow())

# def file2problems(path):
#     f = open(path, 'r')
#     problems, job_count, job = [], 0, 0
#     for line in f:
#         line = line.strip()
#         if not(line.startswith('#') or line == ''):
#             if job > job_count or job == 0:
#                 if job != 0:
#                     problems.append(graph)
#                 job_count = int(line.split(' ')[0])
#                 job = 1
#                 graph = nx.DiGraph()
#             else:
#                 computers = line.split(' ')
#                 graph.add_edge('s', 'j'+str(job), weight=1)
#                 graph.add_edge('j'+str(job), 'c'+computers[0], weight=1)
#                 graph.add_edge('j'+str(job), 'c'+computers[1], weight=1)
#                 graph.add_edge('c'+computers[0], 't', weight=job_count)
#                 graph.add_edge('c'+computers[1], 't', weight=job_count)
#                 job += 1
#     problems.append(graph)
#     return problems
#
#
# def min_load(graph):
#     job_count = len(graph.edges('s'))
#     x, y = 0, job_count
#     while True:
#         load = (x+y)//2
#         for edge in graph.in_edges('t', data=True):
#             edge[2]['weight'] = load
#         ford = FordFulkerson(graph.copy(), 's', 't')
#         max_flow = ford.get_max_flow()
#         if max_flow == job_count:
#             y = load
#         elif y == x + 1:
#             return y
#         else:
#             x = load
#
# if __name__ == '__main__':
#     _list = file2problems('problem1.data')
#     for problem in _list:
#         print('Min Load: ', min_load(problem))
#         input()


def file2problems(path):
    f = open(path, 'r')
    problems, line_num = [], 0
    for line in f:
        line = line.strip()
        if not (line.startswith('#') or line == ''):
            if line_num == 0:
                m, n = map(int, line.split(' '))

            if line_num > 2:
                line_num = 0
                graph = nx.DiGraph()
                for i in range(m):
                    graph.add_edge('s', 'b' + str(i), weight=b[i])
                for j in range(n):
                    graph.add_edge('c' + str(j), 't', weight=c[j])
                for i in range(m):
                    for j in range(n):
                        graph.add_edge('b' + str(i), 'a' + str(i) + str(j), weight=1)
                        graph.add_edge('a' + str(i) + str(j), 'c' + str(j), weight=1)
                problems.append((graph, m, n))
                m, n = map(int, line.split(' '))
            elif line_num == 1:
                b = list(map(int, line.split(' ')))
            else:
                c = list(map(int, line.split(' ')))

            line_num += 1

    problems.append(graph)
    return problems


def flow2matrix(flow, m, n):
    matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if flow.has_node('a' + str(i) + str(j)) \
                    and flow.has_edge('a' + str(i) + str(j), 'c' + str(j))\
                    and flow.edge['a' + str(i) + str(j)][ 'c' + str(j)]['weight'] > 0:
                matrix[i][j] = 1
    return matrix


if __name__ == '__main__':
    _list = file2problems('problem2.data')
    for _graph, _m, _n in _list:
        push = FordFulkerson(_graph, 's', 't')
        push.get_max_flow(True)
        print(flow2matrix(push.pre_flow, _m, _n))