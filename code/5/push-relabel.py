#!/usr/bin/python3
# Push-relabel Algorithm
# Author: HongXin
# 2016.12.14

import networkx as nx
import numpy as np


class PushRelabel:
    def __init__(self, graph, s, t):
        self.__source = s
        self.__target = t
        if type(graph) is nx.classes.digraph.DiGraph:
            self.__graph = graph
        else:
            self.__graph = nx.DiGraph()
            self.__graph.add_weighted_edges_from(graph)
        self.pre_flow = nx.DiGraph()
        self.f = 0
        self.__step = 0

    def get_max_flow(self, mid=True):
        self.__init_node()
        self.__init_pre_flow()
        while True:
            option = False
            for node in self.pre_flow.nodes():
                if self.__excess(node) > 0 and node != self.__source and node != self.__target:
                    edge = self.__get_high_start_edge(node)
                    if edge is None:
                        if self.__relabel(node):
                            if mid:
                                self.__print_label()
                            option = True
                            break
                        else:
                            continue
                    else:
                        bottleneck = self.__push(edge)
                        self.__construct_residual_graph(edge, bottleneck)
                        if mid:
                            self.__print_push()
                        option = True
                        break
            if not option:
                break

        self.f = self.__excess(self.__target)
        return self.f

    def __init_node(self):
        # Init height. Assign source with n and others with 0.
        for node, data in self.__graph.nodes_iter(True):
            data['height'] = 0
        self.__graph.node[self.__source]['height'] = len(self.__graph.node)

    def __init_pre_flow(self):
        # Init pre_flow. Assign edges start from source with its capacity.
        self.pre_flow.add_edges_from(self.__graph.edges(self.__source, data=True))
        # Construct residual graph
        for start, end, data in self.__graph.edges(self.__source, data=True):
            self.__graph.add_edge(end, start, weight=data['weight'])
            self.__graph.add_edge(start, end, weight=0)

    def __excess(self, node):
        into, out, = 0, 0
        for edge in self.pre_flow.in_edges(node, data=True):
            into += edge[2]['weight']
        for edge in self.pre_flow.out_edges(node, data=True):
            out += edge[2]['weight']
        return into - out

    def __get_high_start_edge(self, node):
        for start, end, data in self.__graph.edges(node, data=True):
            if data['weight'] > 0 and self.__graph.node[start]['height'] > self.__graph.node[end]['height']:
                return start, end, data
        return None

    def __push(self, edge):
        start, end, weight = edge[0], edge[1], edge[2]['weight']
        excess = self.__excess(start)
        if self.pre_flow.has_edge(start, end):
            bottleneck = min(weight, excess)
            self.pre_flow.edge[start][end]['weight'] += bottleneck
        elif self.pre_flow.has_edge(end, start):
            bottleneck = min(self.pre_flow[end][start]['weight'], excess)
            if bottleneck == self.pre_flow[end][start]['weight']:
                self.pre_flow.remove_edge(end, start)
            else:
                self.pre_flow.edge[end][start]['weight'] -= bottleneck
        else:
            bottleneck = min(weight, excess)
            self.pre_flow.add_edge(start, end, weight=bottleneck)
        return bottleneck

    def __construct_residual_graph(self, edge, bottleneck):
        start, end = edge[0], edge[1]
        self.__graph.edge[start][end]['weight'] -= bottleneck
        if self.__graph.has_edge(end, start):
            self.__graph.edge[end][start]['weight'] += bottleneck
        else:
            self.__graph.add_edge(end, start, weight=bottleneck)

    def __relabel(self, node):
        min_height = self.__graph.node[node]['height']
        for edge in self.__graph.edges(node):
            min_height = min(self.__graph.node[edge[1]]['height'], min_height)
        if self.__graph.node[node]['height'] > min_height:
            return False
        else:
            self.__graph.node[node]['height'] = min_height + 1
            return True

    def __print_label(self):
        print('Step', self.__step, '(Relabel):', self.__graph.node)
        self.__step += 1

    def __print_push(self):
        print('Step', self.__step, '(Push):', self.pre_flow.edge)
        self.__step += 1


# if __name__ == '__main__':
#     # edge_list = [('s', 'u', 1), ('s', 'v', 1), ('u', 'v', 1), ('u', 't', 1), ('v', 't', 1)]
#     edge_list = [('s','u',3),('u','v',1),('v','t',2)]
#     ford = PushRelabel(edge_list, 's', 't')
#     print('Max Flow:', ford.get_max_flow())


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
        push = PushRelabel(_graph, 's', 't')
        push.get_max_flow(False)
        print(flow2matrix(push.pre_flow, _m, _n))
