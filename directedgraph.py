from array import array
from itertools import chain
import numpy as np


EMPTY = -1
MASK = (np.int32(1) << np.int32(31)).tolist()


def del_index(x):
    return MASK ^ x


class DirectedGraph(object):
    def __init__(self):
        self._num_nodes = 0
        self._num_edges = 0

        self._node_first_available = EMPTY

        self._out_degree = []
        self._out_nodes_first_available = []
        self._out_nodes = []
        self._out_node_indices = []

        self._in_degree = []
        self._in_nodes_first_available = []
        self._in_nodes = []
        self._in_node_indices = []

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return self._num_edges

    def nodes(self):
        for u in xrange(len(self._out_nodes)):
            if self._out_degree[u] >= 0:
                yield u

    def out_degree(self, node=None):
        if node is not None:
            return self._out_degree[node]
        else:
            return (deg for deg in self._out_degree if deg >= 0)

    def in_degree(self, node=None):
        if node is not None:
            return self._in_degree[node]
        else:
            return (deg for deg in self._in_degree if deg >= 0)

    def degree(self, node=None):
        if node is not None:
            return self._out_degree[node] + self._in_degree[node]
        else:
            return chain(self.out_degree(), self.in_degree())

    def out_neighbors(self, node):
        for n in self._out_nodes[node]:
            if n >= 0:
                yield n

    def in_neighbors(self, node):
        for n in self._in_nodes[node]:
            if n >= 0:
                yield n

    def neighbors(self, node):
        for n in self._out_nodes[node]:
            if n >= 0:
                yield n
        for n in self._in_nodes[node]:
            if n >= 0:
                yield n

    def _fill_empty_nodes(self, num_nodes):
        index = self._node_first_available
        while index != EMPTY and num_nodes > 0:
            pos_index = del_index(index)
            index = self._out_degree[pos_index]
            self._out_degree[pos_index] = 0
            self._in_degree[pos_index] = 0
            num_nodes -= 1
        if index != EMPTY:
            self._in_degree[del_index(index)] = EMPTY
        self._node_first_available = index
        return num_nodes

    def _append_nodes(self, num_nodes):
        self._out_nodes.extend(array('l') for _i in xrange(num_nodes))
        self._out_node_indices.extend(array('l') for _i in xrange(num_nodes))
        self._out_nodes_first_available.extend(EMPTY for _i in xrange(num_nodes))
        self._out_degree.extend(0 for _i in xrange(num_nodes))

        self._in_nodes.extend(array('l') for _i in xrange(num_nodes))
        self._in_node_indices.extend(array('l') for _i in xrange(num_nodes))
        self._in_nodes_first_available.extend(EMPTY for _i in xrange(num_nodes))
        self._in_degree.extend(0 for _i in xrange(num_nodes))

    def add_nodes(self, num_nodes):
        remaining_nodes = self._fill_empty_nodes(num_nodes)
        self._append_nodes(remaining_nodes)
        self._num_nodes += num_nodes

    def add_edge(self, src, tgt):
        self.add_edges([(src, tgt)])

#    def _undelete_node(self, node):
#        self._node_next_available[node]

    def add_edges(self, src_tgt_pairs):
        for src, tgt in src_tgt_pairs:
            # in_node_index is the array index in self._in_nodes that this edge
            # will be occupy
            in_node_index = (len(self._in_nodes[tgt])
                             if self._in_nodes_first_available[tgt] == EMPTY
                             else del_index(self._in_nodes_first_available[tgt]))

            # There are no gaps to be filled in the out_nodes array
            if self._out_nodes_first_available[src] == EMPTY:
                # out_node_index is the array index in self._out_nodes that this
                # edge will occupy; the end of self._out_nodes, in this case
                out_node_index = len(self._out_nodes[src])
                # Append the target to _out_nodes
                self._out_nodes[src].append(tgt)
                # Append the position that our other end-point has in self._in_nodes
                self._out_node_indices[src].append(in_node_index)
            # There are gaps to be filled in the source list
            else:
                # Get first open index
                out_node_index = del_index(self._out_nodes_first_available[src])
                # Change the first open index to the next available open index
                self._out_nodes_first_available[src] = self._out_nodes[src][out_node_index]
                self._out_nodes[src][out_node_index] = tgt
                self._out_node_indices[src][out_node_index] = in_node_index
            self._out_degree[src] += 1

            # There are no gaps to be filled in the in_nodes array
            if self._in_nodes_first_available[tgt] == EMPTY:
                # Append the source to _in_nodes
                self._in_nodes[tgt].append(src)
                # Append the position that our other end-point has in self._out_nodes
                self._in_node_indices[tgt].append(out_node_index)
            # There are gaps to be filled in the source list
            else:
                # Change the first open index to the next available open index
                self._in_nodes_first_available[tgt] = self._in_nodes[tgt][in_node_index]
                self._in_nodes[tgt][in_node_index] = src
                self._in_node_indices[tgt][in_node_index] = out_node_index
            self._in_degree[tgt] += 1

            self._num_edges += 1

    def del_edges(self, edges):
        for tgt, tgt_index in edges:
            src = self._in_nodes[tgt][tgt_index]
            src_index = self._in_node_indices[tgt][tgt_index]

            self._in_nodes[tgt][tgt_index] = self._in_nodes_first_available[tgt]
            self._in_nodes_first_available[tgt] = del_index(tgt_index)
            self._in_degree[tgt] -= 1

            self._out_nodes[src][src_index] = self._out_nodes_first_available[src]
            self._out_nodes_first_available[src] = del_index(src_index)
            self._out_degree[src] -= 1

            self._num_edges -= 1

    def clear_node(self, node):
        for i in xrange(len(self._out_nodes[node])):
            tgt = self._out_nodes[node][i]
            if tgt >= 0:
                tgt_index = self._out_node_indices[node][i]

                self._in_nodes[tgt][tgt_index] = self._in_nodes_first_available[tgt]
                self._in_nodes_first_available[tgt] = del_index(tgt_index)
                self._in_degree[tgt] -= 1

            self._out_nodes[node][i] = del_index(i + 1)

        if len(self._out_nodes[node]) > 0:
            self._out_nodes_first_available[node] = del_index(0)
            self._out_nodes[node][-1] = EMPTY
        self._num_edges -= self._out_degree[node]
        self._out_degree[node] = 0

        for i in xrange(len(self._in_nodes[node])):
            src = self._in_nodes[node][i]
            if src >= 0:
                src_index = self._in_node_indices[node][i]

                self._out_nodes[src][src_index] = self._out_nodes_first_available[src]
                self._out_nodes_first_available[src] = del_index(src_index)
                self._out_degree[src] -= 1

            self._in_nodes[node][i] = del_index(i + 1)

        if len(self._in_nodes[node]) > 0:
            self._in_nodes_first_available[node] = del_index(0)
            self._in_nodes[node][-1] = EMPTY
        self._num_edges -= self._in_degree[node]
        self._in_degree[node] = 0

    def del_node(self, node):
        self.clear_node(node)

        self._out_degree[node] = self._node_first_available
        self._in_degree[node] = EMPTY
        if self._node_first_available != EMPTY:
            self._in_degree[del_index(self._node_first_available)] = del_index(node)
        self._node_first_available = del_index(node)
        self._num_nodes -= 1

    def source(self, edge):
        return self._in_nodes[edge[0]][edge[1]]

    def target(self, edge):
        return edge[0]

    def out_edges(self, src):
        for i, tgt in enumerate(self._out_nodes[src]):
            if tgt >= 0:
                yield (tgt, self._out_node_indices[src][i])

    def in_edges(self, tgt):
        for i, src in enumerate(self._in_nodes[tgt]):
            if src >= 0:
                yield (tgt, i)

    def incident(self, node):
        return chain(self.out_edges(node), self.in_edges(node))

    def _print_available_nodes(self):
        index = self._node_first_available
        print del_index(self._node_first_available), " -> "
        while index != EMPTY:
            pos_index = del_index(index)
            print "[", del_index(self._in_degree[pos_index]), " < ", pos_index, " > ", del_index(self._out_degree[pos_index]), "]"
            index = self._out_degree[pos_index]

def get_adjacency():
    return [[0, 1, 3, 4, 6, 8],
            [0, 1, 2, 4, 5, 7],
            [0, 3, 4, 6, 7, 8],
            [],
            [],
            [],
            [],
            [],
            []]

def get_in_adjacency():
    return [[0, 1, 2],
            [0, 1],
            [1],
            [0, 2],
            [0, 1, 2],
            [1],
            [0, 2],
            [1, 2],
            [0, 2]]

def get_edges():
    for u, neighbors in enumerate(get_adjacency()):
        for v in neighbors:
            yield u, v


def get_graph():
    g = DirectedGraph()
    g.add_nodes(9)
    g.add_edges(get_edges())
    return g


def basic_invariants(g):
    assert 2 * g.num_edges == sum(g.out_degree()) + sum(g.in_degree())

    for u in g.nodes():
        out_edges = list(g.out_edges(u))
        assert all(g.source(e) == u for e in out_edges)
        assert [g.target(e) for e in out_edges] == [v for v in g._out_nodes[u] if v >= 0]

    for u in g.nodes():
        in_edges = list(g.in_edges(u))
        assert all(g.target(e) == u for e in in_edges)
        assert [g.source(e) for e in in_edges] == [v for v in g._in_nodes[u] if v >= 0]


def test_basic():
    g = get_graph()

    assert list(g.out_degree()) == [6, 6, 6, 0, 0, 0, 0, 0, 0]
    assert list(g.in_degree()) == [3, 2, 1, 2, 3, 1, 2, 2, 2]

    basic_invariants(g)


def test_edge_delete_1():
    g = get_graph()

    g.del_edges([(0, 1)])
    basic_invariants(g)


def test_edge_delete_2():
    g = get_graph()

    g.del_edges([(0, 1)])
    g.add_edges([(1, 0)])
    basic_invariants(g)


def test_edge_delete_3():
    g = get_graph()

    g.del_edges([(0, 0)])
    basic_invariants(g)


def test_edge_delete_4():
    g = get_graph()

    g.del_edges([(0, 0)])
    g.add_edges([(0, 2), (5, 0)])
    basic_invariants(g)


def test_del_node_1():
    g = get_graph()

    g.del_node(1)
    basic_invariants(g)


def test_del_node_2():
    g = get_graph()

    g.del_node(1)
    g.del_node(3)
    basic_invariants(g)


def test_del_add_node_1():
    g = get_graph()

    N = g.num_nodes
    g.del_node(1)
    assert g.num_nodes == N - 1
    g.add_nodes(1)
    assert g.num_nodes == N
    basic_invariants(g)


def test_del_add_node_2():
    g = get_graph()

    N = g.num_nodes
    g.del_node(1)
    assert g.num_nodes == N - 1
    g.add_nodes(2)
    assert g.num_nodes == N + 1
    basic_invariants(g)


def test_del_add_node_3():
    g = get_graph()

    N = g.num_nodes
    g.del_node(1)
    assert g.num_nodes == N - 1
    g.del_node(3)
    assert g.num_nodes == N - 2
    g.add_nodes(4)
    assert g.num_nodes == N + 2
    basic_invariants(g)


def test_del_add_node_4():
    g = get_graph()

    for n in [3, 7, 1, 8, 2, 6, 4, 5, 0]:
        g.del_node(n)
    assert g.num_nodes == 0
    g.add_nodes(9)
    assert g.num_nodes == 9
    g.add_edges(get_edges())
    basic_invariants(g)


def test_del_add_node_5():
    g = get_graph()

    for n in [3, 7, 1, 8, 2, 6, 4, 5, 0]:
        g.del_node(n)
    g.add_nodes(4)
    basic_invariants(g)

#def test_undirected_neighbors():
#    neighbors = get_adjacency()
#    g = get_undirected_graph()
#    assert [sorted(list(g.neighbors(i))) for i in xrange(20)] == neighbors
#
#def test_undirected_incident():
#    g = get_undirected_graph()
#    adj = get_adjacency()
#    for node, neighbors in enumerate(adj):
#        node_degree = len(neighbors)
#        assert sorted([g.source(e) for e in g.out_edges(node)]) == node_degree * [node]
#        assert sorted([g.target(e) for e in g.out_edges(node)]) == neighbors
#        assert sorted([g.source(e) for e in g.in_edges(node)]) == neighbors
#        assert sorted([g.target(e) for e in g.in_edges(node)]) == node_degree * [node]

if __name__ == '__main__':
    test_basic()
    test_edge_delete_1()
    test_edge_delete_2()
    test_edge_delete_3()
    test_edge_delete_4()
    test_del_node_1()
    test_del_node_2()
    test_del_add_node_1()
    test_del_add_node_2()
    test_del_add_node_3()
    test_del_add_node_4()
    test_del_add_node_5()

# def test_add_edges():
# 
#     g = 
