from array import array
from itertools import chain
import numpy as np


EMPTY = -1
MASK = (np.int32(1) << np.int32(31)).tolist()


def del_index(x):
    return MASK ^ x


class Graph(object):
    def __init__(self, directed=False):
        self._directed = directed
        self._num_nodes = 0
        self._num_edges = 0

        self._node_first_available = EMPTY

        self._out_degree = []
        self._out_nodes_first_available = []
        self._out_nodes = []
        self._out_node_indices = []

        if directed:
            self._in_degree = []
            self._in_nodes_first_available = []
            self._in_nodes = []
            self._in_node_indices = []
        else:
            self._in_degree = self._out_degree
            self._in_nodes_first_available = self._out_nodes_first_available
            self._in_nodes = self._out_nodes
            self._in_node_indices = self._out_node_indices

    @property
    def directed(self):
        return self._directed

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
        if not self.directed:
            if node is not None:
                return self._out_degree[node]
            else:
                return self.out_degree()
        else:
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
        if self.directed:
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
        self._node_first_available = index
        return num_nodes

    def _append_nodes(self, num_nodes):
        self._out_nodes.extend(array('l') for _i in xrange(num_nodes))
        self._out_node_indices.extend(array('l') for _i in xrange(num_nodes))
        self._out_nodes_first_available.extend(EMPTY for _i in xrange(num_nodes))
        self._out_degree.extend(0 for _i in xrange(num_nodes))

        if self.directed:
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

    def add_edges(self, src_tgt_pairs):
        for src, tgt in src_tgt_pairs:
            # There are no gaps to be filled in the out_nodes array
            if self._out_nodes_first_available[src] == EMPTY:
                # out_node_index is the array index in self._out_nodes that this
                # edge will occupy; the end of self._out_nodes, in this case
                out_node_index = len(self._out_nodes[src])
                # Append the target to _out_nodes
                self._out_nodes[src].append(tgt)
                # in_node_index is the array index in self._in_nodes that this edge
                # will be occupy
                # Note that although this line appears in both branches of the
                # conditional, it must appear after modification to self._out_nodes,
                # as for undirected graphs, self._out_nodes == self._in_nodes.
                in_node_index = (len(self._in_nodes[tgt])
                                 if self._in_nodes_first_available[tgt] == EMPTY
                                 else del_index(self._in_nodes_first_available[tgt]))
                # Append the position that our other end-point has in self._in_nodes
                self._out_node_indices[src].append(in_node_index)
            # There are gaps to be filled in the source list
            else:
                # Get first open index
                out_node_index = del_index(self._out_nodes_first_available[src])
                # Change the first open index to the next available open index
                self._out_nodes_first_available[src] = self._out_nodes[src][out_node_index]
                self._out_nodes[src][out_node_index] = tgt
                # in_node_index is the array index in self._in_nodes that this edge
                # will be occupy
                # Note that although this line appears in both branches of the
                # conditional, it must appear after modification to self._out_nodes,
                # as for undirected graphs, self._out_nodes == self._in_nodes.
                in_node_index = (len(self._in_nodes[tgt])
                                 if self._in_nodes_first_available[tgt] == EMPTY
                                 else del_index(self._in_nodes_first_available[tgt]))
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
        self._in_degree[node] =  self._node_first_available
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
            print "[", pos_index, " > ", del_index(self._out_degree[pos_index]), "]"
            index = self._out_degree[pos_index]

