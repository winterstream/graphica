from graph import Graph

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
    g = Graph(True)
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
