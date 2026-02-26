def build_graph_7():
    graph = {
        "X1": ["Y1"],
        "Y1": ["X1", "X3", "Y2"],
        "X3": ["Y1", "X4", "X5"],
        "X4": ["X3"],
        "Y2": ["Y1", "X2", "X5"],
        "X2": ["Y2"],
        "X5": ["Y2", "X3"],
    }
    return graph


def build_types_7():
    types = {
        "X1": "X",
        "X2": "X",
        "X3": "X",
        "X4": "X",
        "X5": "X",
        "Y1": "Y",
        "Y2": "Y",
    }
    return types