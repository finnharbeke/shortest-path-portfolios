from graph import Graph
import re

class Path:
    @staticmethod
    def is_path(path: str, graph: Graph | None = None):
        pattern = r'[a|v]\d+(-\d+)+'
        if graph is None:
            return re.match(pattern, path)
        if not re.match(pattern, path):
            return False
        if path.startswith('v'): bound = graph.n
        else: bound = graph.m

        return all([int(x) in range(bound) for x in path[1:].split('-')])

    @staticmethod
    def to_integers(path: str):
        return [int(x) for x in path[1:].split('-')]

    @staticmethod
    def to_arc_based(path: str, graph: Graph):
        if not Path.is_path(path, graph):
            raise ValueError('paths of incorrect format')
        if path.startswith('a'):
            return path
        else:
            vertices = Path.to_integers(path)
            a_path = 'a'
            for u, v in zip(vertices[:-1], vertices[1:]):
                for a in graph.out_arcs[u]:
                    if (u, v) == graph.arcs[a]:
                        a_path += '-' if len(a_path) > 1 else ''
                        a_path += str(a)
                        break
            return a_path

    @staticmethod
    def to_vertex_based(path: str, graph: Graph):
        if not Path.is_path(path, graph):
            raise ValueError('paths of incorrect format')
        if path.startswith('v'):
            return path
        else:
            arcs = Path.to_integers(path)
            v_path = 'v' + str(graph.arcs[arcs[0]][0])
            for a in arcs:
                v_path += f'-{graph.arcs[a][1]}'
            return v_path
