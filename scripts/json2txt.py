import json
import sys


def ser_node(node, move, depth):
    header = depth * "  "
    if node is None:
        node_string = ser_unvisited(depth)
    else:
        node_string = ser_visited(node, depth)
    spacer = "\t" if len(node_string) > 0 else ""
    move_string = move['move']
    return f'{header}- {move_string}{spacer}{node_string}'


def ser_unvisited(depth):
    return ""


def ser_visited(node, depth):
    shared = (f"{node['handle']}\t{node['score']}\t{node['q']}\t"
              f"{node['n']}\t{node['mvd']}\t{node['pruned']}")
    if node["state"]["type"] == "revision":
        return f"{shared}\t{ser_revision(node['state'], depth)}"
    if node["state"]["type"] == "terminal":
        return f"{shared}\t{ser_terminal(node['state'], depth)}"
    return "unknown"


def ser_revision(state, depth):
    trs_string = state['trs'].replace('\n', ' ')
    return f"{state['type']}\t\"{trs_string}\""


def ser_terminal(state, depth):
    trs_string = state['trs'].replace('\n', ' ')
    return f"{state['type']}\t\"{trs_string}\""


if __name__ == "__main__":

    with open(sys.argv[1], 'r') as fd:
        tree = json.load(fd, encoding="utf8")

    rh = tree['root']
    stack = [(0, {"move": "Empty"}, tree['nodes'][rh])]

    with open(sys.argv[2], 'w') as fd:
        while len(stack) > 0:
            depth, move, node = stack.pop()
            fd.write(ser_node(node, move, depth) + "\n")
            if node is not None:
                for mh in reversed(node['out']):
                    new_depth = depth + 1
                    new_move = tree['moves'][mh]
                    ch = new_move['child']
                    new_node = ch if ch is None else tree['nodes'][ch]
                    stack.append((new_depth, new_move, new_node))
