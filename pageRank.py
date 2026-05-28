import csv
import time
import sys
import numpy as np

def parse_csv(filename, dataset_type="normal", undirected=False):
    edges = []
    nodes = set()

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            if dataset_type == "football":
                src, dst = parse_football_row(row)
                edges.append((src, dst))
            else:
                src, dst = parse_row(row)
                edges.append((src, dst))

                if undirected:
                    edges.append((dst, src))

            nodes.add(src)
            nodes.add(dst)

    return nodes, edges

def parse_row(row):
    node1 = row[0].strip().replace('"', '')
    node2 = row[2].strip().replace('"', '')
    return node1, node2

def parse_football_row(row):
    team1 = row[0].strip().replace('"', '')
    score1 = clean_score(row[1])

    team2 = row[2].strip().replace('"', '')
    score2 = clean_score(row[3])

    if score1 > score2:
        return team2, team1
    else:
        return team1, team2

def clean_score(score_string):
    digits = ''.join(char for char in score_string if char.isdigit())
    return int(digits)

def build_indexed_graph(nodes, edges):
    nodes = sorted(nodes)

    node_to_index = {}
    index_to_node = {}

    for i, node in enumerate(nodes):
        node_to_index[node] = i
        index_to_node[i] = node

    adjacency = {i: [] for i in range(len(nodes))}

    for src, dst in edges:
        src_index = node_to_index[src]
        dst_index = node_to_index[dst]
        adjacency[src_index].append(dst_index)

    return node_to_index, index_to_node, adjacency

def build_transition_matrix(adjacency):
    n = len(adjacency)
    matrix = np.zeros((n, n))

    for src in adjacency:
        outgoing = adjacency[src]

        if len(outgoing) == 0:
            matrix[:, src] = 1 / n
        else:
            probability = 1 / len(outgoing)

            for dst in outgoing:
                matrix[dst, src] += probability

    return matrix

def page_rank(matrix, d=0.85, tolerance=0.0001, max_iterations=1000):
    n = matrix.shape[0]

    ranks = np.ones(n) / n

    for iteration in range(max_iterations):
        new_ranks = ((1 - d) / n) + d * matrix.dot(ranks)

        difference = np.sum(np.abs(new_ranks - ranks))

        if difference < tolerance:
            return new_ranks, iteration + 1

        ranks = new_ranks

    return ranks, max_iterations

def print_rankings(ranks, index_to_node):
    ranked_nodes = []

    for i, score in enumerate(ranks):
        ranked_nodes.append((index_to_node[i], score))

    ranked_nodes.sort(key=lambda x: x[1], reverse=True)

    for rank, (node, score) in enumerate(ranked_nodes, start=1):
        print(f"{rank}. {node} with pagerank: {score}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 pageRank.py <filename> <dataset_type> [undirected]")
        print("Example: python3 pageRank.py NCAA_football.csv football")
        print("Example: python3 pageRank.py dolphins.csv normal")
        print("Example: python3 pageRank.py dolphinsDir.csv normal undirected")
        sys.exit(1)

    filename = sys.argv[1]
    dataset_type = sys.argv[2]

    undirected = False
    if len(sys.argv) >= 4 and sys.argv[3] == "undirected":
        undirected = True

    start_read = time.time()

    nodes, edges = parse_csv(filename, dataset_type, undirected)
    node_to_index, index_to_node, adjacency = build_indexed_graph(nodes, edges)
    matrix = build_transition_matrix(adjacency)

    end_read = time.time()

    damping = 0.85
    tolerance = 0.0001
    max_iterations = 1000

    start_process = time.time()

    ranks, iterations = page_rank(matrix, damping, tolerance, max_iterations)

    end_process = time.time()

    print("Dataset:", filename)
    print("Dataset type:", dataset_type)
    print("Undirected:", undirected)
    print("Number of nodes:", len(nodes))
    print("Number of edges:", len(edges))
    print("Damping factor:", damping)
    print("Tolerance:", tolerance)
    print("Max iterations:", max_iterations)
    print("Read time:", end_read - start_read)
    print("Processing time:", end_process - start_process)
    print("Iterations:", iterations)
    print()

    print_rankings(ranks, index_to_node)