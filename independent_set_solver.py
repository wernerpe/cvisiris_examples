import numpy as np
from pydrake.all import (MathematicalProgram, SolverOptions, 
			            Solve, CommonSolverOption)

from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix, csr_matrix
#from clique_covers import networkx_to_metis_format
import networkx as nx
import subprocess
_KAMISPATH = "/home/peter/git/KaMIS"

def networkx_to_metis_format(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    metis_lines = [f"{num_nodes} {num_edges} {0}\n"]
    
    for node in range(num_nodes):
        neighbors = " ".join(str(neighbor + 1) for neighbor in graph.neighbors(node))
        metis_lines.append(neighbors + "\n")
    
    return metis_lines

def get_connected_components(ad_mat : lil_matrix): 
	graph = ad_mat.tocsr()
    
	# Find the connected components using scipy's connected_components function
	num_components, labels = connected_components(csgraph=graph, directed=False)

	# Initialize an empty dictionary to store the components
	components = {}

	# Iterate over the nodes and assign them to their respective components
	for node, component in enumerate(labels):
		if component not in components:
			components[component] = [node]
		else:
			components[component].append(node)

	# Convert the dictionary of components into a list of lists
	connected_components_list = list(components.values())

	return connected_components_list

def solve_max_independent_set_integer(adj_mat):
	n = adj_mat.shape[0]
	if n == 1:
		return 1, np.array([0])
	prog = MathematicalProgram()
	v = prog.NewBinaryVariables(n)
	prog.AddLinearCost(-np.sum(v))
	for i in range(0,n):
		for j in range(i,n):
			if adj_mat[i,j]:
				prog.AddLinearConstraint(v[i] + v[j] <= 1)

	solver_options = SolverOptions()
	solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)

	result = Solve(prog, solver_options=solver_options)
	return -result.get_optimal_cost(), np.nonzero(result.GetSolution(v))[0]

def solve_max_independet_set_KAMIS(adj_mat, maxtime=20):
    if not isinstance(adj_mat, np.ndarray):
        #nx behaves wierdly with sparse arrays
        adj_mat = adj_mat.toarray()
    nx_graph = nx.Graph(adj_mat)
    metis_lines = networkx_to_metis_format(nx_graph)
    with open("tmp/vgraph_red.metis", "w") as f:
        f.writelines(metis_lines)
        f.flush()  # Flush the buffer to ensure data is written immediately
        f.close()
    binary_loc = f"{_KAMISPATH}/deploy/redumis "
    options = f"--time_limit={maxtime} --seed=5 --output=tmp/stable_set.txt "
    file = "tmp/vgraph_red.metis"
    command = binary_loc + options + file
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    print(str(str(output)[2:-1]).replace('\\n', '\n ').replace('\t', ' '))
    with open("tmp/stable_set.txt", "r") as f:
        stable_set_idx = f.readlines()
    stable_set = np.nonzero([int(i) for i in stable_set_idx])[0]
    return len(stable_set), stable_set


if __name__ == "__main__":
    # Code to be executed when the script is run directly

	adj_matrix = lil_matrix((6, 6))
	adj_matrix[0, 1] = 1
	adj_matrix[1, 0] = 1
	adj_matrix[1, 2] = 1
	adj_matrix[2, 1] = 1
	adj_matrix[3, 4] = 1
	adj_matrix[4, 3] = 1
	adj_matrix[4, 5] = 1
	adj_matrix[5, 4] = 1

	a = get_connected_components(adj_matrix)
	cost, mis = solve_max_independent_set_integer(adj_matrix)

	print('done')