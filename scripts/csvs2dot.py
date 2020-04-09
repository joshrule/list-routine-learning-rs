import pandas as pd
import pygraphviz as pgv

edges = pd.read_csv("edges.csv", header=0)
nodes = pd.read_csv("nodes.csv", header=0)

g = pgv.AGraph(directed=True)
for i, row in nodes.iterrows():
    if row['type'] == "visited":
        g.add_node(row['name'])
    # else:
    #     g.add_node(row['name'], shape="point", weight=0.2)
g.add_edges_from([(row['source'], row['target'])
                  for i, row in edges.iterrows()
                  if row['target'][0] != "u"])

g.draw("tree.pdf", prog="dot", args="-Goverlap=prism -Gmodel=subset")

{}
