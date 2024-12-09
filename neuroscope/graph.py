# graph.py
#   neuroscope graph structure
# by: Noah Syrkis

#imports
import jax.numpy as jnp
import jraph

nodes_features = jnp.array([[0.], [1.], [2.]])
senders        = jnp.array([0, 1, 2])
receivers      = jnp.array([1, 2, 0])
edges          = jnp.array([[0.], [1.], [2.]])

n_node         = jnp.array([len(nodes_features)])
n_edge         = jnp.array([len(edges)])

global_context = jnp.array([[0.]])
graph          = jraph.GraphsTuple(nodes=nodes_features, edges=edges,
                                   senders=senders, receivers=receivers,
                                   n_node=n_node, n_edge=n_edge,
                                   globals=global_context)

graphs = jraph.batch([graph, graph])

node_targets = jnp.array([[True], [False], [True]])
graph = graph._replace(nodes={'inputs': graph.nodes, 'targets': node_targets})