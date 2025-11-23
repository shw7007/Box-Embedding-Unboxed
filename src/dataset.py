import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def dict_to_triples(data:dict, on_grandparent, relation="IsA"):
    # 1. Use reverse mapping (child->parent) 
    # needed to trace back the tree
    child_to_parent = {}
    all_entities = set()

    for parent, children in data.items():
        all_entities.add(parent)
        for child in children:
            all_entities.add(child)
            child_to_parent[child] = parent  # who is parent of child?

    triples = []

    # Search lineage tree for every entitiy 
    for entity in all_entities:
        current_node = entity
        
        # Check whether "making grandparent-grandchild relation" is on
        if on_grandparent is True:
            # Search upward until there is no parent(until find root)
            while current_node in child_to_parent:
                parent = child_to_parent[current_node]

                # add (Me, IsA, ancestor) triple 
                # entity is fixed to Me, parent keep go upward(father -> grandfather...)
                triples.append((entity, 'IsA', parent))

                # move a step upward
                current_node = parent
        elif current_node in child_to_parent:
            # If "making granparent-grandchild relation" is off, make only parent-child triple
            triples.append((entity, 'IsA', child_to_parent[entity]))

    # Eliminate duplication and sort
    triples = sorted(list(set(triples)))

    return triples

def triples_to_list(data):
    # using triple make list and dict
    
    # entities = [(child1, parent1),....]
    entities = sorted(list(set([t[0] for t in data] + [t[2] for t in data])))
    
    # entity2id = {child1 : 1, child2 : 2,....}
    entity2id = {e:i for i, e in enumerate(entities)}
    return entities, entity2id

# using triple, draw the whole graph
# Use only in ipynb notebook not in console
"""
def draw_by_triple(triples, name="Test dataset"):
    plt.figure(figsize=(8, 6))
    G = nx.DiGraph()
    G.add_edges_from([(t[0], t[2]) for t in triples])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True)
    plt.title("Graph of {}".format(name))
    plt.show()
"""

# Using triple, find each entities level in the dataset tree -> used for coloring boxes
def get_entity_levels(triples):
    
    # dict initialize
    levels = dict()
    
    # iterate every triples and find child level first
    for child, _, parent in triples:
        if(child in levels.keys()):
            levels[child] += 1
        else:
            levels[child] = 1
            
    # after child, find every root node
    for _,__,parent in triples:
        if(parent not in levels.keys()):
            levels[parent] = 0
    return levels

class data_dealer:
    def __init__(self, data:dict, on_grandparent=1):
        self.data = data

        self.triples = dict_to_triples(data, on_grandparent)

        self.entities, self.entity2id = \
        triples_to_list(self.triples)
        
        self.level_dict = get_entity_levels(self.triples)
        
        #do not use in console environment
        #self.draw = draw_by_triple(self.triples)
