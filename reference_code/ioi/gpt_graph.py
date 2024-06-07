from pathlib import Path
from matplotlib import pyplot as plt
import yaml

import torch
import numpy as np
from graphviz import Digraph


def no_revisit(func):
    def wrapper(self, *args, **kwargs):
        if not self.visited:
            func(self, *args, **kwargs)
            self.visited = True
    return wrapper

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []

        self.visited = False

    def __call__(self):
        if self.name == 'input':
            return ('input',)
        elif self.name == 'output':
            return ('output',)
        elif self.name.startswith('attn'):
            t, i, j = self.name.split('_')
            i, j = int(i), int(j)
            return t, i, j
        elif self.name.startswith('mlp'):
            t, i = self.name.split('_')
            i = int(i)
            return t, i


    @property
    def color(self):
        if self.name == 'input':
            return 'blue'
        elif self.name == 'output':
            return 'blue'
        elif self.name.startswith('attn'):
            return 'red'
        elif self.name.startswith('mlp'):
            return 'green'
        else:
            raise ValueError

    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)

    def __repr__(self):
        return f'[{self.name}]'



    # @no_revisit
    def dfs_connections(self):
        conns = set()
        for child in self.children:
            conns.add((self.name, child.name))
            conns = conns.union(child.dfs_connections())
        return conns

    # @no_revisit
    def dfs_connections_r(self):
        conns = set()
        for parent in self.parents:
            conns.add((parent.name, self.name))
            conns = conns.union(parent.dfs_connections_r())
        return conns



class GPTGraph:
    def __init__(self):
        nodes = {
            'input': Node('input'),
            'output': Node('output'),
        }
        for i in range(12):
            for j in range(12):
                nodes[f'attn_{i}_{j}_Q'] = Node(f'attn_{i}_{j}_Q')
                nodes[f'attn_{i}_{j}_K'] = Node(f'attn_{i}_{j}_K')
                nodes[f'attn_{i}_{j}_V'] = Node(f'attn_{i}_{j}_V')
                nodes[f'attn_{i}_{j}_O'] = Node(f'attn_{i}_{j}_O')
            nodes[f'mlp_{i}'] = Node(f'mlp_{i}')
        self.nodes = nodes
        self.input_node = self.nodes['input']
        self.output_node = self.nodes['output']
        self.attn = lambda i,j,M: self.nodes[f'attn_{i}_{j}_{M}']
        self.mlp = lambda i: self.nodes[f'mlp_{i}']

    def reset_graph(self):
        for node in self.nodes.values():
            node.visited = False

    def set_connections(self, masks):
        output_mask, attn_Q_masks, attn_K_masks, attn_V_masks, edge_mask_mlp = masks

        for i in range(12):
            output_mask_i = output_mask[1:][i*13:(1+i)*13]
            mlp_i_output_edge = output_mask_i[-1]
            if mlp_i_output_edge == 1:
                self.output_node.add_child(self.mlp(i))

            for j in range(12):
                attn_ij_output_edge = output_mask_i[j]
                if attn_ij_output_edge == 1:
                    self.output_node.add_child(self.attn(i,j,'O'))

        for i in range(12):
            edge_mask_mlps_i = edge_mask_mlp[i]
            if edge_mask_mlps_i[0] == 1:
                self.mlp(i).add_child(self.input_node)
            for j in range(i+1):
                edge_mask_mlps_ij = edge_mask_mlps_i[1:][j*13:(1+j)*13]
                if j < i:
                    edge_mask_mlps_i_mlp_j = edge_mask_mlps_ij[-1]
                    if edge_mask_mlps_i_mlp_j == 1:
                        self.mlp(i).add_child(self.mlp(j))
                for k in range(12):
                    edge_mask_mlps_i_attn_jk = edge_mask_mlps_ij[k]
                    if edge_mask_mlps_i_attn_jk == 1:
                        self.mlp(i).add_child(self.attn(j,k,'O'))

        for i in range(12):
            for j in range(12):
                edge_mask_attn_ij_Q = attn_Q_masks[i][:, j]
                edge_mask_attn_ij_K = attn_K_masks[i][:, j]
                edge_mask_attn_ij_V = attn_V_masks[i][:, j]
                
                if edge_mask_attn_ij_Q[0] == 1:
                    self.attn(i,j,'Q').add_child(self.input_node)
                if edge_mask_attn_ij_K[0] == 1:
                    self.attn(i,j,'K').add_child(self.input_node)
                if edge_mask_attn_ij_V[0] == 1:
                    self.attn(i,j,'V').add_child(self.input_node)
                    
                for k in range(i):
                    edge_mask_attn_ij_k_Q = edge_mask_attn_ij_Q[1:][k*13:(1+k)*13]
                    edge_mask_attn_ij_k_K = edge_mask_attn_ij_K[1:][k*13:(1+k)*13]
                    edge_mask_attn_ij_k_V = edge_mask_attn_ij_V[1:][k*13:(1+k)*13]

                    if edge_mask_attn_ij_k_Q[-1] == 1:
                        self.attn(i,j,'Q').add_child(self.mlp(k))
                    if edge_mask_attn_ij_k_K[-1] == 1:
                        self.attn(i,j,'K').add_child(self.mlp(k))
                    if edge_mask_attn_ij_k_V[-1] == 1:
                        self.attn(i,j,'V').add_child(self.mlp(k))
                        
                    for l in range(12):
                        if edge_mask_attn_ij_k_Q[l] == 1:
                            self.attn(i,j,'Q').add_child(self.attn(k,l,'O'))
                        if edge_mask_attn_ij_k_K[l] == 1:
                            self.attn(i,j,'K').add_child(self.attn(k,l,'O'))
                        if edge_mask_attn_ij_k_V[l] == 1:
                            self.attn(i,j,'V').add_child(self.attn(k,l,'O'))

    
    def render_plot(self, fig_name, mode='both'):
        
        edges, nodes = self.get_connections(mode)
        graph = Digraph()
        
        for node in nodes:
            graph.node(node.name, color=node.color)
            
        graph.edges(edges)
        
        graph.render(fig_name)


    def get_connections(self, mode='both'):

        if mode == 'dfs':
            conns = self.output_node.dfs_connections()
        elif mode == 'dfs_r':
            conns = self.input_node.dfs_connections_r()
        elif mode == 'both':
            conns = self.output_node.dfs_connections()
            conns_r = self.input_node.dfs_connections_r()
            conns = conns.intersection(conns_r)
        elif mode == 'all':
            conns = set()
            for node in self.nodes.values():
                for child in node.children:
                    conns.add((node.name, child.name))
        nodes = []
        for e in conns:
            nodes.append(self.nodes[e[0].name])
            nodes.append(self.nodes[e[1].name])
        return conns, set(nodes)

