from mpl_toolkits import mplot3d
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from sys import exit
from random import shuffle
from os.path import expanduser
import os_util as pt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from queue import Queue
from os_util import walk_rec
import pandas as pd
import os


from visualization.TreePlan import get_the_min_time
def make_tree(dir_p):
    path_to_file = "{}/{}/{}".format(expanduser("~"),dir_p, "p.csv")
    results = hlp.load__p(path_to_file)
    PR = BTree(dir_p,results)
    d = tree_to_dict(PR.root_dummy.children[0])
    return d,PR.root_dummy.children[0],PR.dict_node()

def make_belief_G(dir_p):
    # res = make_dict_from_csv()
    res,root,dict_node = make_tree(dir_p)
    # print(results)
    G = nx.from_dict_of_lists(res)
    d_color = {0:'green',1:"turquoise",2:"red",3:"black",4:"yellow",5:"pink",-1:"silver"}
    color_map = []
    # min_d = get_the_min_time()
    for node in G:
        x = []
        if node in res:
            x = res[node]
        color = d_color[len(x)]
        num_plans = node[-1]
        if num_plans==1:
            color=d_color[-1]
            if len(x)==0:
                color=d_color[0]
        # if dict_node[node].min_t is True:
        #     color = "blue"
        color_map.append(color)

    # G = nx.from_dict_of_lists(dol)
    pos = nx.spring_layout(G, k=0.3 * 1 / np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(30, 30))
    nx.draw(G, pos=pos)
    # nx.draw_networkx_labels(G, pos=pos)
    # plt.show()
    pos = graphviz_layout(G, prog="dot",root=root.get_id())  # ‘dot’, ‘twopi’, ‘fdp’, ‘sfdp’, ‘circo’
    nx.draw(G, pos,node_color=color_map)
    plt.savefig("{}{}/belief.png".format(expanduser("~"),dir_p))
    plt.show()


def tree_to_dict(root):
    q = Queue()
    q.put(root)
    d={}
    while q.empty() is False:
        node = q.get()
        node_ky = node.get_id()
        for child in node.children:
            if node_ky not in d:
                d[node_ky]=[child.get_id()]
            else:
                d[node_ky].append(child.get_id())
            q.put(child)
    for ky in d:
        d[ky]=list(set(d[ky]))
    return d

class NodeB(object):
    def __init__(self,pos=None,speed=None,time=-1):
        self.children=[]
        self.plans=[]
        self.prob=[]
        self.pos=pos
        self.speed = speed
        self.t = time
        self.min_t = False
    def get_id(self):
        return (self.pos,self.t,tuple(self.plans))

    def is_unique_plan(self):
        return len(self.plans)==1
    def search_node(self,loc):
        for child in self.children:
            if child.pos == loc:
                return child
        return None
    def add_belief(self,b):
        if b in self.plans:
            assert False
        self.plans.append(b)
    def add_child(self,loc,time,path_idx):
        n = NodeB(loc,None,time)
        n.add_belief(path_idx)
        self.children.append(n)
        return n
class BTree(object):

    def __init__(self,dir_path,results):
        self.root_dummy=NodeB()
        self.dict_nodes={}
        self.d_min = get_the_min_time(dir_path)
        self.CONST=2000
        self.make_tree(results)
    def dict_node(self):
        return self.dict_nodes
    def get_root(self):
        return self.root_dummy
    def get_father(self,t,path_id):
        if t==-1:
            return self.root_dummy
        ky = self.node_hash(t,path_id)
        if ky not in self.dict_nodes:
            assert False
        return self.dict_nodes[ky]

    def node_hash(self,time_ti,path_id):
        return time_ti+path_id*self.CONST


    def make_tree(self,paths):
        time_t=0
        done_ctr=0
        array_of_path_idx=[x for x in range(len(paths))]
        to_del=[]
        while True:
            if done_ctr==len(paths):
                break
            for i in array_of_path_idx:
                if time_t==len(paths[i]['traj']):
                    done_ctr+=1
                    to_del.append(i)
                    continue

                father_node = self.get_father(time_t-1,i)
                node = father_node.search_node(paths[i]['traj'][time_t][0])
                if node == None:
                    node = father_node.add_child(paths[i]['traj'][time_t][0],time_t,i)
                    # print("add: {} {} - {}".format(time_t, i, len(paths[i]['traj'])))
                    self.dict_nodes[self.node_hash(time_t,i)]=node
                else:

                    node.add_belief(i)
                    # print("add: {} {} - {}".format(time_t,i,len(paths[i]['traj'])))
                    self.dict_nodes[self.node_hash(time_t,i)]=node
            time_t+=1

            if len(to_del)>0:
                for x in to_del:
                    array_of_path_idx.remove(x)
                to_del=[]


if __name__ == '__main__':
    make_belief_G(dir_p="/car_model/debug")