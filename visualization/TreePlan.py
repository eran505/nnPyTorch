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



class Node(object):
    def __init__(self,pos=None,speed=None,time=-1):
        self.children=[]
        self.plans=[]
        self.prob=[]
        self.pos=pos
        self.speed = speed
        self.t = time
        self.min_t = False
    def get_id(self):
        return (self.pos,self.t,len(self.plans))
    def is_unique_plan(self):
        return len(self.plans)==1

class PlanRec(object):

    def __init__(self,dir_p):
        self.root_dummy=Node()
        self.cur=None
        self.d_min = get_the_min_time(dir_p)

    def dict_node(self):
        q = Queue()
        q.put(self.root_dummy.children[0])
        d = {}
        while q.empty() is False:
            node = q.get()
            node_ky = node.get_id()
            d[node_ky]=node
            for child in node.children:
                q.put(child)
        return d

    def add_path(self,path,idx_plan):
        time_t = 0
        p = path['p']
        trajectory = path['traj']
        min_time = self.d_min[idx_plan]
        for state in trajectory:
            pos = state[0]
            speed = state[1]
            node = self.search_node(self.cur.children,pos,time_t)
            if node is None:
                node = Node(pos, speed, time_t)
                node.plans.append(idx_plan)
                node.prob.append(p)
                self.cur.children.append(node)
            else:
                node.plans.append(idx_plan)
                node.prob.append(p)
            if min_time == time_t:
                node.min_t=True
            self.cur=node
            time_t+=1

        assert(time_t>min_time)

    def search_node(self,children,pos,time_t):
        for child in children:
            if (child.pos == pos and child.t == time_t):
                return child
        return None

    def make_tree(self,paths):
        for idx,path_i in enumerate(paths):
            self.cur = self.root_dummy
            self.add_path(path_i,idx)


def make_tree(dir_p):
    path_to_file = "{}/{}/{}".format(expanduser("~"),dir_p, "p.csv")
    results = hlp.load__p(path_to_file)
    PR = PlanRec(dir_p)
    PR.make_tree(results)
    d = tree_to_dict(PR.root_dummy.children[0])
    return d,PR.root_dummy.children[0],PR.dict_node()

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


def make_dict_from_csv(dir_p):
    path_to_file = "{}/{}/{}".format(expanduser("~"),dir_p,"/p.csv")
    results = hlp.load__p(path_to_file)
    d={}
    for item in results:
        print(item)
        path = item["traj"]
        w = item["p"]
        for idx in range(len(path)-1):
            ky = (path[idx][0],idx)
            v = (path[idx+1][0],idx+1)
            if ky not in d:
                d[ky]= [v]
            else:
                d[ky].append(v)
    print(d)
    for item in d.keys():
        d[item]=list(set(d[item]))
    return d

def make_graph(dir_p):
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
        if dict_node[node].min_t is True:
            color = "blue"
        color_map.append(color)

    # G = nx.from_dict_of_lists(dol)
    pos = nx.spring_layout(G, k=0.3 * 1 / np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(30, 30))
    nx.draw(G, pos=pos)
    # nx.draw_networkx_labels(G, pos=pos)
    # plt.show()
    pos = graphviz_layout(G, prog="dot",root=root.get_id())  # ‘dot’, ‘twopi’, ‘fdp’, ‘sfdp’, ‘circo’
    nx.draw(G, pos,node_color=color_map)
    plt.savefig("{}{}/tree.png".format(expanduser("~"),dir_p))
    plt.show()


def make_belief_tree(dir_p):
    res,root,dict_node = make_tree(dir_p)


def get_df_con(dir_p):
    path_to_dir = "{}/{}".format(expanduser("~"), dir_p)

    res = walk_rec(path_to_dir, [], ".csv")
    for item in res:
        if str(item).split(os.sep)[-1].split('.')[0].__contains__("con"):
            return pd.read_csv(item)
    return None

def pos_diff_max(pos_1,pos_2):
    return max(abs(pos_1[0]-pos_2[0]),abs(pos_1[1]-pos_2[1]),abs(pos_1[-1]-pos_2[-1]))

def get_the_min_time(dir_p):
    df_con = get_df_con(dir_p)
    s_p = tuple(list(map(lambda x : int(x),str(df_con['D_start'].values[0]).split('|'))))
    results = hlp.load__p("{}/{}/{}".format(expanduser("~"),dir_p,"/p.csv"))
    d={}
    for idx,path_i in enumerate(results) :
        traj = path_i['traj']
        dif = pos_diff_max(s_p,traj[-2][0])
        #print(dif)
        d[idx]=len(traj)-dif
    return d

if __name__ == "__main__":

    make_graph(dir_p="/car_model/debug")