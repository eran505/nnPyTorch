from mpl_toolkits import mplot3d

#matplotlib inline
from CSV.get_info import  get_info_path_gen
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from sys import exit
from random import shuffle
from os.path import expanduser
import os_util as pt
from visualization.BeliefTree import make_belief_G
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from queue import Queue
from visualization.TreePlan import make_graph
from os import path
from os import system

class wPoint(object):
    def __init__(self, grid=(550.200,5), x_p=0.2, y=11 ,z_p=0.75):
        self._grid=grid
        self.x=x_p
        self.y=y
        self.z=z_p

    def get_list_(self,y_append=0):
        ans=[]
        if self.y==1:
            return [(self._grid[0]*self.x,int(self._grid[1]*0.5),self._grid[-1]*self.z)]
        yi = np.linspace(0, self._grid[1], num=self.y+2)
        for i in yi[1:-1]:
            ans.append((self._grid[0]*self.x,i+y_append,self._grid[-1]*self.z))
        return ans

class Node(object):
    def __init__(self,pos=None,speed=None,time=-1):
        self.children=[]
        self.plans=[]
        self.prob=[]
        self.pos=pos
        self.speed = speed
        self.t = time

    def get_id(self):
        return (self.pos,self.t,len(self.plans))
    def is_unique_plan(self):
        return len(self.plans)==1
class PlanRec(object):

    def __init__(self):
        self.root_dummy=Node()
        self.cur=None

    def add_path(self,path,idx_plan):
        time_t = 0
        p = path['p']
        trajectory = path['traj']
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
            self.cur=node
            time_t+=1

    def search_node(self,children,pos,time_t):
        for child in children:
            if (child.pos == pos and child.t == time_t):
                return child
        return None

    def make_tree(self,paths):
        for idx,path_i in enumerate(paths):
            self.cur = self.root_dummy
            self.add_path(path_i,idx)


def make_tree():
    path_to_file = "{}/{}".format(expanduser("~"), "/car_model/debug/p.csv")
    results = hlp.load__p(path_to_file)
    PR = PlanRec()
    PR.make_tree(results)
    d = tree_to_dict(PR.root_dummy.children[0])
    return d,PR.root_dummy.children[0]

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


def make_dict_from_csv():
    path_to_file = "{}/{}".format(expanduser("~"),"/car_model/debug/p.csv")
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
#
# def make_graph():
#     # res = make_dict_from_csv()
#     res,root = make_tree()
#     # print(results)
#     G = nx.from_dict_of_lists(res)
#     d_color = {0:'green',1:"blue",2:"red",3:"black",4:"yellow",6:"pink",-1:"orange"}
#     color_map = []
#
#     for node in G:
#         x = []
#         if node in res:
#             x = res[node]
#         color = d_color[len(x)]
#         num_plans = node[-1]
#         if num_plans==1:
#             color=d_color[-1]
#             if len(x)==0:
#                 color=d_color[0]
#         color_map.append(color)
#
#     # G = nx.from_dict_of_lists(dol)
#     pos = nx.spring_layout(G, k=0.3 * 1 / np.sqrt(len(G.nodes())), iterations=20)
#     plt.figure(3, figsize=(30, 30))
#     nx.draw(G, pos=pos)
#     # nx.draw_networkx_labels(G, pos=pos)
#     # plt.show()
#     pos = graphviz_layout(G, prog="dot",root=root.get_id())  # ‘dot’, ‘twopi’, ‘fdp’, ‘sfdp’, ‘circo’
#     nx.draw(G, pos,node_color=color_map)
#     plt.savefig("{}/car_model/debug/tree.png".format(expanduser("~")))
#     plt.show()



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def path_score(path_to_csv_file):
    l = hlp.load__p(path_to_csv_file)
    l_all = []
    acc=0
    for item in  l:
        l_all.append(np.array(item['traj']))
        acc+=float(item['p'])
        if acc>=1:
            break
    d={}
    for i,list_f1 in enumerate(l_all):
        for item in list_f1:
            item = item.flatten()
            item = tuple(item)
            if item in d:
                d[item]+=1
            else:

                d[item]=1
    ones = 0
    big_one=0
    for item in d.values():
        if item == 1:
            ones+=1
        else:
            big_one+=1
    #print("uniq - {}:{} -> {}".format(ones,big_one,ones/(big_one+ones)))

def main_f():
    cmap = get_cmap(15)

    p='car_model/debug'
    res = pt.walk_rec("{}/{}".format(expanduser("~"),p),[],"p.csv")
    res = [x for x in res if str(x).split('/')[-1].__contains__("map") is False]
    shuffle(res)
    index = 0
    path_score(res[index])
    # for i,x in enumerate(res):
    #     if str(res).split('/')[-1] == '31158_p.csv':
    #         index=i
    #         break
    print(res[index])
    l = hlp.load__p(res[index])

    print(l)
    matrix = []
    for item in l:
        size = len(item["traj"])
        path = np.zeros([size, 3])
        for idx, pos in enumerate(item["traj"]):
            path[idx] = pos[0]
        matrix.append(path)

    # exit(0)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    color = ['red', 'green', 'gold', 'black', 'lime', 'violet', 'plum',
             'orange', 'navy', 'salmon']
    for i, p in enumerate(matrix):
        zline = np.array(p[:, 2])
        xline = np.array(p[:, 1])
        yline = np.array(p[:, 0])
        ax.plot3D(xline, yline, zline, color[i % len(color)])




    ax.view_init(azim=0, elev=270)

    plt.savefig("{}/car_model/debug/paths.png".format(expanduser("~")))
    plt.show()
    exit()
def cp_p_file(dirx="/car_model/debug"):
    dirx = "{}{}".format(expanduser("~"),dirx)
    f1 ="{}/p_1.csv".format(dirx)
    f = "{}/p.csv".format(dirx)
    if path.isfile(f1):
        system("cp {} {}".format(f1,f))

def comper_paths():
    p='car_model/debug'
    res = pt.walk_rec("{}/{}".format(expanduser("~"),p),[],"p.csv")
    res = [x for x in res if str(x).split('/')[-1].__contains__("map") is False]
    shuffle(res)
    index = 0
    l = hlp.load__p(res[index])
    for j,item in enumerate(l):
        x = item['traj']
        for i in range(j+1,len(l)):
            h = int(len(l[i]['traj'])*0.95)
            if x[:h] == l[i]['traj'][:h]:
                print("{} == {}".format(j,i))



def crate_map(p="/home/eranhe/car_model/exp/new/full/p.csv"):
    l = hlp.load__p(p)

    print(l)
    matrix = []
    for item in l:
        size = len(item["traj"])
        path = np.zeros([size, 3])
        for idx, pos in enumerate(item["traj"]):
            path[idx] = pos[0]
        matrix.append(path)

    grid=(550,200,5)
    w_points = [0.1*grid[1],0.2*grid[1],0.9*grid[1]]
    # Data for a three-dimensional line
    color = ['red', 'green', 'gold', 'black', 'lime', 'violet', 'plum',
             'orange', 'navy', 'salmon']
    for i, p in enumerate(matrix):
        xline = np.array(p[0:1, 1])
        yline = np.array(p[0:1, 0])
        plt.plot(yline,xline,marker='$S_e$',c='red',markersize=15,label='Evader')
        break
    for i, p in enumerate(matrix):
        xline = np.array(p[-1:, 1])
        yline = np.array(p[-1:, 0])
        if i==0:
            plt.plot(yline,xline,marker='*',c="green",markersize=10,label='Goal')
        else:
            plt.plot(yline, xline, marker='*', c="green", markersize=10)

    w1 = wPoint(grid=(550,60,5),x_p=0.1,y=3)
    w2 = wPoint(grid=(550, 60, 5), x_p=0.2, y=3)
    w3 = wPoint(grid=(550, 200, 5), x_p=0.3, y=1)
    w4 = wPoint(grid=(550, 200, 5), x_p=0.5, y=1)
    l=[(w1,70),(w2,70),(w3,0),(w4,0)]
    for idx,tuple_object in enumerate(l):
        obj,y_append =  tuple_object
        list_ppints = obj.get_list_(y_append)
        c_i = color[idx % len(color)]
        for i,item in enumerate(list_ppints) :
            print(item)
            if idx==0 and i==0:
                plt.plot(item[0],item[1],marker='x',c='black',label='Way-point')
            else:
                plt.plot(item[0], item[1], marker='x', c='black')

    sp=[(450,100),(470,100),(490,100),(510,100),(530,100),(550,100)]
    for i,item in enumerate(sp):
        if i==0:
            plt.plot(item[0],item[1],marker='$s_p$',c='blue',markersize=10,label='Pursuer')
        else:
            plt.plot(item[0], item[1], marker='$s_p$', c='blue', markersize=10)
    plt.legend()
    plt.savefig("{}/car_model/debug/waypoint.png".format(expanduser("~")))
    plt.show()
    exit()

if __name__ == "__main__":
    #crate_map()
    comper_paths()
    get_info_path_gen()
    cp_p_file()
    main_f()
    make_graph("/car_model/debug")
    make_belief_G(dir_p="/car_model/debug")


