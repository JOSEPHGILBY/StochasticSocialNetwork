from enum import Enum
import random
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import csv
import operator
import numpy as np
import pickle
import scipy.optimize as optimize


class StepStatistics(object):

    def __init__(self):
        #row , col is transmission from , to 
        self.transmissions_counts = [[0,0,0,0,0],
                                     [0,0,0,0,0],
                                     [0,0,0,0,0],
                                     [0,0,0,0,0],
                                     [0,0,0,0,0]
                                    ]

        #row, col is transition from, to
        self.transitions_counts = [[0,0,0,0,0],
                                   [0,0,0,0,0],
                                   [0,0,0,0,0],
                                   [0,0,0,0,0],
                                   [0,0,0,0,0]
                                  ]
        self.comb_i = list(itertools.combinations_with_replacement({0,1,2,3,4},2))                         
        self.edge_counts = [0] * len(self.comb_i)
        
        

class Opinion(object):
    
    class opinionType(Enum):
        strong_A, lean_A, neutral, lean_B, strong_B = 0,1,2,3,4
    
    def __init__(self, opinion_val, v ):
        self.opinion_val = opinion_val
        self.friend_add_receptiveness = 0.00
        self.strength = 0.5 #strength of argument
        self.blash = v[0]
        self.blashbonus = v[1]
        
        r = 0.8 #resistivity
        rb = v[2] #resistivity bonus
        e = v[3] #receptiveness
        b = self.blash #backlash
        bb = self.blashbonus #backlash bonus
        
        f = 0.05
        fmult = 3
        self.transmission_vec = [f*fmult,f,0.0,f,f*fmult]
        
        self_m = min(1.0, r + 2*rb) #base probability of staying the same opinion
        m = 1 - self_m #base probability of moving to the 'right'
        lblash = min(1,b + 2*bb)
        self.strong_change_opinion_mat = [[self_m + m*(1-b)  ,m*b          ,0.00,0.00,0.00],
                                          [self_m            ,m            ,0.00,0.00,0.00],
                                          [1.00              ,0.00         ,0.00,0.00,0.00],
                                          [self_m            ,m            ,0.00,0.00,0.00],
                                          [self_m + m*lblash ,m*(1-lblash) ,0.00,0.00,0.00]
                                         ]
        #print self.strong_change_opinion_mat    
        
        self_m = min(1.0, r + 1*rb)
        m = 1 - self_m
        lblash = min(1,b + bb)
        self.lean_change_opinion_mat = [[m*e*(1-b) ,self_m + m*(1-e)*(1-b)      ,m*b            ,0.00,0.00],
                                        [m*e       ,self_m + m*(1-e)            ,0.00           ,0.00,0.00],
                                        [0.00      ,1.0                         ,0.00           ,0.00,0.00],
                                        [m*(1-e)   ,self_m                      ,m*e            ,0.00,0.00],
                                        [m*lblash  ,self_m + m*(1-lblash)*(1-e) ,m*e*(1-lblash) ,0.00,0.00]
                                       ]
        #print self.lean_change_opinion_mat
        
        self_m = r
        m = 1 - self_m
        self.neutral_change_opinion_mat = [[0.00, m*e*(1-b) ,self_m + m*(1-e)*(1-b) ,m*b       ,0.00],
                                           [0.00, m*e       ,self_m + m*(1-e)       ,0.00      ,0.00],
                                           [0.00, 0.00      ,1.00                   ,0.00      ,0.00],
                                           [0.00, 0.00      ,self_m + m*(1-e)       ,m*e       ,0.00],
                                           [0.00, m*b       ,self_m + m*(1-e)*(1-b) ,m*e*(1-b) ,0.00]
                                          ]
        #print self.neutral_change_opinion_mat
        
        self.delta_opinion_change_count = 0
    
    def transmit_decide(self):
        return sample_event_with_probability(self.transmission_vec[self.opinion_val.value])
        
    def add_friend_decide(self):
        return sample_event_with_probability(self.friend_add_receptiveness)
        
    def add_one_delta_opinion_change(self, sender_opinion_val):
        remove_friend = False
        
        if self.opinion_val == Opinion.opinionType.strong_A:
            val = sample_from_vec_probs(self.strong_change_opinion_mat[sender_opinion_val.value])
        elif self.opinion_val == Opinion.opinionType.lean_A:
            val = sample_from_vec_probs(self.lean_change_opinion_mat[sender_opinion_val.value])
        elif self.opinion_val == Opinion.opinionType.neutral:
            val = sample_from_vec_probs(self.neutral_change_opinion_mat[sender_opinion_val.value])
        elif self.opinion_val == Opinion.opinionType.lean_B:
            val = abs(4 - sample_from_vec_probs(self.lean_change_opinion_mat[abs(4 - sender_opinion_val.value)]))
        else:
            val = abs(4 - sample_from_vec_probs(self.strong_change_opinion_mat[abs(4 - sender_opinion_val.value) ] ))
        
        #if (val - self.opinion_val.value)*(sender_opinion_val.value - 2) < 0 and sender_opinion_val.value in [0,4]:
        #    if (self.opinion_val.value - 2)*(sender_opinion_val.value - 2) < 0 and sample_event_with_probability(min(1, self.blash + self.blashbonus)):
        #        remove_friend = True
        #    elif sample_event_with_probability(self.blash):
        #        remove_friend = True
        self.delta_opinion_change_count += val - self.opinion_val.value
        return remove_friend
        
    def change_opinion(self):
        t = self.opinion_val.value + self.delta_opinion_change_count
        self.opinion_val = self.opinionType(max(0 , min(t, 4)))
        self.delta_opinion_change_count = 0
        return self.opinion_val.value
        

class Person(object):
    
    def __init__(self, out_conn_list=[], opinion=Opinion(Opinion.opinionType.neutral, [0.0,0.0,0.0,0.0,0.0,0.0]), index = 0):
        self.outward_connection_list = out_conn_list
        self.opinion = opinion
        self.doubly_connected = False
        self.index = index
        
    def determine_and_add_friend(self):
        if self.opinion.add_friend_decide():
            new_friends_to_make = [fof for connection in self.outward_connection_list for fof in connection.outward_connection_list if (fof not in self.outward_connection_list and fof != self) ]
            if len(new_friends_to_make) > 0:
                i = random.randint(0, len(new_friends_to_make) - 1)
                self.outward_connection_list.append(new_friends_to_make[i])
                new_friends_to_make[i].outward_connection_list.append(self)

    def receive_transmission(self,sender):
        if self.opinion.add_one_delta_opinion_change(sender.opinion.opinion_val) and len(self.outward_connection_list) > 2 and len(sender.outward_connection_list) > 2:
            sender.outward_connection_list.remove(self)
            self.outward_connection_list.remove(sender)
   
    def determine_and_send_transmission(self,stats = None):
        if self.opinion.transmit_decide():
            for person in self.outward_connection_list:
                if stats:
                    stats.transmissions_counts[self.opinion.opinion_val.value][person.opinion.opinion_val.value] += 1
                person.receive_transmission(self)
                
    def send_transmission(self):
        for person in self.outward_connection_list:
            person.receive_transmission(self)
    
    def update_opinion(self, stats = None):
        old_op = self.opinion.opinion_val.value
        new_op = self.opinion.change_opinion()
        if stats and old_op != new_op:
            stats.transitions_counts[old_op][new_op] += 1
    
def sample_from_vec_probs(vec_probs):
    v = random.uniform(0,1)
    choice = 0
    sum = 0
    for x in range (0, len(vec_probs)):
        sum += vec_probs[x]
        if sum >= v:
            choice = x
            break
    return choice
    
def sample_event_with_probability(p):
    return random.uniform(0,1) <= p

def transition_proabilities_test():
    for x in range(0,5):
        for y in range(0,5):
            person_list = []
            person_list.append(Person([], Opinion(Opinion.opinionType(x)), 0))
            person_list.append(Person([person_list[0]], Opinion(Opinion.opinionType(y)), 1))
            counts = [0.0,0.0,0.0,0.0,0.0]
            for n in range (0,1000):
                person_list[1].send_transmission()
                counts[person_list[0].opinion.opinion_val.value] += 1
                person_list[0].opinion.opinion_val = Opinion.opinionType(x)
            
            counts2 = [z / 1000 for z in counts]
            print str(x) + " "  + str (y)
            print "from " + str(person_list[1].opinion.opinion_val) + " to " + str(person_list[0].opinion.opinion_val)
            print counts2
                
#models
#renyi-erdos
#watts_strogatz
#barabasi_albert_graph
#Jackson-Rodgers
#https://networkx.github.io/documentation/stable/tutorial.html
def create_graph_list(n, double=True):
    person_list = []
    for x in range(0,n-1):
        for y in range(0,x):
            pass
        person_list.append(Person())
    return person_list
    
def test_sample_vec():
    list = [0,0,0,1,0]
    counts = [0,0,0,0,0]
    for x in range (0,100000):
        i = sample_from_vec_probs(list)
        counts[i] += 1
    for y in counts:
        print float(y) / 100000

def generate_people_list_graph_from_network_graph(network_graph,v):
    person_list = []
    for z in range(0,len(list(network_graph.nodes))):
        person_list.append(Person([], Opinion(Opinion.opinionType.neutral,v), z))
    for x in list(network_graph.nodes):
        for y in list(network_graph.adj[x]):
            person_list[x].outward_connection_list.append(person_list[y])
    return person_list

def generate_network_graph_from_person_list(person_list):
    network_graph = nx.Graph()
    for x in range(0,len(person_list)):
        network_graph.add_node(x, opinion = person_list[x].opinion.opinion_val)
        
    for x in range(0,len(person_list)):  
        for y in person_list[x].outward_connection_list:
            network_graph.add_edge(x,y.index)
    return network_graph

def export_statistics(stats_l,k):
    file = open('statistics.csv', 'w')  
    writer = csv.writer(file, dialect='excel',lineterminator = '\n')
    cumul = [1,0,k-2,0,1]
    s1_v = []
    aut_v = []
    for i in range(0,len(stats_l)):
        uzip_transmissions = [x for sublist in stats_l[i].transmissions_counts for x in sublist]
        uzip_transitions = [x for sublist in stats_l[i].transitions_counts for x in sublist]
        d_cumul_neg = [(-1)*sum(sublist) for sublist in stats_l[i].transitions_counts ]
        transitions_transpose = map(list, zip(*stats_l[i].transitions_counts))
        d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
        cumul = map(operator.add, cumul, d_cumul_pos)
        cumul = map(operator.add, cumul, d_cumul_neg)
        same_color_edges = [stats_l[i].edge_counts[j] for j in [0, 1, 5, 12, 13, 14]]
        s1 = sum(same_color_edges)
        #s1_v.append(s1)
        opposite_color_edges = [stats_l[i].edge_counts[j] for j in [3,4,7,8]]
        s2 = sum(opposite_color_edges)
        neutral_color_edges = [stats_l[i].edge_counts[j] for j in [2,6,9,10,11]]
        s3 = sum(neutral_color_edges)
        writer.writerow([i] + uzip_transmissions + uzip_transmissions + stats_l[i].edge_counts + [s1] + [s2] + [s3] + cumul)
    #for k in range(1, len(stats_l)):
    #    aut_v.append(autocorrelation(k, np.var(s1_v),np.mean(s1_v), s1_v))
    #plt.plot(aut_v, 'ro')
    #plt.show()

def autocorrelation(k, sigs, m, data):
    s = 0
    for i in range(1, len(data) - k + 1):
        s += (data[i-1] - m)*(data[i + k - 1] - m)
    return s / ((len(data) - k)*sigs)
    
def n_moving_average(n, i, data):
    sub_d = data[0:i+1]
    s = 0
    if len(sub_d) < n:
        for j in range(0, len(sub_d)):
            s += sub_d[j]
        return s / len(sub_d)
    else:
        for j in range(len(sub_d) - n, len(sub_d)):
            s += sub_d[j]
        return s / n
        

def make_and_save_graph():
    k = 100
    network_graph = nx.barabasi_albert_graph(k,2)
    pickle_out = open("saved_graph.pickle","wb")
    pickle.dump(network_graph, pickle_out)
    pickle_out.close()
    
    
def optimize_leanAB_count():
    v = [0.5804325091224238, 0.02338628837325465, 0.006623553730283064, 0.5730440475304664]
    bnds = ((0,1),(0,1),(0,1),(0,1))
    result = optimize.minimize(lean_AB_run, v, method='TNC', bounds=bnds, options = {"eps" : 0.01})
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)  
 
def lean_AB_run(v):
    k = 100
    s = 0
    for l in range(0,3):
        all_step_statistics = one_run(v,k)
        
        lab_v = []
        cumul = [1,0,k-2,0,1]    
        for j in range(0,len(all_step_statistics)):
            d_cumul_neg = [(-1)*sum(sublist) for sublist in all_step_statistics[j].transitions_counts ]
            transitions_transpose = map(list, zip(*all_step_statistics[j].transitions_counts))
            d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
            cumul = map(operator.add, cumul, d_cumul_pos)
            cumul = map(operator.add, cumul, d_cumul_neg)
            lab_v.append(cumul[1] + cumul[2])
        s += -n_moving_average(400, len(lab_v) - 1, lab_v)
    print float(s)/3
    return float(s)/3
    
def check_reliability_optimized_leanAB():
    v = [0.5804325091224238, 0, 0, 0.5730440475304664]
    k = 100
    s = 0
    min = 0
    max = 0
    for l in range(0,20):
        all_step_statistics = one_run(v,k)
        
        lab_v = []
        cumul = [1,0,k-2,0,1]    
        for j in range(0,len(all_step_statistics)):
            d_cumul_neg = [(-1)*sum(sublist) for sublist in all_step_statistics[j].transitions_counts ]
            transitions_transpose = map(list, zip(*all_step_statistics[j].transitions_counts))
            d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
            cumul = map(operator.add, cumul, d_cumul_pos)
            cumul = map(operator.add, cumul, d_cumul_neg)
            lab_v.append(cumul[1] + cumul[2] + cumul[3])
        avg = n_moving_average(400, len(lab_v) - 1, lab_v)
        s += avg
        print s/(l+1)
        if l == 0:
            min = avg
        if avg < min:
            min = avg
        if avg > max:
            max = avg
    print max
    print min
    print s / 20
    
def lucky_guess():
    k = 100
    for l in range(0,100):
        v = [random.random() for x in range(0,4)]
        v[1] = random.uniform(0, (1 - v[0])* 2 / 3)
        v[2] = random.uniform(0, 0.2 * 2/3)
        all_step_statistics = one_run(v,k)
        lab_v = []
        cumul = [1,0,k-2,0,1]    
        for j in range(0,len(all_step_statistics)):
            d_cumul_neg = [(-1)*sum(sublist) for sublist in all_step_statistics[j].transitions_counts ]
            transitions_transpose = map(list, zip(*all_step_statistics[j].transitions_counts))
            d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
            cumul = map(operator.add, cumul, d_cumul_pos)
            cumul = map(operator.add, cumul, d_cumul_neg)
            lab_v.append(cumul[1] + cumul[2])
        print str(v) + " " + str(n_moving_average(400, len(lab_v) - 1, lab_v))
    
def one_run(v,k):
    all_step_statistics = []
    
    pickle_in = open("saved_graph.pickle","rb")
    network_graph = pickle.load(pickle_in)
    person_list = generate_people_list_graph_from_network_graph(network_graph,v)
    person_list[0].opinion.opinion_val = Opinion.opinionType.strong_A
    person_list[1].opinion.opinion_val = Opinion.opinionType.strong_B
    network2 = generate_network_graph_from_person_list(person_list)
    pos = nx.spring_layout(network2)
    n = 1000
    for x in range(1,n):
        stats = StepStatistics()
        for person in person_list:
            person.determine_and_add_friend()
        for person in person_list:
            person.determine_and_send_transmission(stats)
        for person in person_list:
            person.update_opinion(stats)
        network2 = generate_network_graph_from_person_list(person_list)
        for edge in network2.edges():
            p1 = person_list[edge[0]].opinion.opinion_val.value
            p2 = person_list[edge[1]].opinion.opinion_val.value
            a = min(p1,p2)
            b = max(p1,p2)
            i = stats.comb_i.index((a,b))
            stats.edge_counts[i] += 1
            
        all_step_statistics.append(stats)
    return all_step_statistics
    
def normal_run():
    all_step_statistics = []
    
    k = 100
    pickle_in = open("saved_graph.pickle","rb")
    network_graph = pickle.load(pickle_in)
    #v = [0.5804325091224238, 0, 0, 0.5730440475304664]
    v = [random.random() for x in range(0,4)]
    v[1] = random.uniform(0, (1 - v[0])* 2 / 3)
    v[2] = random.uniform(0, 0.2 * 2/3)
    print v
    person_list = generate_people_list_graph_from_network_graph(network_graph,v)
    person_list[0].opinion.opinion_val = Opinion.opinionType.strong_A
    person_list[1].opinion.opinion_val = Opinion.opinionType.strong_B
    network2 = generate_network_graph_from_person_list(person_list)
    pos = nx.spring_layout(network2)
    n = 1000
    for x in range(1,n):
        stats = StepStatistics()
        for person in person_list:
            person.determine_and_add_friend()
        for person in person_list:
            person.determine_and_send_transmission(stats)
        for person in person_list:
            person.update_opinion(stats)
        network2 = generate_network_graph_from_person_list(person_list)
        for edge in network2.edges():
            p1 = person_list[edge[0]].opinion.opinion_val.value
            p2 = person_list[edge[1]].opinion.opinion_val.value
            a = min(p1,p2)
            b = max(p1,p2)
            i = stats.comb_i.index((a,b))
            stats.edge_counts[i] += 1
            
        all_step_statistics.append(stats)
        if x == 1:
            network2 = generate_network_graph_from_person_list(person_list)
            plt.subplot(111)
            strong_A_nodes = [node for node in network2.nodes() if network2.nodes[node]['opinion'] == Opinion.opinionType.strong_A]
            lean_A_nodes = [node for node in network2.nodes() if network2.nodes[node]['opinion'] == Opinion.opinionType.lean_A]
            neutral_nodes = [node for node in network2.nodes() if network2.nodes[node]['opinion'] == Opinion.opinionType.neutral]
            lean_B_nodes = [node for node in network2.nodes() if network2.nodes[node]['opinion'] == Opinion.opinionType.lean_B]
            strong_B_nodes = [node for node in network2.nodes() if network2.nodes[node]['opinion'] == Opinion.opinionType.strong_B]
            nx.draw_networkx_nodes(network2, pos,
                               nodelist=strong_A_nodes,
                               node_color='#ff0000',
                               node_size=100,
                               alpha=0.8)
            nx.draw_networkx_nodes(network2, pos,
                               nodelist=lean_A_nodes,
                               node_color='#ff9999',
                               node_size=100,
                               alpha=0.8)
            nx.draw_networkx_nodes(network2, pos,
                               nodelist= neutral_nodes,
                               node_color='#d3d3d3',
                               node_size=100,
                               alpha=0.8)
            nx.draw_networkx_nodes(network2, pos,
                               nodelist= lean_B_nodes,
                               node_color='#00ffff',
                               node_size=100,
                               alpha=0.8)
            nx.draw_networkx_nodes(network2, pos,
                               nodelist= strong_B_nodes,
                               node_color='#0000ff',
                               node_size=100,
                               alpha=0.8)
            nx.draw_networkx_edges(network2, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(network2, pos)
            #nx.draw(network2, cmap=plt.get_cmap('jet'), node_color=values, **options)
            plt.axis('off')
            plt.show()
    
    export_statistics(all_step_statistics, k)
    lab_v = []
    cumul = [1,0,k-2,0,1]    
    for j in range(0,len(all_step_statistics)):
        d_cumul_neg = [(-1)*sum(sublist) for sublist in all_step_statistics[j].transitions_counts ]
        transitions_transpose = map(list, zip(*all_step_statistics[j].transitions_counts))
        d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
        cumul = map(operator.add, cumul, d_cumul_pos)
        cumul = map(operator.add, cumul, d_cumul_neg)
        lab_v.append(cumul[1] + cumul[2])
    print str(v) + " " + str(n_moving_average(400, len(lab_v) - 1, lab_v))

def get_max_convergence_length_run():
    
    k = 100
    pickle_in = open("saved_graph.pickle","rb")
    network_graph = pickle.load(pickle_in)
    x_l = []
    for i in range(1,2):
        all_step_statistics = []
        v = [random.random() for x in range(0,4)]
        v[1] = random.uniform(0, (1 - v[0])* 2 / 3)
        v[2] = random.uniform(0, 0.2 * 2/3)
        print i
        print v
            
        person_list = generate_people_list_graph_from_network_graph(network_graph,v)
        person_list[0].opinion.opinion_val = Opinion.opinionType.strong_A
        person_list[1].opinion.opinion_val = Opinion.opinionType.strong_B
        network2 = generate_network_graph_from_person_list(person_list)
        pos = nx.spring_layout(network2)
        n = 1000
        for x in range(1,n):
            stats = StepStatistics()
            for person in person_list:
                person.determine_and_add_friend()
            for person in person_list:
                person.determine_and_send_transmission(stats)
            for person in person_list:
                person.update_opinion(stats)
            network2 = generate_network_graph_from_person_list(person_list)
            for edge in network2.edges():
                p1 = person_list[edge[0]].opinion.opinion_val.value
                p2 = person_list[edge[1]].opinion.opinion_val.value
                a = min(p1,p2)
                b = max(p1,p2)
                i = stats.comb_i.index((a,b))
                stats.edge_counts[i] += 1
                
            all_step_statistics.append(stats)
            cumul = [1,0,k-2,0,1]
            s1_v = []
            ma_v = []
            dma_v = []
            madma_v = []
            for j in range(0,len(all_step_statistics)):
                d_cumul_neg = [(-1)*sum(sublist) for sublist in all_step_statistics[j].transitions_counts ]
                transitions_transpose = map(list, zip(*all_step_statistics[j].transitions_counts))
                d_cumul_pos = [sum(sublist) for sublist in transitions_transpose]
                cumul = map(operator.add, cumul, d_cumul_pos)
                cumul = map(operator.add, cumul, d_cumul_neg)
                s1_v.append(cumul[2])
            for l in range(0, len(all_step_statistics)):
                ma_v.append(n_moving_average(50, l, s1_v))
                if l == 0 or l == len(all_step_statistics) - 1:
                    dma_v.append(0)
                else:
                    dma_v.append((n_moving_average(50, l+1, s1_v) - n_moving_average(50, l-1, s1_v))*25)
                madma_v.append(n_moving_average(20, l, dma_v))
                # plt.plot(s1_v, 'ro')
                # plt.plot(ma_v, 'bo')
                # plt.plot(dma_v, 'go')
                # plt.plot(madma_v, 'co')
                # plt.show()
            if x > 50 and madma_v[x-1] == 0:
               x_l.append(x)
               break
    print x_l
    print max(x_l)
    
if __name__ == '__main__':
    random.seed()
    check_reliability_optimized_leanAB()