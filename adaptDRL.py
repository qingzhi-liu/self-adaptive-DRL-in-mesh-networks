#!/usr/bin/python

import os
from xdg.Mime import FREE_NS
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import keras
import numpy as np
import random
import json
import math
import networkx as nx
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import sgd
from keras.models import model_from_json
from keras.optimizers import Adam

class Cluster(object):
    def __init__(self):
        self.init_states()
        self.setup_network()
        self.setup_parameters()

    def init_states(self):
        self.state_G=nx.Graph()
        self.state_cluster_G=nx.Graph()
        self.state_xcor = []
        self.state_ycor = []
        self.state_link = []
        self.state_header = []
        self.state_cluster_id = []
        self.state_border_node = []
        self.state_send_workload = []        
        self.state_cluster_topology_link = []
        self.cluster_topology_links_work_source = []
        self.cluster_topology_links_work_target = []
        self.state_cluster_topology_flow = []
        self.state_server_aggregate_flow = []
        self.state_server_aggregate_flow_speed = []
        self.state_flow_in_network = []
        self.state_fail_flow_counter = 0
        self.state_pass_flow_rate = 0
        self.state_total_reward = 0
        self.state_flow_in_network_matrix = []
        self.total_reward = 0

    def reuse_network(self, s_xcor, s_ycor):
        self.reuse_network_topology(s_xcor, s_ycor)

    def setup_network(self):
        self.set_network_topology()
        while self.all_nodes_connected() == False:
            self.set_network_topology()
    
    def setup_parameters(self):        
        self.set_node_work()
        self.set_cluster_id()        
        self.set_border_node()
        self.set_cluster_topology_link()
        self.set_cluster_topology_link_pair()
        self.set_cluster_topology_flow()
    
    def draw_network(self):
        labeldict = {}
        for i in range(0, node_number+server_number):
            labeldict[i] = str(i) + " | " + str(self.state_cluster_id[i])
        
        pos=nx.kamada_kawai_layout(self.state_G)
        nx.draw(self.state_G, pos, labels=labeldict, with_labels=True, cmap=plt.get_cmap('Accent'), node_color=self.state_cluster_id, node_size=200)
        plt.show()        
    
    def check_neighbor_distance_larger_than_min_range(self, node_id):
        good_position = 1
        for j in range(0, node_id):
            ax = self.state_xcor[node_id]
            ay = self.state_ycor[node_id]
            bx = self.state_xcor[j]
            by = self.state_ycor[j]
            distance = ((ax-bx)**2 + (ay-by)**2)**0.5
            if distance < min_distance_between_nodes:
                good_position = 0
        
        return good_position
    
    def scatter_node_random_position(self):
        for i in range(0, node_number+server_number):
            if cluster_position_collection[i] == 0:
                good_position = 0
                for find_good_position_time in range(0, max_find_good_position_time):
                    if good_position == 0:
                        self.state_xcor[i] = random.random() * deploy_range_x
                        self.state_ycor[i] = random.random() * deploy_range_y
                        good_position = self.check_neighbor_distance_larger_than_min_range(i)
               
        if save_state_counter == 1:
            for i in range(0, node_number+server_number):
                if cluster_position_collection[i] == 0:
                    empty = 0
            for i in range(0, node_number+server_number):
                if cluster_position_collection[i] != 0:
                    self.state_xcor[i] = self.state_xcor[cluster_position_collection[i]]
                    self.state_ycor[i] = self.state_ycor[cluster_position_collection[i]]
        else:
            for i in range(0, node_number+server_number):
                if cluster_position_collection[i] == 0:
                    self.state_xcor[i] = save_state_xcor[i]
                    self.state_ycor[i] = save_state_ycor[i]
            for i in range(0, node_number+server_number):
                if cluster_position_collection[i] != 0:
                    self.state_xcor[i] = self.state_xcor[cluster_position_collection[i]]
                    self.state_ycor[i] = self.state_ycor[cluster_position_collection[i]]
                
    
    def set_network_connectivity(self):
        self.state_link = []
        for i in range(0, node_number+server_number):
            node_link = []
            for j in range(0, node_number+server_number):
                if i!=j and ((self.state_xcor[i]-self.state_xcor[j])**2 + (self.state_ycor[i]-self.state_ycor[j])**2)**0.5 <= transmit_range:
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)
        self.set_graph()
    
    def reuse_network_topology(self, s_xcor, s_ycor):
        self.state_xcor = s_xcor
        self.state_ycor = s_ycor
        self.set_network_connectivity()
    
    def set_network_topology(self):
        self.state_xcor = []
        self.state_ycor = []
        for i in range(0, node_number+server_number):
            self.state_xcor.append(0)
            self.state_ycor.append(0)
        
        if flag_random_position == 1:
            self.scatter_node_random_position()
        else:
            for i in range(0, node_number+server_number):
                if cluster_position_collection[i] == 0:
                    good_position = 0
                    for find_good_position_time in range(0, max_find_good_position_time):
                        if good_position == 0:
                            self.state_xcor[i] = (i % grid_xcor_node_number)* grid_distance
                            self.state_ycor[i] = int(i / grid_xcor_node_number) * grid_distance
                            good_position = self.check_neighbor_distance_larger_than_min_range(i)
            
            if save_state_counter == 1:
                for i in range(0, node_number+server_number):
                    if cluster_position_collection[i] == 0:
                        empty = 0
                for i in range(0, node_number+server_number):
                    if cluster_position_collection[i] != 0:
                        self.state_xcor[i] = self.state_xcor[cluster_position_collection[i]]
                        self.state_ycor[i] = self.state_ycor[cluster_position_collection[i]]
            else:
                for i in range(0, node_number+server_number):
                    if cluster_position_collection[i] == 0:
                        self.state_xcor[i] = save_state_xcor[i]
                        self.state_ycor[i] = save_state_ycor[i]
                for i in range(0, node_number+server_number):
                    if cluster_position_collection[i] != 0:
                        self.state_xcor[i] = self.state_xcor[cluster_position_collection[i]]
                        self.state_ycor[i] = self.state_ycor[cluster_position_collection[i]]
        
        self.set_network_connectivity()
    
    def set_node_workload(self):
        self.state_send_workload = []
        for i in range(server_number, node_number+server_number):
            if i == hot_node_id:
                self.state_send_workload.append(hot_node_work_load)
            else:
                self.state_send_workload.append(0)
    
    def set_node_work(self):
        self.state_workload_in_wait_queue = []
        self.state_server_aggregate_flow = []
        self.state_header = []
        self.set_node_workload()
        
        for i in range(0, server_number):
            self.state_server_aggregate_flow.append(0)
        
        for i in range(0, len(cluster_header_collection)):
            self.state_header.append(cluster_header_collection[i])
    
    def reset_state_statistics(self):
        self.check_graph_error()
        
        for i in range(0, server_number):
            self.state_server_aggregate_flow[i] = 0
        
        for i in range(0, len(self.state_header)):
            for j in range(0, len(self.state_header)):
                self.state_cluster_topology_flow[i][j] = 0
    
    def set_cluster_id(self):
        self.state_cluster_id = [0] * (node_number+server_number)
        for i in range(0, node_number+server_number):
            closest_header_id = -1
            closest_distance = node_number
            for j in range(0, len(self.state_header)):
                if isinstance(self.state_header[j], int):
                    header_id = self.state_header[j]
                    hop_distance = len(self.find_route(i, header_id)) - 1
                    if hop_distance < closest_distance:
                        closest_header_id = header_id
                        closest_distance = hop_distance
                else:
                    header_id_of_cluster = self.state_header[j][0]
                    for k in range(0, len(self.state_header[j])):
                        header_id = self.state_header[j][k]
                        hop_distance = len(self.find_route(i, header_id)) - 1
                        if hop_distance < closest_distance:
                            closest_header_id = header_id_of_cluster
                            closest_distance = hop_distance    
            self.state_cluster_id[i] = closest_header_id
        
        all_cluster_id = list(set(self.state_cluster_id))
        for i in range(0, node_number+server_number):
            original_cluster_id = self.state_cluster_id[i]
            new_cluster_id = all_cluster_id.index(original_cluster_id) + server_number
            self.state_cluster_id[i] = new_cluster_id
    
    def set_border_node(self):
        self.state_border_node = [0] * (node_number+server_number)
        for i in range(0, node_number+server_number):
            self.state_border_node[i] = 0
            for j in range(0, node_number+server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1 and self.state_cluster_id[i] != self.state_cluster_id[j]:
                    self.state_border_node[i] = 1
    
    def set_cluster_topology_flow(self):
        for i in range(0, len(self.state_header)):
            cluster_flow = [0] * cluster_header_number
            self.state_cluster_topology_flow.append(cluster_flow)
    
    def set_cluster_topology_link(self):
        self.state_cluster_topology_link = []
        for i in range(0, len(self.state_header)):
            cluster_link = []
            for j in range(0, len(self.state_header)):
                cluster_link.append(0)
            self.state_cluster_topology_link.append(cluster_link)
        for i in range(0, node_number+server_number):
            for j in range(0, node_number+server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1 and self.state_cluster_id[i] != self.state_cluster_id[j]:
                    self.state_cluster_topology_link[self.state_cluster_id[i]-server_number][self.state_cluster_id[j]-server_number] = 1
                    self.state_cluster_topology_link[self.state_cluster_id[j]-server_number][self.state_cluster_id[i]-server_number] = 1
    
    def set_cluster_topology_link_pair(self):
        for cluster_id_s_counter in range(0, len(self.state_header)):
            for cluster_id_t_counter in range(cluster_id_s_counter, len(self.state_header)):
                if isinstance(self.state_header[cluster_id_s_counter], int):
                    cluster_id_s = self.state_cluster_id[self.state_header[cluster_id_s_counter]]
                else:
                    cluster_id_s = self.state_cluster_id[self.state_header[cluster_id_s_counter][0]]
                    
                if isinstance(self.state_header[cluster_id_t_counter], int):
                    cluster_id_t = self.state_cluster_id[self.state_header[cluster_id_t_counter]]
                else:
                    cluster_id_t = self.state_cluster_id[self.state_header[cluster_id_t_counter][0]]    
                
                if self.state_cluster_topology_link[cluster_id_s-server_number][cluster_id_t-server_number] == 1:
                    self.cluster_topology_links_work_source.append(cluster_id_s)
                    self.cluster_topology_links_work_target.append(cluster_id_t)
    
    def all_nodes_connected(self):
        for i in range(0, node_number+server_number):
            for j in range(0, node_number+server_number):
                check = nx.has_path(self.state_G, i, j)
                if check == False:
                    return False
        return True    
    
    def set_graph(self):
        self.state_G=nx.Graph()
        for i in range(0, node_number+server_number):
            self.state_G.add_node(i)
        for i in range(0, node_number+server_number):
            for j in range(i, node_number+server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
    
    def find_route(self, s, t):
        check = nx.has_path(self.state_G, source=s, target=t)
        if check == True:
            path = nx.dijkstra_path(self.state_G, source=s, target=t)
        else:
            path = []  
        print('The path is: ' + str(s) + '-' + str(t) + ' = ' + str(path))      
        return path
    
    def set_flow_in_network(self, start_node, end_node, start_time, deadline_time_length, work_load, pass_route, flow_set_done):
        exist_flow = 0
        for i in range(0, len(self.state_flow_in_network)):
            if self.state_flow_in_network[i][0] == start_node and self.state_flow_in_network[i][1] == end_node and self.state_flow_in_network[i][2] == start_time:
                exist_flow = 1
                self.state_flow_in_network[i][5] = pass_route
        
        if exist_flow == 0 and flow_set_done == 0:
            new_flow_state = [start_node, end_node, start_time, deadline_time_length, work_load, pass_route]
            self.state_flow_in_network.append(new_flow_state)
    
    def get_flow_source(self):
        flow_source_nodes = []
        for i in range(0, len(self.state_flow_in_network)):
            source = self.state_flow_in_network[i][0]
            flow_source_nodes = flow_source_nodes + [source]
            
        return flow_source_nodes
    
    def transmit_flow_in_network(self):
        flow_in_network_matrix = []
        for i in range(0, node_number+server_number):
            node_link = [0] * (node_number+server_number)
            flow_in_network_matrix.append(node_link)
        
        for i in range(0, len(self.state_flow_in_network)):
            pass_route = self.state_flow_in_network[i][5]
            if len(pass_route) > 0:
                for p in range(0, len(pass_route)-1):
                    start_in_route_link = pass_route[p]
                    end_in_route_link = pass_route[p+1]
                    flow_in_network_matrix[start_in_route_link][end_in_route_link] = flow_in_network_matrix[start_in_route_link][end_in_route_link] + 1
                    
        self.state_flow_in_network_matrix = flow_in_network_matrix
        
        flow_communication_speed = [0] * len(self.state_flow_in_network)
        for i in range(0, len(self.state_flow_in_network)):
            pass_route = self.state_flow_in_network[i][5]
            if len(pass_route) > 0:
                max_flow_number = 1
                for p in range(0, len(pass_route)-1):
                    start_in_route_link = pass_route[p]
                    end_in_route_link = pass_route[p+1]
                    if flow_in_network_matrix[start_in_route_link][end_in_route_link] > max_flow_number:
                        max_flow_number = flow_in_network_matrix[start_in_route_link][end_in_route_link]
                    
                    if self.state_cluster_id[start_in_route_link] == self.state_cluster_id[end_in_route_link]:
                        if flow_in_network_matrix[start_in_route_link][end_in_route_link] > 1:
                            c_id = self.state_cluster_id[start_in_route_link]
                            sta_hot_cluster[c_id - server_number] = sta_hot_cluster[c_id - server_number] + 1
                
                flow_communication_speed[i] = communication_speed / max_flow_number
            else:
                flow_communication_speed[i] = 0
        self.state_server_aggregate_flow_speed = flow_communication_speed
        
        sum_flow_in_network_pass_rate = [0] * len(self.state_flow_in_network)
        counter_flow_in_network_pass_rate = [0] * len(self.state_flow_in_network)
        avg_flow_in_network_pass_rate = [0] * len(self.state_flow_in_network)
        for i in range(0, len(self.state_flow_in_network)):
            pass_route = self.state_flow_in_network[i][5]
            if len(pass_route) > 0:
                for p in range(0, len(pass_route)-1):
                    start_in_route_link = pass_route[p]
                    end_in_route_link = pass_route[p+1]
                    if flow_in_network_matrix[start_in_route_link][end_in_route_link] != 0:
                        sum_flow_in_network_pass_rate[i] = sum_flow_in_network_pass_rate[i] + (1 / flow_in_network_matrix[start_in_route_link][end_in_route_link])
                        counter_flow_in_network_pass_rate[i] = counter_flow_in_network_pass_rate[i] + 1
        
        for i in range(0, len(self.state_flow_in_network)):
            if counter_flow_in_network_pass_rate[i] != 0:
                avg_flow_in_network_pass_rate[i] = sum_flow_in_network_pass_rate[i] / counter_flow_in_network_pass_rate[i]
            else:
                avg_flow_in_network_pass_rate[i] = 0
        
        if len(avg_flow_in_network_pass_rate) != 0:
            self.state_pass_flow_rate = sum(avg_flow_in_network_pass_rate) / len(avg_flow_in_network_pass_rate)
        else:
            self.state_pass_flow_rate = 0
        
        for i in range(0, len(self.state_flow_in_network)):
            work_load = self.state_flow_in_network[i][4]
            work_load = work_load - flow_communication_speed[i]
            self.state_flow_in_network[i][4] = work_load
        
        for s in range(0, server_number):
            self.state_server_aggregate_flow[s] = np.sum(flow_communication_speed)
        
        for i in range(0, len(self.state_flow_in_network)):
            start_node = self.state_flow_in_network[i][0]
            pass_route = self.state_flow_in_network[i][5]
            
            for p in range(0, len(pass_route)-1):
                start_in_route_link = pass_route[p]
                end_in_route_link = pass_route[p+1]
                if self.state_cluster_id[start_in_route_link] != self.state_cluster_id[end_in_route_link]:
                    sid = self.state_cluster_id[start_in_route_link]-server_number
                    tid = self.state_cluster_id[end_in_route_link]-server_number
                    self.state_cluster_topology_flow[sid][tid] = self.state_cluster_topology_flow[sid][tid] + 1
                    self.state_cluster_topology_flow[tid][sid] = self.state_cluster_topology_flow[tid][sid] + 1
    
    def check_flow_in_network_fail(self, tick):
        del_record = 1
        while del_record == 1:
            del_record = 0
            for i in range(0, len(self.state_flow_in_network)):
                s_time = self.state_flow_in_network[i][2]
                d_time_length = self.state_flow_in_network[i][3]
                w_load = self.state_flow_in_network[i][4]
                if w_load <= 0:
                    del self.state_flow_in_network[i]
                    del_record = 1
                    break
                if tick > (s_time + d_time_length) and w_load > 0:
                    self.state_fail_flow_counter = self.state_fail_flow_counter + 1
                    del self.state_flow_in_network[i]
                    del_record = 1
                    break
        
        return 0
    
    def check_graph_error(self):
        for i in range(0, len(self.state_header)):
            for j in range(0, len(self.state_header)):
                if self.state_cluster_topology_link[i][j] != self.state_cluster_topology_link[j][i]:
                    print("Error: cluster network topology.")
                    exit()
        
        for i in range(0, node_number+server_number):
            for j in range(0, node_number+server_number):
                if self.state_link[i][j] != self.state_link[j][i]:
                    print("Error: node network topology.")
                    exit()
    
    def reset_node_link(self, world_tick):
        if world_tick % hot_node_period == 0:
            for i in range(0, node_number+server_number):
                for j in range(0, node_number+server_number):
                    if i!=j and ((self.state_xcor[i]-self.state_xcor[j])**2 + (self.state_ycor[i]-self.state_ycor[j])**2)**0.5 <= transmit_range:
                        self.state_link[i][j] = 1
            
            r = int(world_tick/2)%2
            if r == 0:
                self.state_link[0][0] = 0
            else:
                self.state_link[0][0] = 0
    
    def action_route(self, action, world_tick, tick):
        cid = action % len(self.cluster_topology_links_work_source)
        set_switch = 0
        if action < len(self.cluster_topology_links_work_source):
            set_switch = 1
        else:
            set_switch = 0
        
        cluster_id_source = self.cluster_topology_links_work_source[cid]
        cluster_id_target = self.cluster_topology_links_work_target[cid]
        
        for i in range(0, node_number+server_number):
            for j in range(i, node_number+server_number):
                if i!=j and ((self.state_xcor[i]-self.state_xcor[j])**2 + (self.state_ycor[i]-self.state_ycor[j])**2)**0.5 <= transmit_range:
                    if self.state_cluster_id[i] != self.state_cluster_id[j]:
                        if (self.state_cluster_id[j] == cluster_id_source and self.state_cluster_id[i] == cluster_id_target) or (self.state_cluster_id[i] == cluster_id_source and self.state_cluster_id[j] == cluster_id_target): 
                            self.state_link[i][j] = set_switch
                            self.state_link[j][i] = set_switch
        
        if flag_fixed_topology_test == 1:
            self.reset_node_link(world_tick)
        
        self.check_graph_error()
        self.set_cluster_topology_link()
        self.set_graph()
        self.check_graph_error()
    
    def get_reward_aggregate_flow(self):
        reward = 0
        for s in range(0, server_number):
            reward = reward + self.state_server_aggregate_flow[s]
        
        return reward

    def get_arrive_rate(self):
        pass_route = [1]*len(hot_node_collection)
        for hot_node_id_counter in range(0, len(hot_node_collection)):
            i = hot_node_collection[hot_node_id_counter]
            for s in range(0, server_number):
                path = self.find_route(i, s)
                if len(path) != 0:
                    pass_route[hot_node_id_counter] = 1
                else:
                    pass_route[hot_node_id_counter] = 0
        
        route_rate = np.sum(pass_route)/len(hot_node_collection)
        return route_rate
        
    def get_pass_rate(self):
        server_aggregate_flow_speed_rate = np.array(self.state_server_aggregate_flow_speed) / communication_speed
        if len(server_aggregate_flow_speed_rate) == 0:
            reward_pass_rate = 1
        else:
            reward_pass_rate = (np.sum(server_aggregate_flow_speed_rate)/len(server_aggregate_flow_speed_rate))
        return reward_pass_rate
    
    def get_reward(self):
        aggregate_flow = self.get_reward_aggregate_flow()
        arrive_rate = self.get_arrive_rate()
        pass_rate = self.get_pass_rate()
        reward = aggregate_flow * arrive_rate * pass_rate
        
        return reward
    
    def choose_action(self, p_state):
        if np.random.rand() <= exp_replay.epsilon or len(p_state) == 0:
            action = np.random.randint(0, model_output_size)
        else:
            q = exp_replay.model.predict(p_state)
            action = np.argmax(q[0])
        
        return action
    
    def get_state(self, world_tick, tick):
        cluster_links = [0] * len(env.cluster_topology_links_work_source)
        for v in range(0, len(env.cluster_topology_links_work_source)):
            s = self.cluster_topology_links_work_source[v]
            t = self.cluster_topology_links_work_target[v]
            cluster_links[v] = self.state_cluster_topology_link[s-server_number][t-server_number]
        
        state = [self.state_cluster_id[hot_node_id], (tick % train_ticks), self.state_fail_flow_counter]
        state = state + hot_node_collection + [hot_node_period]
        
        for i in range(0, len(self.state_header)):
            state = state + self.state_cluster_topology_link[i]
        
        for i in range(0, len(self.state_header)):
            state = state + self.state_cluster_topology_flow[i]
        
        self.check_graph_error()
        p_state = np.asarray(state)
        p_state = p_state[np.newaxis]
        return p_state
    
    def act_action(self, p_state, world_tick, tick):
        if flag_run_benchmark == 0:
            action = self.choose_action(p_state)
            self.action_route(action, world_tick, tick)
        else:
            action = 0
        
        return action
    
    def act_reward(self, world_tick, tick):
        reward = self.get_reward()
        p_next_state = self.get_state(world_tick, tick)
        
        return reward, p_next_state

class ExperienceReplay(object):
    def __init__(self, epsilon):
        self.memory = list()
        self.model = self._build_model()
        self.epsilon = epsilon

    def _build_model(self):
        model = Sequential()
        model.add(Dense(hidden_size0, input_shape=(model_input_size,), activation='relu'))
        model.add(Dense(hidden_size1, activation='relu'))
        model.add(Dense(model_output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model
    
    def remember(self, state):
        self.memory.append([state])
        if len(self.memory) > max_memory:
            del self.memory[0]
    
    def replay(self):
        minibatch = random.sample(self.memory, batch_size)
        for mem_state in minibatch:
            state, action, reward, next_state, tick, game_end_tick, game_over = mem_state[0]
            
            if tick == game_end_tick or game_over == 1:
                target = reward
            else:
                target = (reward + discount * np.amax(self.model.predict(next_state)[0]))
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > epsilon_min:
            self.epsilon = self.epsilon * epsilon_decay

if __name__ == "__main__":
    flag_draw_graph = 1
    flag_fixed_topology_test = 0
    flag_random_position = 0
    flag_hot_node_part_of_clusters = 1
    flag_save_model = 0
    flag_run_benchmark = 1
    
    state_schedule = 1
    save_state_xcor = []
    save_state_ycor = []
    save_state_counter = 0
    hot_node_collection = []
    history_content = []
    history_topology_x = []
    history_topology_y = []
    
    if flag_run_benchmark == 1:
        with open("schedule_history.txt") as f:
            history_content = f.readlines()
        history_content = [x.strip() for x in history_content]
        for i in range(0, len(history_content)):
            history_content[i] = list(map(int, history_content[i].split()))
        
        with open("schedule_topology_x.txt") as f:
            history_topology_x = f.readlines()
        history_topology_x = [x.strip() for x in history_topology_x]
        history_topology_x = [x.strip('[]') for x in history_topology_x]
        history_topology_x = [x.strip(',') for x in history_topology_x]
        for i in range(0, len(history_topology_x)):
            history_topology_x[i] = list(map(float, history_topology_x[i].split(', ')))
            
        with open("schedule_topology_y.txt") as f:
            history_topology_y = f.readlines()
        history_topology_y = [x.strip() for x in history_topology_y]
        history_topology_y = [x.strip('[]') for x in history_topology_y]
        history_topology_y = [x.strip(',') for x in history_topology_y]
        for i in range(0, len(history_topology_y)):
            history_topology_y[i] = list(map(float, history_topology_y[i].split(', ')))
    
    while True:
        save_state_counter = save_state_counter + 1
        if state_schedule != 0:
            insert_new_cluster_seed = []
            f = open("schedule_headers.txt", "r")
            for x in f:
                all_list = [int(x)]
                insert_new_cluster_seed = insert_new_cluster_seed + all_list
            print("Cluster Headers: ", insert_new_cluster_seed)
            cluster_header_collection = insert_new_cluster_seed
            
            total_node_number = len(cluster_header_collection)
            server_number = 1
            node_number = total_node_number-server_number
            
            transmit_range = 11
            min_distance_between_nodes = 3
            grid_xcor_node_number = 4
            grid_ycor_node_number = 5
            grid_distance = 10
            deploy_range_x = (grid_xcor_node_number - 1) * grid_distance
            deploy_range_y = (grid_ycor_node_number - 1) * grid_distance
            
            max_find_good_position_time = 3
            max_find_hotnode_position_time = total_node_number*10
            hot_node_id = 0
            communication_speed = 1
            hot_node_period = 2
            
            insert_new_cluster_seed = []
            f = open("schedule_positions.txt", "r")
            for x in f:
                all_list = [int(x)]
                insert_new_cluster_seed = insert_new_cluster_seed + all_list
            print("Cluster Positions: ", insert_new_cluster_seed)
            cluster_position_collection = insert_new_cluster_seed
            if len(cluster_position_collection) != total_node_number:
                print("Error: cluster position.")
            
            if flag_hot_node_part_of_clusters == 0:
                if save_state_counter == 1:
                    hot_node_collection = []
                    for i in range(0, len(cluster_header_collection)):
                        if cluster_header_collection[i] != 0:
                            if cluster_position_collection[i] == 0:
                                a = cluster_header_collection[i]
                                hot_node_collection.append(a)
            else:
                if save_state_counter == 1:
                    hot_node_collection = [1, 2, 3]
            
            print("Cluster Hotspots: ", hot_node_collection)
            
            f = open("schedule_results.txt", "a")
            f.write(">>>>>>>>>>>>>>>>>>>" + str(save_state_counter) + ": " + str(cluster_header_collection) + "==" + str(cluster_position_collection) + "<<<<<<<<<<<<<<<<<<<<\n")
            f.close()
            
            cluster_header_number = len(cluster_header_collection)
            
            env = Cluster()
            if flag_run_benchmark == 0:
                save_state_xcor = env.state_xcor
                save_state_ycor = env.state_ycor
            else:
                save_state_xcor = history_topology_x[0]
                save_state_ycor = history_topology_y[0]
            
            env.init_states()
            env.reuse_network(save_state_xcor, save_state_ycor)
            env.setup_parameters()
            
            cluster_link_number = len(env.cluster_topology_links_work_source)
            train_ticks = cluster_link_number
            hot_node_work_load = hot_node_period * 32
            communication_deadline_lengh = (hot_node_work_load / communication_speed) * 2
            print("Hot node workload: ", hot_node_work_load)
            print("Train ticks: ", train_ticks)
                        
            epoch = 200
            game_time = 10
            learning_rate = 0.0001
            discount = 0.5
            epsilon = 1.0
            epsilon_min = 0.01
            epsilon_decay = 0.999
            hidden_size0 = 200
            hidden_size1 = 100
            max_memory = 2000
            batch_size = 10
            
            model_input_size = 4 + len(hot_node_collection) + cluster_header_number**2 + cluster_header_number**2
            model_output_size = 2 * cluster_link_number
            exp_replay = ExperienceReplay(epsilon)
            
            sta_congestion_in_rounds_rate = 0.1
            sta_congestion_in_rounds = int(epoch*sta_congestion_in_rounds_rate)
            sta_max_total_reward = 0.0
            sta_max_total_tick = 0.0
            sta_hot_cluster = [0] * cluster_header_number
            sta_congestion_in_network_matrix = []
            for i in range(0, node_number+server_number):
                node_link = [0] * (node_number+server_number)
                sta_congestion_in_network_matrix.append(node_link)
            
            sta_free_clusters = [0] * cluster_header_number
            sta_free_clusters_counter = 0
            
            history_success_rate = []
            evaluate_success_rate = 0.9
            evaluate_busy_cluster_rate = 0.9
            
            if flag_draw_graph == 1 and save_state_counter == 1:
                env.draw_network()
            
            if flag_run_benchmark == 0:
                f = open("schedule_topology_x.txt", "a")
                f.write(str(save_state_xcor) + "\n")
                f.close()
                f = open("schedule_topology_y.txt", "a")
                f.write(str(save_state_ycor) + "\n")
                f.close()
            
            for a in range(epoch):
                env.init_states()
                env.reuse_network(save_state_xcor, save_state_ycor)
                env.setup_parameters()
                
                print("====================================================================================================================")
                print(env.cluster_topology_links_work_source)
                print(env.cluster_topology_links_work_target)
                
                sta_max_reward = 0.0
                sta_total_reward = 0
                sta_cluster_links = [0] * len(env.cluster_topology_links_work_source)
                sta_flow_in_run = 0
                sta_total_reward_flow = [0] * (game_time*train_ticks)
                sta_high_bound_reward_counter = 0
                
                mem_state = []
                game_over = 0
                env.state_fail_flow_counter = 0
                
                for world_tick in range(0, game_time):
                    print("--------------------------------------------------------------------------------------------------------------------")
                    
                    hot_node_id = 0
                    make_flow = 0
                    
                    if world_tick % hot_node_period == 0:
                        source_nodes = env.get_flow_source()
                        
                        if len(source_nodes) == 0:
                            pick_hot_node_index = random.randint(0, len(hot_node_collection)-1)
                            pick_hot_node = hot_node_collection[pick_hot_node_index]
                            hot_node_id = pick_hot_node
                            make_flow = 1
                        else:
                            hot_node_id = source_nodes[0]
                            find_hotnode_time = 0
                            while hot_node_id in source_nodes:
                                pick_hot_node_index = random.randint(0, len(hot_node_collection)-1)
                                
                                pick_hot_node = hot_node_collection[pick_hot_node_index]
                                
                                hot_node_id = pick_hot_node
                                
                                make_flow = 1
                                
                                find_hotnode_time = find_hotnode_time + 1
                                if find_hotnode_time > max_find_hotnode_position_time:
                                    make_flow = 0
                                    break
                    
                    env.set_node_workload()
                    
                    game_start_tick = world_tick*train_ticks
                    game_end_tick = game_time*train_ticks - 1
                    flow_set_done = 0
                    
                    for tick in range (world_tick*train_ticks, (world_tick+1)*train_ticks):
                        if flag_run_benchmark == 0:
                            f = open("schedule_history.txt", "a")
                            f.write(str(save_state_counter)+' '+str(a)+' '+str(world_tick)+' '+str(tick)+' '+str(state_schedule)+' '+str(make_flow)+' '+str(hot_node_id)+' '+"\n")
                            f.close()
                        else:
                            run_history_step = -1
                            run_history_step_counter = 0
                            for i in range(0, len(history_content)):
                                run_history_step_counter = run_history_step_counter + 1
                                if history_content[i][0]==save_state_counter and history_content[i][1]==a and history_content[i][2]==world_tick and history_content[i][3]==tick:
                                    run_history_step = i
                                    break
                            
                            if run_history_step == -1:
                                if run_history_step_counter < len(history_content):
                                    print("Error: run schedule history in benchmark.")
                                    exit()
                                else:
                                    state_schedule = 0
                                    make_flow = 0
                                    hot_node_id = 0
                                    
                                    game_over = 1
                            else:
                                state_schedule = history_content[i][4]
                                make_flow = history_content[i][5]
                                hot_node_id = history_content[i][6]
                        
                        if make_flow == 1:
                            if tick == world_tick*train_ticks:
                                time_cost = math.ceil(hot_node_work_load / communication_speed)
                                for ti in range(tick, tick+time_cost):
                                    if ti < len(sta_total_reward_flow):
                                        sta_total_reward_flow[ti] = sta_total_reward_flow[ti] + 1
                            
                            action = env.act_action(mem_state, world_tick, tick)
                            
                            start_node = hot_node_id
                            end_node = server_number-1
                            start_time = game_start_tick
                            deadline_time_length = communication_deadline_lengh
                            work_load = hot_node_work_load
                            pass_route = env.find_route(start_node, end_node)
                            print(pass_route)
                            
                            env.set_flow_in_network(start_node, end_node, start_time, deadline_time_length, work_load, pass_route, flow_set_done)
                            
                            if flow_set_done == 0:
                                flow_set_done = 1
                            
                            env.transmit_flow_in_network()
                            
                            reward, state = env.act_reward(world_tick, tick)
                            if tick == 0:
                                exp_replay.remember([state, action, reward, state, tick, game_end_tick, game_over])
                            else:
                                exp_replay.remember([mem_state, action, reward, state, tick, game_end_tick, game_over])
                            mem_state = state
                            
                            if len(exp_replay.memory) >= batch_size:
                                exp_replay.replay()
                            
                            sta_total_reward = sta_total_reward + reward
                            
                            sta_high_bound_reward_counter = sta_high_bound_reward_counter + sta_total_reward_flow[tick]
                            
                            sta_flow_in_run = len(env.state_flow_in_network)
                            if sta_total_reward > sta_max_total_reward:
                                sta_max_total_reward = sta_total_reward
                            if reward > sta_max_reward:
                                sta_max_reward = reward
                            if tick > sta_max_total_tick:
                                sta_max_total_tick = tick
                            for v in range(0, len(env.cluster_topology_links_work_source)):
                                s = env.cluster_topology_links_work_source[v]
                                t = env.cluster_topology_links_work_target[v]
                                sta_cluster_links[v] = env.state_cluster_topology_link[s-server_number][t-server_number]
                            print("Epoch {:03d}/{} | Tick {}/{} | Position {} | Reward {}/{} | Link {} | Flow {}".format(a, epoch, tick, train_ticks*game_time, hot_node_id, round(reward, 1), round(sta_max_reward, 1), sta_cluster_links, sta_flow_in_run))
                            
                            if (epoch - a - 1) < sta_congestion_in_rounds:
                                sta_congestion_in_network_matrix = (np.array(sta_congestion_in_network_matrix) + np.array(env.state_flow_in_network_matrix)).tolist()
                                sta_free_clusters_counter = sta_free_clusters_counter + 1
                                
                                for ct in range(0, cluster_header_number):
                                    check_free_cluster = 0
                                    
                                    for flow_i in range(0, cluster_header_number):
                                        if env.state_cluster_topology_flow[ct][flow_i] != 0 or env.state_cluster_topology_flow[flow_i][ct] != 0:
                                            check_free_cluster = check_free_cluster + 1
                                    
                                    for neighbor_ct in range(0, cluster_header_number):
                                        if env.state_cluster_topology_link[ct][neighbor_ct] == 1 or env.state_cluster_topology_link[neighbor_ct][ct] == 1:
                                            for flow_i in range(0, cluster_header_number):
                                                if env.state_cluster_topology_flow[neighbor_ct][flow_i] != 0 or env.state_cluster_topology_flow[flow_i][neighbor_ct] != 0:
                                                    check_free_cluster = check_free_cluster + 1
                                    
                                    if check_free_cluster != 0:
                                        sta_free_clusters[ct] = sta_free_clusters[ct] + 1
                            
                            env.reset_state_statistics()
                        else:
                            env.transmit_flow_in_network()
                            state = env.get_state(world_tick, tick)
                            mem_state = state
                        
                        if state_schedule == 1:
                            game_over = env.check_flow_in_network_fail(tick)
                        
                        if game_over == 1:
                            break
                    
                    if flag_save_model == 1:
                        exp_replay.model.save_weights("model.h5", overwrite=True)
                        with open("model.json", "w") as outfile:
                            json.dump(exp_replay.model.to_json(), outfile)
                    
                    if game_over == 1:
                        break
                
                print("--------------------------------------------------------------------------------------------------------------------")
                print("==>>Epoch {:03d}/{} | Total Tick {}/{} | Total Reward {}/{}/{} | Cluster Block {}".format(a, epoch, tick, sta_max_total_tick, round(sta_total_reward, 1), round(sta_max_total_reward, 1), sta_high_bound_reward_counter, sta_hot_cluster))
                print("====================================================================================================================")
                
                f = open("schedule_results.txt", "a")
                f.write(str(sta_total_reward/sta_high_bound_reward_counter)+"\n")
                f.close()
                
                if (epoch - a - 1) < sta_congestion_in_rounds:
                    history_success_rate.append(sta_total_reward/sta_high_bound_reward_counter)
            
            print("The rate of correct clustering management in the network: ", sum(history_success_rate)/len(history_success_rate), history_success_rate)
            
            for i in range(0, server_number):
                server_cluster_id = env.state_cluster_id[i]
                sta_free_clusters[server_cluster_id - server_number] = sta_free_clusters_counter
                
            for i in range(0, len(hot_node_collection)):
                h_cluster_id = env.state_cluster_id[hot_node_collection[i]]
                sta_free_clusters[h_cluster_id - server_number] = sta_free_clusters_counter
            
            for i in range(0, len(sta_free_clusters)):
                temp = sta_free_clusters[i]
                sta_free_clusters[i] = temp / sta_free_clusters_counter
            
            print("Free Clusters: ", sta_free_clusters)
            
            exist_x_flag = 0
            exist_y_flag = 0
            
            if sum(history_success_rate)/len(history_success_rate) > evaluate_success_rate:
                remove_index = []
                for i in range(0, len(sta_free_clusters)):
                    if sta_free_clusters[i] < 1-evaluate_busy_cluster_rate:
                        remove_index.append(i)
                        
                if len(remove_index) == 0:
                    print("Find the correct clustering management in the network.")
                    state_schedule = 0
                else:
                    print("Remove unnecessary clusters.")
                    new_cluster_header_collection = cluster_header_collection
                    new_cluster_position_collection = cluster_position_collection
                    
                    if len(new_cluster_header_collection) != len(new_cluster_position_collection):
                        print("Error: cluster setup position.")
                    
                    for i in range(0, len(remove_index)):
                        new_cluster_header_collection[remove_index[i]] = -1
                        new_cluster_position_collection[remove_index[i]] = -1
                    
                    pop_exist = 1
                    while pop_exist == 1:
                        pop_exist = 0
                        for i in range(0, len(new_cluster_position_collection)):
                            if new_cluster_header_collection[i] == -1 and new_cluster_position_collection[i] == -1:
                                new_cluster_header_collection.pop(i)
                                new_cluster_position_collection.pop(i)
                                save_state_xcor.pop(i)
                                save_state_ycor.pop(i)
                                pop_exist = 1
                                break
                    
                    new_new_cluster_header_collection = list(new_cluster_header_collection)
                    new_new_cluster_position_collection = list(new_cluster_position_collection)
                    
                    print("Header Settings: ", new_cluster_header_collection, new_cluster_position_collection)
                    
                    for i in range(0, len(new_new_cluster_header_collection)):
                        
                        if new_new_cluster_header_collection[i] in hot_node_collection:
                            node_change_number_index = hot_node_collection.index(new_new_cluster_header_collection[i])
                            hot_node_collection[node_change_number_index] = i
                        
                        new_new_cluster_header_collection[i] = i
                        if new_new_cluster_position_collection[i] != 0:
                            set_node_position = new_cluster_position_collection[i]
                            new_set_node_position = new_cluster_header_collection.index(set_node_position)
                            new_new_cluster_position_collection[i] = new_set_node_position
                    
                    print("Header Settings: ", new_new_cluster_header_collection, new_new_cluster_position_collection)
                    
                    if flag_run_benchmark == 0:
                        f = open("schedule_headers.txt", "w")
                        for i in range(0, len(new_new_cluster_header_collection)):
                            f.write(str(new_new_cluster_header_collection[i])+"\n")
                        f.close()
                        
                        f = open("schedule_positions.txt", "w")
                        for i in range(0, len(new_new_cluster_position_collection)):
                            f.write(str(new_new_cluster_position_collection[i])+"\n")
                        f.close()
            else:
                all_flow_route = []
                for i in range(0, len(sta_congestion_in_network_matrix)):
                    all_flow_route = all_flow_route + sta_congestion_in_network_matrix[i]
                max_flow = max(all_flow_route)
                congestion_from_node = -1
                congestion_to_node = -1
                for i in range(0, len(sta_congestion_in_network_matrix)):
                    for j in range(0, len(sta_congestion_in_network_matrix)):
                        if sta_congestion_in_network_matrix[i][j] == max_flow:
                            congestion_from_node = i
                            congestion_to_node = j
                
                if flag_run_benchmark == 0:
                    if congestion_from_node != -1 and congestion_to_node != -1:
                        print("Need new cluster to solve congestion: ", congestion_from_node, congestion_to_node)
                        f = open("schedule_headers.txt", "a")
                        f.write(str(total_node_number)+"\n")
                        f.close()
                        
                        f = open("schedule_positions.txt", "a")
                        f.write(str(congestion_from_node)+"\n")
                        f.close()
        else:
            break
    
    print("Done: Find the correct clustering management in the network.")
    print("===========================================================================================")
    
# :-)
