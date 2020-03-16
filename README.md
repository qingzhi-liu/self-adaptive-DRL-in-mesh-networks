# self-adaptive-DRL-in-mesh-networks


--------------------------READ ME---------------------------------

The system controls the flow of head nodes in a cluster-based wireless mesh network. 

----------------------------INPUT---------------------------------
Before running the code, you need set up the following parameters:

1. 
Specify the number nodes by setting the number of nodes in rows and columns by "grid_xcor_node_number" and "grid_ycor_node_number" respectively in a grid network. 

2.
Specify whether the grid network nodes are randomly scattered by "flag_random_position". 

3. 
In the setup file "schedule_headers.txt", specify the ID of head nodes in the initial state. 
In the setup file "schedule_positions.txt", specify the position of head nodes in the initial state. In the inialization, set the value of each node to 0, which represents the initial position.   
The same row number in "schedule_headers.txt" and "schedule_positions.txt" represents a node. 

4. 
The parameters to set up the system are as follows.
flag_draw_graph: to draw a graph in the first round.
flag_fixed_topology_test: debug use for testing in a specific topology.
flag_random_position: set the network to random or grid.
flag_hot_node_part_of_clusters: for extension use.
flag_save_model: save DQN model.
flag_run_benchmark: 0=DRL solution; 1=benchmark solution.

5. 
The parameters to set up the DQN model and self-adaptive model are in the code with self-explain variable names.

---------------------------OUTPUT---------------------------------
After running the code, the output files are:

1. 
schedule_topology_x.txt and schedule_topology_x.txt are used to save the position of head nodes. 
The files are produced by the DRL solution. The benchmark does not set the positions of nodes, but MUST reused ones from DRL experiments. 

2. 
schedule_history.txt is used to save the route and data source of head nodes. 
The files are produced by the DRL solution. The benchmark does not set the data sources and route of nodes, but MUST reused ones from DRL experiments. 

3. 
schedule_history.txt is used to save the reward results. 

4.  
schedule_headers.txt: some of the IDs are removed and some new IDs are added compared with the initial state. 
schedule_positions.txt: if the value is 0, it means the position of the node does not change. If there is a non-zero value, it represents the position of the node with the same ID value. 


