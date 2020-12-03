import networkx as nx
import numpy as np
import itertools
from networkx.algorithms.bipartite.generators import random_graph as random_bi_graph

class BaseResourceSimulation():

    def __init__(self, outer_G_args, already_developed_p,
                 dist_scale, comm_G_args, init_pro_dev_p,
                 com_threshold=.5, base_spread_p=.5,
                 outside_influence_k=.75, vote_every=1,
                 vote_threshold=.5, interest_groups = [],
                 seed=None):
        ''' Main class for running resource / voting simulation

        Parameters
        ------------
         outer_G_args : dict
            These parameters are passed a dictionary, and they represent
            the parameters for creating a random bi-partite network,
            with the following params:
            - 'n_resources' : The number of resource nodes.
            - 'n_communities' : The number of community nodes.
            - 'comm_to_resource_p' : The prob. any community will have an edge with any resource.
            See: https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.bipartite.generators.random_graph.html#networkx.algorithms.bipartite.generators.random_graph

        already_developed_p : float
            Between 0 and 1, this represents the starting prob. of any
            resource node, that it should be set to developed at timestep 0.
            E.g., if 0, then none will be set, if 1, then all will be set to
            developed.

        dist_scale : float
            This parameter controls the weighting in generating weighted connections between
            different communities. It represents how strongly the weights between communities should
            be influenced by their physical promixity (as represented by distance on the outer network).

            A dist_scale value of 0, means that distance is ignored, and all connections
            between different communities are drawn from a uniform prob. between 0 and 1.
            Higher values start to weight closer nodes higher and higher, with dist_scale = 1, means
            that on average communities that are distance 1 away, will have twice as strong a link
            as those that are distance 2 away.

            The weight between two communities is calculated as prob. p divided by
            the distance between two communities raised to the dist_scale power.

        comm_G_args : dict
            These parameters are passed as a dictionary, and they represent
            the parameters for generating each communities random network.
            For now, this random network is generated as a random small world
            network, with the following fixed parameters:

            - 'n' : The number of nodes in each community.

            - 'k' : Each node is connected to k nearest neighbors in ring topology.

            - 'p' : The probability of rewiring each edge.

            See: https://networkx.org/documentation/networkx-1.9/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html

        init_pro_dev_p : float
            Between 0 and 1, this represents the starting prob. that any node
            within any of the community networks is set initially to be pro-development.
            This will correspond to roughly the percent of nodes in state 1 in any
            communities network.

        com_threshold : float
            Between 0 and 1, this represent the percent of nodes within a community
            that have to be in state 1, in order for this community to switch to state
            1 in the outer network. In other words, the threshold of pro-resource development
            needed, in order for this community as a whole to vote pro-resource.

            By default this can just be .5, but changing it could represent
            changing policy.

        base_spread_p : float
            Between 0 and 1, this represent a base prob.
            within a community where all of that communities neighbors are
            against resource development, i.e., all in state 0.
            This spread rate is then adjusted to be higher when some neighboring
            communities are in state 1. The rate of this adjustment is controlled by
            additional parameter outside_influence_k, such that a communtities adjusted
            spread at any given timestep is computed by,

            base_spread_p + ((1 - base_spread_p) * pos_influence * outside_influence_k)

            Where pos_influence is the weighted percentage of this communities neighbors which
            are in state 1.

            This adjusted spread rate is used within each community to define the
            chance that any given node switches states at any timestep - which depends on
            that node's neighbors states as well as this influence rate.

        outside_influence_k : float
            Between 0 and 1, simply a scaler on how much a communities neighbors should
            influence spreading within that community, where 1 means it should influence it
            a lot, and 0 means that the base spread rate will always be used, totally
            ignoring neighbors.

        vote_every : int
            This controls an optional lag in when voting for resources occurs.
            When set to 1, resources are voted on at the end of every timestep,
            but this can be set higher, e.g., to 5, such that resources are voted on
            every 5 timesteps.

        vote_threshold : float
            Between 0 and 1, this represents the percent of neighbor community nodes
            to a resource node are needed in order to develop that resource.
            Where there must be greater than that porportion of neighbors who
            "vote" yes, for the resource to be developed.

        interest_groups : list
            A list of the interest groups that will be influencing the citizens.
            This holds the pro development and anti development groups. Each
            group is represented by a dict with the folowing params:
            - 'pro_development' : (bool) if the group is pro or anti development
            - 'resources' : (int) the number of citizens a group can influence per round
            - 'strategy' : 

        seed : int or None
            The random state seed in which this simulation should be run.
            Used to generate the initial networks, and for different transitions.
        '''

        # Simulation random seed
        self.com_threshold = com_threshold
        self.base_spread_p = base_spread_p
        self.outside_influence_k = outside_influence_k
        self.vote_every = vote_every
        self.vote_threshold = vote_threshold

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        # Keep track of timesteps
        self.timestep = 0

        # Keep track of model history for each timestep
        # resources_state_H will store the percent of the resource nodes that
        # are in the developed state.
        self.resource_state_H = []

        # citizen_state_H will store the percent of citizens that are in favor
        # of development totaled over all comunities
        self.citizen_state_H = []

        # comunity_state_H will store the percent of comunities that are in foavor
        # of development
        self.comunity_state_H = []

        # Init the outer network
        self.generate_G_outer(**outer_G_args)

        # Remove isolated nodes + set resource and community nodes
        # In G_outer, then proc chance resources are already developed
        self.proc_G_outer(already_developed_p)

        # Generate the community network, e.g., the weighted
        # connected between different community nodes
        self.generate_G_com(dist_scale)

        # Generate graphs for each community
        self.generate_communites(**comm_G_args)

        # Set random initial nodes as pro dev,
        # Then update the state of that community within
        # G_outer if needed
        self.proc_communities(init_pro_dev_p)

        # save initial state of model to history
        self.save_history()

    def generate_G_outer(self, n_resources=10, n_communities=20,
                         comm_to_resource_p=.3):

        # Generate random bi-partite graph
        self.G_outer = random_bi_graph(n=n_resources,
                                       m=n_communities,
                                       p=comm_to_resource_p,
                                       seed=self.seed)

    def _set_node_types(self):

        # Save which nodes are which
        self.resource_nodes, self.community_nodes = [], []
        for x, y in self.G_outer.nodes(data=True):
            if y['bipartite'] == 0:
                self.resource_nodes.append(x)
            else:
                self.community_nodes.append(x)
            
    def proc_G_outer(self, already_developed_p):

        # Set initial state of all resources + communities to 0
        nx.set_node_attributes(self.G_outer, values=0, name='state')

        # Check for any disconnected nodes, either communties or resources
        disconnected = list(nx.isolates(self.G_outer))
        print('Removing for disconnected:', len(disconnected))
        self.G_outer.remove_nodes_from(disconnected)

        # Save which nodes are which
        self._set_node_types()

        # Sort lists
        self.resource_nodes.sort()
        self.community_nodes.sort()

        # Set if resources are already developed, w/ p
        for node in self.resource_nodes:
            if self.random_state.random() < already_developed_p:
                self.G_outer.nodes[node]['state'] = 1

    def generate_G_com(self, dist_scale):

        # Compute the shortest path between all community nodes in the
        # bipartite projection of the graph
        G_outer_proj = nx.projected_graph(B=self.G_outer, nodes=self.community_nodes)
        distances = dict(nx.all_pairs_shortest_path_length(G_outer_proj))

        # Init new graph to hold weighted connections
        # between community nodes
        self.G_com = nx.Graph()

        # For each combination of community nodes
        for n1, n2 in itertools.combinations(self.community_nodes, 2):

            # Random uniform prob
            p = self.random_state.random()

            # Compute scale by distance,
            # dist_scale of 0, removes scale,'
            # higher values give higher weight to closer nodes
            try:
                scale = distances[n1][n2] ** dist_scale

                # Scale the random uniform prob
                weight = p / scale

                # Add weighted edge
                self.G_com.add_edge(n1, n2, weight=weight)
            
            # If the two nodes are disconnected in dif networks - dont add edge for now
            except KeyError:
                
                # Make sure nodes are added though
                self.G_com.add_node(n1)
                self.G_com.add_node(n2)

    def generate_communites(self, n=100, k=3, p=.5):

        # Store communities as a dict of networks
        self.communities = {}

        for com in self.community_nodes:

            # Generate a unique random seed for each communities network
            # but still based off the global random seed passed
            seed = self.random_state.randint(0, 100000)

            # Generate a random small world network
            G = nx.watts_strogatz_graph(n=n,
                                        k=k,
                                        p=p,
                                        seed=seed)

            # Add to communities dict
            self.communities[com] = G

    def proc_communities(self, init_pro_dev_p):

        for com in self.communities:
            G = self.communities[com]

            # Set initial state of nodes to 0
            nx.set_node_attributes(G, values=0, name='state')

            # With prob, change each node to state 1
            for node in G.nodes:
                if self.random_state.random() < init_pro_dev_p:
                    G.nodes[node]['state'] = 1

            # Once set, check the community state in G_outer,
            # which will set the communities state in G_outer if over
            # the threshold
            self.check_community_state(com)

    def check_community_state(self, com):

        # If in fixed state, make sure state is 1, then return
        if self._is_fixed_state(com):
            self.G_outer.nodes[com]['state'] = 1
            return

        # Otherwise, base of current porportions
        G = self.communities[com]

        # Calculate current proportion of pos to neg nodes
        pos_nodes = sum([n[1]['state'] for n in G.nodes(data=True)])
        total_nodes = len(G.nodes)

        # Check if majority of nodes are positive - if so change state to pos
        if (pos_nodes / total_nodes) > self.com_threshold:
            self.G_outer.nodes[com]['state'] = 1

        # Otherwise set to negative
        else:
            self.G_outer.nodes[com]['state'] = 0


    def _is_fixed_state(self, com):

        # Check to see if this community is next to a developed resource,
        # if any neighboring resource is developed, return True, else False
        for resource in self.G_outer.neighbors(com):
            if self.G_outer.nodes[resource]['state'] == 1:
                return True

        return False

    def run_timestep(self):

        self.timestep += 1

        # Don't bother updating fixed state nodes
        coms_to_update = [node for node in self.community_nodes
                           if not self._is_fixed_state(node)]

        # Timesteps are run where each community is updated
        # based on the state of the graph at the last time-point
        # So go through first and set each communities updated
        # community spread prob.
        for com in coms_to_update:
            self.update_community_spread_ps(com)

        # Then, update each community
        for com in coms_to_update:
            self.update_community(com)

        # Check if time to vote on if resources get developed
        if self.timestep % self.vote_every == 0:
            self.check_votes()

        # Will want to likely keep track of some history
        self.save_history()

    def update_community_spread_ps(self, com):

        # Each communities spread rate is based on the influence from
        # it's neighboring nodes

        # Compute the 'positive influence' as the weighted porportion of
        # neighbors that are pos

        if self.G_com.degree(com) > 0:

            total_weight = self.G_com.degree(com, weight='weight')
            edges = self.G_com.edges(com, data=True)
            pos_weight = sum([e[2]['weight'] for e in edges if self.G_outer.nodes[e[1]]['state'] == 1])
            pos_influence = pos_weight / total_weight

            # Calculate the adjusted spread for this community at this timestep
            additional_spread = (1 - self.base_spread_p) * pos_influence * self.outside_influence_k
            adjusted_spread = self.base_spread_p + additional_spread

            # Set as attr in the communities graph
            self.communities[com].spread = adjusted_spread

        # If a community has no neighbors, the spread is fixed
        else:
            self.communities[com].spread = self.base_spread_p

    def update_community(self, com):

        # Every node in this community is updated based on the
        # state of the communities network
        # at the last timestep - so make copy
        G = self.communities[com]
        G_c = G.copy()

        for node in G_c.nodes:

            # Calculate the percent of positive /neg. neighbors - from the copy of the graph
            # So considering last timestep
            total = G_c.degree(node)
            pos_neighbors = sum([G_c.nodes[n]['state'] for n in G_c.neighbors(node)])
            pos_weight = pos_neighbors / total

            # If node's current state is 0, calc prob to change to state 1
            if G.nodes[node]['state'] == 0:
                spread_chance = pos_weight * G.spread
          
                # With random chance, change state
                if self.random_state.random() < spread_chance:
                    G.nodes[node]['state'] = 1

        # After the round of updates has been done,
        # check to see if the state of the community changes
        self.check_community_state(com)

    def check_votes(self):

        # Base on a copy of the outer graph
        G_outer_c = self.G_outer.copy()

        # Check every resource's neighbors
        for resource in G_outer_c.nodes:

            # If already developed, skip
            if G_outer_c.nodes[resource]['state'] == 1:
                continue

            # Calc percent of positive neighbors
            total = G_outer_c.degree(resource)
            pos_neighbors = sum([G_outer_c.nodes[n]['state']
                                 for n in G_outer_c.neighbors(resource)])
            pos_percent = pos_neighbors / total

            # If over the vote threshold, change resource state to developed
            if pos_percent > self.vote_threshold:
                self.G_outer.nodes[resource]['state'] = 1

                # Also make sure all neighboring communities are switched to 1
                for n in self.G_outer.neighbors(resource):
                     self.G_outer.nodes[n]['state'] = 1

    def save_history(self):
        
        # resource_state_H saves the percentage of resources developed at each timestep
        num_resources_developed = sum([self.G_outer.nodes[resource]['state']
                                        for resource in self.resource_nodes])
        num_total_resources = len(self.resource_nodes)
        percent_resources_developed = (float(num_resources_developed)/num_total_resources) * 100
        self.resource_state_H.append(percent_resources_developed)

        # comunity_state_H saves the percentage of comunities in favor of developing
        # at each timestep
        num_developing_comunities = sum([self.G_outer.nodes[comunnity]['state']
                                         for comunnity in self.community_nodes])
        num_total_comunities = len(self.community_nodes)
        percent_developing_comunities = (float(num_developing_comunities)/num_total_comunities) * 100
        self.comunity_state_H.append(percent_developing_comunities)

        # citizen_state_H saves the percentage of citizens in all comunities in favor
        # of developing
        num_developing_citizens, num_citizens = 0, 0

        for G in self.communities.values():
            num_developing_citizens += sum([n[1]['state'] for n in G.nodes(data=True)])
            num_citizens += len(G.nodes)

        percent_developing_citizens = (float(num_developing_citizens)/num_citizens) * 100
        self.citizen_state_H.append(round(percent_developing_citizens, 2))

    def get_history(self):
        history = {
            'resources' : self.resource_state_H,
            'comunities' : self.comunity_state_H,
            'citizens' : self.citizen_state_H
        }
        return history

class BiChainSimulation(BaseResourceSimulation):
    ''' Same as base, but with different outer_G

    Parameters
    ------------
    
    outer_G_args : dict
            These parameters are passed a dictionary, and they represent
            the parameters for creating a bi-partite network based on a bi-partite chain
            with the following params:

            - 'n_resources' : The number of resource nodes.

            - 'n_communities' : The number of community nodes.

            - 'p_extra' : The number of extra possible connections added to the original bi-partite structure,
                with 0 as the default of None.

            The way it works is the common overlap of n_resources and n_communities will form a bi-partite chain,
            so in an example with 20 communities and 10 resource nodes, there will be 10 resource and 10 community nodes connected in a chain,
            e.g., 1 to 2, 2 to 3, 3 to 4,... where every other is a community and every other is a node.
            
            The way overlapping nodes are handled, e.g., there are 10 more communities then resources,
            is that they will be added randomly with 1 edge to an existing resource node that is part of the original chain
            If there were more resources then communities, it would happen vice versa.

            The last piece is p_extra, by default if this is 0, then only the edges described before will be added, but
            if it is say .5, then 50% of the remaining possible valid bi-partite edges will be randomly filled in
            or if .1, then .1 of the remaining edges will be filled in. 
            This gives a tune-able parameter to make the network more or less one big bi-partite chain

    '''

    def generate_G_outer(self, n_resources=10, n_communities=20, p_extra=0):

        self.G_outer = nx.Graph()
    
        # This is the base length of the chain
        chain_length = min([n_resources, n_communities])

        # Add resource nodes and community nodes up to chain length
        i = 0
        for _ in range(chain_length):
            self.G_outer.add_node(i, bipartite=0)
            self.G_outer.add_node(i+1, bipartite=1)
            i += 2

        # Add edges
        for i in range(chain_length * 2):
            self.G_outer.add_edge(i, i+1)
            
        # Add extra either resource or community nodes
        i = chain_length * 2
        
        if n_resources > chain_length:
            for _ in range(n_resources - chain_length):
                i += 1
                self.G_outer.add_node(i, bipartite=0)
                
                # Connect to random community in original chain
                choice = self.random_state.choice(range(chain_length))
                choice = (choice * 2) + 1
                self.G_outer.add_edge(i, choice)

        if n_communities > chain_length:
            for _ in range(n_communities - chain_length):
                i += 1
                self.G_outer.add_node(i, bipartite=1)
                
                # Connect to random resource in original chain
                choice = self.random_state.choice(range(chain_length))
                choice = (choice * 2)
                self.G_outer.add_edge(i, choice)
        
        # Set self.resource nodes and community nodes
        self._set_node_types()
        
        # Add extra random connections
        if p_extra > 0:
            
            current_edges = len(self.G_outer.edges())
            max_edges = n_resources * n_communities
            available_edges = max_edges - current_edges
            to_add = np.round(p_extra * available_edges)
            
            cnt = 0
            while cnt < to_add:
                r = self.random_state.choice(self.resource_nodes)
                c = self.random_state.choice(self.community_nodes)

                if (r, c) not in self.outer_G.edges():
                    self.outer_G.add_edge(r, c)
                    cnt += 1