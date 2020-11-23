import networkx as nx
import numpy as np
import itertools
from networkx.algorithms.bipartite.generators import random_graph as random_bi_graph

class BaseResourceSimulation():

    def __init__(self, outer_G_args, already_developed_p,
                 dist_scale, comm_G_args, init_pro_dev_p,
                 com_threshold=.5, base_spread_p=.5,
                 outside_influence_k=.75, vote_every=1,
                 vote_threshold=.5,
                 seed=None):
        ''' Main class for running resource / voting simulation

        Parameters
        ------------
        outer_G_args : dict
            These parameters are passed a dictionary, and they represent
            the parameters for creating a random bi-partite network,
            with the following params:
            
            - 'n_resources' : 

            - 'n_communities' : 

            - 'comm_to_resource_p' : 

        already_developed_p : float
            Between 0 and 1, this represents the starting percent of
            resource nodes which should be set to developed at timestep 0.
            E.g., if 0, then none will be set, if 1, then all will be set to
            developed.


        vote_threshold : float
            Between 0 and 1, this represents the percent of neighbor community nodes
            to a resource node are needed in order to develop that resource.
            Where there must be greater than that porportion of neighbors who
            "vote" yes, for the resource to be developed.

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

    def generate_G_outer(self, n_resources=10, n_communities=20,
                         comm_to_resource_p=.3):

        # Generate random bi-partite graph
        self.G_outer = random_bi_graph(n=n_resources,
                                       m=n_communities,
                                       p=comm_to_resource_p,
                                       seed=self.seed)
        
    def proc_G_outer(self, already_developed_p):

        # Set initial state of all resources + communities to 0
        nx.set_node_attributes(self.G_outer, values=0, name='state')

        # Check for any disconnected nodes, either communties or resources
        disconnected = list(nx.isolates(self.G_outer))
        self.G_outer.remove_nodes_from(disconnected)

        # Save which nodes are which
        self.resource_nodes, self.community_nodes = [], []
        for x, y in self.G_outer.nodes(data=True):
            if y['bipartite'] == 0:
                self.resource_nodes.append(x)
            else:
                self.community_nodes.append(x)

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
            scale = distances[n1][n2] ** dist_scale
            
            # Scale the random uniform prob
            weight = p / scale
            
            # Add weighted edge
            self.G_com.add_edge(n1, n2, weight=weight)

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

    def _is_fixed_state(self, com):

        # Check to see if this community is next to a developed resource,
        # if any neighboring resource is developed, return True, else False
        for resource in self.G_outer.neighbors(com):
            if self.G_outer.nodes[resource] == 1:
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
        total_weight = self.G_com.degree(com, weight='weight')
        edges = self.G_com.edges(com, data=True)
        pos_weight = sum([e[2]['weight'] for e in edges if self.G_outer.nodes[e[1]]['state'] == 1])
        pos_influence = pos_weight / total_weight

        # Calculate the adjusted spread for this community at this timestep
        additional_spread = (1 - self.base_spread_p) * pos_influence * self.outside_influence_k
        adjusted_spread = self.base_spread_p + additional_spread

        # Set as attr in the communities graph
        self.communities[com].spread = adjusted_spread

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
            neg_weight = 1 - pos_weight

            # If node's current state is 0, calc prob to change to state 1
            if G.nodes[node]['state'] == 0:
                spread_chance = neg_weight * G.spread
  
            # Otherwise, node's state is 1, calc prob to change to state 0
            else:
                spread_chance = pos_weight * G.spread
            
            # With random chance, change state
            if self.random_state.random() < spread_chance:
                G.nodes[node]['state'] = (G.nodes[node]['state'] + 1) % 2

        # After the round of updates has been done,
        # check to see if the state of the community changes
        self.check_community_state(com)

    def check_votes(self):

        # Check every resource's neighbors
        for resource in self.G_outer.nodes:

            # If already developed, skip
            if self.G_outer.nodes[resource]['state'] == 1:
                continue

            # Calc percent of positive neighbors
            total = self.G_outer.degree(resource)
            pos_neighbors = sum([self.G_outer.nodes[n]['state'] 
                                 for n in self.G_outer.neighbors(resource)])
            pos_percent = pos_neighbors / total

            # If over the vote threshold, change resource state to developed
            if pos_percent > self.vote_threshold:
                self.G_outer.nodes[resource]['state'] = 1

                # Also make sure all neighboring communities are switched to 1
                for n in self.G_outer.neighbors(resource):
                     self.G_outer.nodes[n]['state'] = 1

    def save_history(self):
        pass


# Right now these are the parameters relevant for generating
# a random bi-partite graph
outer_G_args = {'n_resources' : 10,
                'n_communities' : 20,
                'comm_to_resource_p' : .3}

# Right now these are the parameters for generating
# a random small world network
comm_G_args = {'n': 100,
               'k': 3,
               'p': .5}

test = BaseResourceSimulation(outer_G_args=outer_G_args,
                              already_developed_p=0,
                              dist_scale=1,
                              comm_G_args=comm_G_args,
                              com_threshold=.5,
                              init_pro_dev_p=.55,
                              base_spread_p=.5,
                              outside_influence_k=.75,
                              vote_every=1,
                              seed=5) 
test.run_timestep()