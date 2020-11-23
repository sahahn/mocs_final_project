from BaseResourceSimulation import BaseResourceSimulation

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