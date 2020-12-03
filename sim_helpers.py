from BaseResourceSimulation import BaseResourceSimulation, BiChainSimulation
from joblib import Parallel, delayed
import numpy as np


def simulate(args, num_timesteps=50, repeat=1):
    
    args['seed'] = repeat
    
    # Run the model
    if args.pop('chain'):
        model = BiChainSimulation(**args)
    else:
        model = BaseResourceSimulation(**args)
    
    for i in range(num_timesteps):
        model.run_timestep()

    return np.array(model.get_history()['resources'])

def sim_repeated(args, num_timesteps=50):
   
    # Fixed
    n_repeats = 128

    resources = Parallel(n_jobs=16)(delayed(simulate)(args=args.copy(),
                                                      num_timesteps=num_timesteps,
                                                      repeat=i) for i in range(n_repeats))
    return np.mean(resources, axis=0)

def get_title(args):
    
    title = ''
    
    if args['chain']:
        title += 'Bi-Partite Chain Outer Network ('
    else:
        title += 'Random Outer Network ('
        
    title += 'n_r=' + str(args["outer_G_args"]['n_resources']) + ' '
    title += 'n_c=' + str(args["outer_G_args"]['n_communities']) + ')'

    return title

def get_base_args():
    
    args = {
    "outer_G_args" : {
    'n_resources' : 100,
    'n_communities' : 100,
    'comm_to_resource_p' : .05
    },
    
     "comm_G_args" : {
    'n': 100,
    'k': 3,
    'p': .2
    },
    
    "base_spread_p" : .1,
    "init_pro_dev_p" : .1,
    
    "dist_scale" : 2,
    "com_threshold" : .5,
    "vote_threshold" : .5,
    
    "outside_influence_k" : .15,
    "already_developed_p" : 0,
    "vote_every" : 1,
    }

    args['chain'] = False
    
    return args

def get_base_chain_args():

    # Start w/ same
    args = get_base_args()

    del args["outer_G_args"]['comm_to_resource_p']
    args["outer_G_args"]['p_extra'] = .001

    args['chain'] = True

    return args