# import all algorithms this benchmark implement
import wandb
def call_algo(algo_name, config, mode, device):
    if mode == 1:
        algo_name = algo_name.lower()
        assert algo_name in [ 'bc_vgdf', 'bc_sac', 'h2o', 'bc_par','vflow']
        # offline online setting
        from offline_online.bc_vgdf import BCVGDF
        from offline_online.bc_sac import BCSAC
        from offline_online.h2o import H2O
        from offline_online.bc_par import BCPAR
        from offline_online.vflow import VFlowPolicy
        algo_to_call = {
            'bc_vgdf': BCVGDF,
            'bc_sac': BCSAC,
            'h2o': H2O,
            'bc_par': BCPAR,
            'vflow': VFlowPolicy
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)
    else:
        raise NotImplementedError(f"The mode '{mode}' is not implemented yet in call_algo().")
    return policy