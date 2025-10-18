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
        print(f"WARN: The mode {mode} is not implemented yet in call_algo.")
    return policy

    # if mode == 0:
    #     algo_name = algo_name.lower()
    #     assert algo_name in ['sac', 'darc', 'vgdf', 'sac_iw', 'par']
    #     # online online setting
    #     from online_online.darc import DARC
    #     from online_online.sac import SAC
    #     from online_online.vgdf import VGDF
    #     from online_online.sac_iw import SAC_IW
    #     from online_online.par import PAR

    #     algo_to_call = {
    #         'sac': SAC,
    #         'darc': DARC,
    #         'vgdf': VGDF,
    #         'sac_iw': SAC_IW,
    #         'par': PAR,
    #     }

    #     algo = algo_to_call[algo_name]
    #     policy = algo(config, device)

    # elif mode == 1:
    #     algo_name = algo_name.lower()
    #     assert algo_name in ['cql_sac', 'bc_vgdf', 'bc_sac', 'h2o', 'mcq_sac', 'rlpd', 'bc_par','flow', 'sac_tar', 'pure_sac', 'pretrain_cql_sac', 'pretrain_flow', 'pretrain_h2o', 'vflow', 'sac']
    #     # offline online setting
    #     from offline_online.cql_sac import CQLSAC
    #     from offline_online.bc_vgdf import BCVGDF
    #     from offline_online.bc_sac import BCSAC
    #     from offline_online.mcq_sac import MCQSAC
    #     from offline_online.h2o import H2O
    #     from offline_online.rlpd import RLPD
    #     from offline_online.bc_par import BCPAR
    #     from offline_online.bc_flow import FlowPolicy
    #     from offline_online.sac_tar import SACTAR
    #     from offline_online.pure_sac import PureSAC
    #     from offline_online.pretrain_cql_sac import PretrainCQLSAC
    #     from offline_online.pretrain_flow import PretrainFlowPolicy
    #     from offline_online.vflow import VFlowPolicy
    #     from offline_online.sac import SAC
    #     algo_to_call = {
    #         'cql_sac': CQLSAC,
    #         'bc_vgdf': BCVGDF,
    #         'bc_sac': BCSAC,
    #         'mcq_sac': MCQSAC,
    #         'h2o': H2O,
    #         'rlpd': RLPD,
    #         'bc_par': BCPAR,
    #         'flow': FlowPolicy,
    #         'sac_tar': SACTAR,
    #         'pure_sac': PureSAC,
    #         'pretrain_cql_sac': PretrainCQLSAC,
    #         'pretrain_flow': PretrainFlowPolicy,
    #         'pretrain_h2o': H2O,
    #         'vflow': VFlowPolicy,
    #         'sac': SAC
    #     }

    #     algo = algo_to_call[algo_name]
    #     policy = algo(config, device)
    # elif mode == 2:
    #     algo_name = algo_name.lower()
    #     assert algo_name in ['sac_bc', 'h2o', 'sac_cql', 'sac_mcq', 'par_bc']
    #     # online offline setting
    #     from online_offline.sac_bc import SACBC
    #     from online_offline.h2o import H2O
    #     from online_offline.sac_cql import SACCQL
    #     from online_offline.sac_mcq import SACMCQ
    #     from online_offline.par_bc import PARBC

    #     algo_to_call = {
    #         'sac_bc': SACBC,
    #         'h2o': H2O,
    #         'sac_cql': SACCQL,
    #         'sac_mcq': SACMCQ,
    #         'par_bc': PARBC,
    #     }
    #     algo = algo_to_call[algo_name]
    #     policy = algo(config, device)

    # elif mode == 3:
    #     algo_name = algo_name.lower()
    #     assert algo_name in ['dara', 'bosa', 'iql', 'td3_bc', 'igdf']
    #     # offline offline setting
    #     from offline_offline.dara import DARA
    #     from offline_offline.bosa import BOSA
    #     from offline_offline.iql import IQL
    #     from offline_offline.td3_bc import TD3BC
    #     from offline_offline.igdf import IGDF

    #     algo_to_call = {
    #         'dara': DARA,
    #         'bosa': BOSA,
    #         'iql': IQL,
    #         'td3_bc': TD3BC,
    #         'igdf': IGDF,
    #     }

    #     algo = algo_to_call[algo_name]
    #     policy = algo(config, device)
    
    # return policy