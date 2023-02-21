import unittest
from My_DeepRMSA_Agent import *
from My_Deep_RMSA_A3C import get_link_map


class MyTestCase(unittest.TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    def test_something(self):
        linkmap = get_link_map()
        agent = DeepRMSAAgent(
                    0, trainer, linkmap, LINK_NUM, NODE_NUM, SLOT_TOTAL, k_path, M, lambda_req,
                    lambda_time,
                    len_lambda_time, gamma, episode_size, batch_size, Src_Dst_Pair, Candidate_Paths,
                    num_src_dst_pair, model_path, global_episodes, regu_scalar, x_dim_p, x_dim_v, n_actions,
                    num_layers, layer_size, model2_flag, nonuniform, prob_arr,
                    configfile=configfile, results_path='results', maxDR=1200
                )
        first_demand = agent.load_test_demands()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
