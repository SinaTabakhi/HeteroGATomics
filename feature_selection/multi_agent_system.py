from sys import float_info
import random as rand

import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from raw_data_loader import MultiOmicsDataset
from feature_selection.agent import Agent


class MultiAgentSystem:
    r"""The multi-agent system

    Args:
        max_workers (int): Maximum number of worker processes.
        dataset (MultiOmicsDataset): The multimodal dataset.
        max_iters (int): Maximum number of iterations.
        num_agents (int): Number of agents used in each modality.
        num_feats (int): Number of features to be selected by each agent.
        q0 (float): The constant parameter in the state transition rule.
        node_discount_rate (float): Node pheromone evaporation rate.
        edge_discount_rate (float): Edge pheromone evaporation rate.
        prob_discount_rate (float): Probability evaporation rate.
    """
    def __init__(
            self,
            max_workers: int,
            dataset: MultiOmicsDataset,
            max_iters: int,
            num_agents: int,
            num_feats: int,
            q0: float,
            node_discount_rate: float,
            edge_discount_rate: float,
            prob_discount_rate: float
    ) -> None:
        self._max_workers = max_workers
        self.dataset = dataset
        self.max_iters = max_iters
        self.num_agents = num_agents
        self.num_feats = num_feats
        self.q0 = q0
        self.node_discount_rate = node_discount_rate
        self.edge_discount_rate = edge_discount_rate
        self.prob_discount_rate = prob_discount_rate

        # total number of agents used in all modalities
        self.total_agents = self.num_agents * self.dataset.num_omics

        # count the number of times that a specific feature is selected by agents in which
        # keys indicate the indices of the modalities and
        # values show dictionaries that count number of times that a specific feature is selected in a specific modality
        self.feat_counter = {}

        # count the number of times that a specific edge is visited by agents in which
        # keys indicate the edge (tuple of indices) of the modalities and
        # values count the number of times that a specific edge is visited
        self.edge_counter = {}
        self.total_selected_edges = 0

        # count the number of correlation values between each pair of features in a feature subset
        # this value will be used in computing the agent's fitness
        self._num_corr = sum(range(self.num_feats))

        # initialize agents
        self.agents = []
        for agent_idx in range(self.total_agents):
            agent = Agent(self.num_feats)
            self.agents.append(agent)

        # best local agent (selected based on the agent's fitness)
        self._best_local_feat_subset = {}
        self._best_local_edge_subset = {}
        self._best_local_fit_val = 0

        # best global agent (selected based on the agent's fitness)
        self._best_global_feat_subset = {}
        self._best_global_edge_subset = {}
        self._best_global_fit_val = 0

    @property
    def best_feat_subset(self):
        return self._best_global_feat_subset

    def _reset_counters(self) -> None:
        self.feat_counter = {}
        self.edge_counter = {}
        self.total_selected_edges = 0

    def _update_feature_counter(self, feat_idx, modality_idx):
        modality_value = self.feat_counter.get(modality_idx, {})
        modality_value[feat_idx] = modality_value.get(feat_idx, 0) + 1
        self.feat_counter[modality_idx] = modality_value

    def _update_edge_counter(self, feat_idx_1, modality_idx_1, feat_idx_2, modality_idx_2):
        # only consider edge counter update for intra transitions (ignore edges for inter transitions)
        if modality_idx_1 == modality_idx_2:
            feat_idx_1, feat_idx_2 = (feat_idx_1, feat_idx_2) if feat_idx_1 < feat_idx_2 else (feat_idx_2, feat_idx_1)
            modality_value = self.edge_counter.get(modality_idx_1, {})
            modality_value[(feat_idx_1, feat_idx_2)] = modality_value.get((feat_idx_1, feat_idx_2), 0) + 1
            self.edge_counter[modality_idx_1] = modality_value
            self.total_selected_edges += 1

    def _rebuild_agents(self):
        for agent_idx in range(self.total_agents):
            self.agents[agent_idx].reset_parameters()

    def _get_modality_probabilities(self):
        modality_prob = []
        for modality_idx in range(self.dataset.num_omics):
            modality_prob.append(self.dataset.get(modality_idx).prob_val)
        return modality_prob

    def _set_modality_probabilities(self, modality_probs):
        for modality_idx in range(self.dataset.num_omics):
            self.dataset.get(modality_idx).prob_val = modality_probs[modality_idx]

    def _set_start_nodes(self):
        agent_counter = 0
        for modality_idx in range(self.dataset.num_omics):
            # generate initial nodes of the agent (unique elements chosen from the list of features)
            initial_features = rand.sample(range(self.dataset.get(modality_idx).num_features), self.num_agents)

            for feat_idx in initial_features:
                rel_val = self.dataset.get(modality_idx).get_relevance(feat_idx)
                self.agents[agent_counter].add_next_feat(feat_idx, modality_idx, rel_val)
                agent_counter += 1
                self._update_feature_counter(feat_idx, modality_idx)

    def _select_by_greedy_rule(self, agent_idx):
        agent = self.agents[agent_idx]
        current_modality = self.dataset.get(agent.last_modality)
        selected_feats = agent.feat_set.get(agent.last_modality, [])

        max_value = -float_info.max
        max_index = -1
        for feat_idx in range(current_modality.num_features):
            if feat_idx not in selected_feats and current_modality.is_connected(feat_idx, agent.last_feat):
                avg_corr = 0
                count_corr = 0
                next_feat_val = current_modality.train_data.iloc[:, feat_idx]
                for prev_modality_idx in agent.feat_set.keys():
                    prev_modality = self.dataset.get(prev_modality_idx)
                    for pre_feat_idx in agent.feat_set.get(prev_modality_idx):
                        corr_val = next_feat_val.corr(prev_modality.train_data.iloc[:, pre_feat_idx], method='pearson')
                        avg_corr += abs(corr_val)
                        count_corr += 1
                avg_corr /= count_corr
                edge_val = current_modality.get_edge_pheromone(agent.last_feat, feat_idx)
                result = current_modality.get_relevance(feat_idx) + current_modality.get_node_pheromone(feat_idx) + edge_val - \
                         avg_corr

                if result > max_value:
                    max_value = result
                    max_index = feat_idx

        return agent.last_modality, max_index

    def select_by_probability_rule(self, agent_idx, modality_probs):
        # select the next modality based on the probability distribution
        next_modality_idx = rand.choices(range(self.dataset.num_omics), weights=modality_probs, k=1)[0]
        agent = self.agents[agent_idx]
        next_modality = self.dataset.get(next_modality_idx)

        current_selected_feats = agent.feat_set.get(next_modality_idx, [])
        total_feat_set = np.arange(next_modality.num_features)
        candidate_feats = np.setdiff1d(total_feat_set, current_selected_feats)

        if next_modality_idx == agent.last_modality:
            connected_feats = next_modality.get_connected_feats(agent.last_feat)
            candidate_feats = np.intersect1d(candidate_feats, connected_feats)

        feat_probs = []
        for feat_idx in candidate_feats:
            avg_corr = 0
            count_corr = 0
            next_feat_val = next_modality.train_data.iloc[:, feat_idx]
            for prev_modality_idx in agent.feat_set.keys():
                prev_modality = self.dataset.get(prev_modality_idx)
                for pre_feat_idx in agent.feat_set.get(prev_modality_idx):
                    corr_val = next_feat_val.corr(prev_modality.train_data.iloc[:, pre_feat_idx], method='pearson')
                    avg_corr += abs(corr_val)
                    count_corr += 1

            avg_corr /= count_corr

            feat_rel = next_modality.get_relevance(feat_idx)
            feat_pheromone = next_modality.get_node_pheromone(feat_idx)
            result = feat_rel + feat_pheromone - avg_corr
            feat_probs.append(result)

        # apply softmax to convert values to probabilities
        exp_scores = np.exp(feat_probs)
        feat_probs = exp_scores / np.sum(exp_scores)

        # select the next feature based on its probability value
        next_feat_idx = rand.choices(candidate_feats, weights=feat_probs, k=1)[0]
        return next_modality_idx, next_feat_idx

    def _apply_state_transition(self, agent_idx, modality_probs):
        q = np.random.rand()

        # selection between greedy or probability rule
        if q <= self.q0:
            next_modality_idx, next_feat_idx = self._select_by_greedy_rule(agent_idx)
        else:
            next_modality_idx, next_feat_idx = self.select_by_probability_rule(agent_idx, modality_probs)

        # compute the correlation values of the new selected feature with previous selected features by agent
        agent = self.agents[agent_idx]
        rel_value = self.dataset.get(next_modality_idx).get_relevance(next_feat_idx)
        next_feat = self.dataset.get(next_modality_idx).train_data.iloc[:, next_feat_idx]
        sum_corr = 0
        for modality_idx in agent.feat_set.keys():
            current_modality = self.dataset.get(modality_idx)
            for feat_idx in agent.feat_set.get(modality_idx):
                corr_val = current_modality.train_data.iloc[:, feat_idx].corr(next_feat, method='pearson')
                sum_corr += abs(corr_val)

        return next_modality_idx, next_feat_idx, sum_corr, rel_value

    def _apply_state_transition_parallel(self, args):
        agent_idx, modality_probs = args
        next_modality, next_feat, sum_corr, rel_value = self._apply_state_transition(agent_idx, modality_probs)
        return agent_idx, next_modality, next_feat, sum_corr, rel_value

    def _evaluate_feat_subset(self):
        for agent_idx in range(self.total_agents):
            self.agents[agent_idx].eval_feat_subset(self._num_corr, self.dataset)

    def _update_best_selected_subset(self):
        self._best_local_fit_val = -float_info.max

        # update the best local solution
        for agent_idx in range(self.total_agents):
            if self._best_local_fit_val <= self.agents[agent_idx].fit_val:
                self._best_local_fit_val = self.agents[agent_idx].fit_val
                self._best_local_feat_subset = self.agents[agent_idx].feat_set
                self._best_local_edge_subset = self.agents[agent_idx].edge_set

        # update the best global solution
        if self._best_global_fit_val <= self._best_local_fit_val:
            self._best_global_fit_val = self._best_local_fit_val
            self._best_global_feat_subset = self._best_local_feat_subset
            self._best_global_edge_subset = self._best_local_edge_subset

    def _update_pheromone_values(self):
        total_selected_feats = self.total_agents * self.num_feats

        for modality_idx in range(self.dataset.num_omics):
            modality_value = self.dataset.get(modality_idx)
            modality_feat_value = self.feat_counter.get(modality_idx, {})
            modality_edge_value = self.edge_counter.get(modality_idx, {})
            best_modality_feat_value = self._best_local_feat_subset.get(modality_idx, [])
            best_modality_edge_value = self._best_local_edge_subset.get(modality_idx, [])

            # update node pheromone values
            for feat_idx in range(modality_value.num_features):
                second_term = modality_feat_value.get(feat_idx, 0) / total_selected_feats

                # add additional quantity to the features belonging to the best subset
                # 3 means three factors (relevance, correlation, accuracy)
                if feat_idx in best_modality_feat_value:
                    second_term += self._best_local_fit_val

                second_term *= self.node_discount_rate
                first_term = (1 - self.node_discount_rate) * modality_value.get_node_pheromone(feat_idx)
                modality_value.set_node_pheromone(feat_idx, first_term + second_term)

            # update edge pheromone values
            for feat_idx_1 in range(modality_value.num_features - 1):
                for feat_idx_2 in range(feat_idx_1 + 1, modality_value.num_features):
                    second_term = modality_edge_value.get((feat_idx_1, feat_idx_2), 0) / self.total_selected_edges

                    # add additional quantity to the edges belonging to the best subset
                    # 3 means three factors (relevance, correlation, accuracy)
                    if (feat_idx_1, feat_idx_2) in best_modality_edge_value:
                        second_term += self._best_local_fit_val

                    second_term *= self.edge_discount_rate
                    first_term = (1 - self.edge_discount_rate) * modality_value.get_edge_pheromone(feat_idx_1, feat_idx_2)
                    modality_value.set_edge_pheromone(feat_idx_1, feat_idx_2, first_term + second_term)
                    modality_value.set_edge_pheromone(feat_idx_2, feat_idx_1, first_term + second_term)

    def _update_modality_probs(self, modality_probs):
        total_selected_feats = self.total_agents * self.num_feats
        for modality_idx in range(self.dataset.num_omics):
            modality_idx_value = self.feat_counter.get(modality_idx, {})
            numerator_value = sum(modality_idx_value.values())
            new_prob = self.prob_discount_rate * (numerator_value / total_selected_feats)
            modality_probs[modality_idx] = (1 - self.prob_discount_rate) * modality_probs[modality_idx] + new_prob

        sum_probs = sum(modality_probs)
        modality_probs = list(map(lambda x: x/sum_probs, modality_probs))
        self._set_modality_probabilities(modality_probs)

    def run(self) -> None:
        for iter_idx in range(self.max_iters):
            print(f"    ------------------------------- Iteration {iter_idx + 1} -------------------------------")
            self._reset_counters()
            self._rebuild_agents()
            print("                 ---------- Current selected feature 1 --------------------- ")
            self._set_start_nodes()
            modality_probs = self._get_modality_probabilities()

            for feat_idx in range(self.num_feats - 1):
                print(f"                 ---------- Current selected feature {feat_idx + 2} --------------------- ")
                agent_list = [(agent_idx, modality_probs) for agent_idx in range(self.total_agents)]

                with Pool(max_workers=self._max_workers) as pool:
                    agent_results = pool.map(self._apply_state_transition_parallel, agent_list)

                for agent_idx, next_modality, next_feat, sum_corr, rel_value in agent_results:
                    self._update_feature_counter(next_feat, next_modality)
                    self._update_edge_counter(self.agents[agent_idx].last_feat, self.agents[agent_idx].last_modality,
                                              next_feat, next_modality)
                    self.agents[agent_idx].add_next_feat(next_feat, next_modality, rel_value, sum_corr)

            self._evaluate_feat_subset()
            self._update_best_selected_subset()
            self._update_pheromone_values()
            self._update_modality_probs(modality_probs)

    def __str__(self):
        return f"""
    Multi-agent system configurations
        Number of iterations:           {self.max_iters}
        Number of selected features:    {self.num_feats}     
        q0:                             {self.q0}
        Node discount rate:             {self.node_discount_rate}
        Edge discount rate:             {self.edge_discount_rate}
        Omics importance discount rate: {self.prob_discount_rate}
        Number of agents per omics:     {self.num_agents}
        Total number of agents:         {self.total_agents}
"""
