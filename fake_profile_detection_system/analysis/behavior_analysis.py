import pandas as pd
import networkx as nx

class BehaviorAnalyzer:
    def __init__(self):
        pass

    def follower_ratio(self, followers, following):
        if following == 0:
            return 0
        return followers / following

    def posting_frequency(self, posts, account_age_days):
        if account_age_days == 0:
            return 0
        return posts / account_age_days

    def build_network_graph(self, edges):
        """
        Build a network graph from follower/following edges.
        edges: list of tuples (follower, following)
        """
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    def network_features(self, G, node):
        """
        Extract network features for a node.
        """
        degree = G.degree(node)
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        clustering = nx.clustering(G.to_undirected(), node)
        return {
            "degree": degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "clustering": clustering
        }
