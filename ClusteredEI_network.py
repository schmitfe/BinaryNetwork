import BinaryNetwork
import numpy as np

class ClusteredEI_network(BinaryNetwork.BinaryNetwork):
    def __init__(self, Q, N_E, N_I, p_plus, p_minus, j_plus, j_minus, threshold_E, threshold_I, name="Clustered EI Network"):
        super().__init__(name)
        self.Q = Q
        self.N_E = N_E
        self.N_I = N_I
        self.p_plus = p_plus # probability of connection between neurons in the same cluster [EE, EI, IE, II]
        self.p_minus = p_minus # probability of connection between neurons in different clusters [EE, EI, IE, II]
        self.j_plus = j_plus # synaptic strength between neurons in the same cluster [EE, EI, IE, II]
        self.j_minus = j_minus # synaptic strength between neurons in different clusters [EE, EI, IE, II]
        self.threshold_E = threshold_E
        self.threshold_I = threshold_I
        self.E_pops = []
        self.I_pops = []
        self.other_pops = []

    def create_populations(self):
        for i in range(self.Q):
            self.E_pops.append(self.add_population(BinaryNetwork.BinaryNeuronPopulation(self, N=self.N_E, threshold=self.threshold_E, name="E" + str(i))))
        for i in range(self.Q):
            self.I_pops.append(self.add_population(BinaryNetwork.BinaryNeuronPopulation(self, N=self.N_I, threshold=self.threshold_I, name="I" + str(i))))

    def create_synapses(self):
        # add synapses between populations
        Pops = [self.E_pops, self.I_pops]
        for j, pre_type in enumerate(Pops):
            for i, post_type in enumerate(Pops):
                for k, pre in enumerate(pre_type):
                    for l, post in enumerate(post_type):
                        if k == l:
                            self.add_synapse(BinaryNetwork.PairwiseBernoulliSynapse(self, pre, post, p=self.p_plus[i, j], j=self.j_plus[i, j]))
                        else:
                            self.add_synapse(BinaryNetwork.PairwiseBernoulliSynapse(self, pre, post, p=self.p_minus[i, j], j=self.j_minus[i, j]))

    def create_Background(self):
        self.other_pops.append(self.add_population(BinaryNetwork.BackgroundActivity(self, N=1, Activity=1.0, name="Background_E")))
        for i in range(self.Q):
            self.add_synapse(BinaryNetwork.AllToAllSynapse(self, self.other_pops[-1], self.E_pops[i], j=1.0))
            self.add_synapse(BinaryNetwork.AllToAllSynapse(self, self.other_pops[-1], self.I_pops[i], j=1.0))

    def initialize(self):
        self.create_populations()
        self.create_synapses()
        self.create_Background()
        super().initialize()


def calculateRBN_weights(g, p, N_E, N_I, threshold_E, threshold_I):
    """
    Calculate the weights for a random balanced network
    :param g: ratio of inhibitory to excitatory weights
    :param p: probability of connection between neurons  [EE, EI, IE, II]
    :param Q: number of clusters
    :param N_E: number of excitatory neurons per cluster
    :param N_I: number of inhibitory neurons per cluster
    :param threshold_E: threshold of excitatory neurons
    :param threshold_I: threshold of inhibitory neurons
    :return: weights
    """
    ne=N_E/(N_E+N_I)
    ni=N_I/(N_E+N_I)
    jee=threshold_E/(np.sqrt(p[0,0]*ne))
    jei=-g*jee*(p[0,0]*ne)/(p[1,0]*ni)
    jie=threshold_I/(np.sqrt(p[1,0]*ne))
    jii=-g*jie*(p[1,0]*ne)/(p[1,1]*ni)
    return np.array([[jee, jei], [jie, jii]])


class WeightClusteredEI_Network(ClusteredEI_network):
    def __init__(self, Q, N_E, N_I, p, g, jep, Rj, threshold_E, threshold_I, name="Clustered EI Network"):
        j=calculateRBN_weights(g, p, N_E, N_I, threshold_E, threshold_I)
        jem=(Q-jep)/(Q-1)
        jip=1+Rj*(jep-1)
        jim=(Q-jip)/(Q-1)
        j_plus=np.multiply(j, np.array([[jep, jip], [jip, jip]]))
        j_minus=np.multiply(j, np.array([[jem, jim], [jim, jim]]))
        super().__init__(Q, N_E, N_I, p, p, j_plus, j_minus, threshold_E, threshold_I, name)


class ProbClusteredEI_Network(ClusteredEI_network):
    def __init__(self, Q, N_E, N_I, p, g, pep, Rj, threshold_E, threshold_I, name="Clustered EI Network"):
        j=calculateRBN_weights(g, p, N_E, N_I, threshold_E, threshold_I)
        pem=(Q-pep)/(Q-1)
        pip=1+Rj*(pep-1)
        pim=(Q-pip)/(Q-1)

        p_plus=np.multiply(p, np.array([[pep, pip], [pip, pip]]))
        p_minus=np.multiply(p, np.array([[pem, pim], [pim, pim]]))
        print(p_plus)
        print(p_minus)
        super().__init__(Q, N_E, N_I, p_plus, p_minus, j, j, threshold_E, threshold_I, name)


