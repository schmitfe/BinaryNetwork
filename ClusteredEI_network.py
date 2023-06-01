import BinaryNetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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



if __name__ == "__main__":
    # Q = 5
    # N_E = 100
    # N_I = 100
    # p_plus = np.array([[0.5, 0.2], [0.2, 0.2]])
    # p_minus = np.array([[0.25, 0.1], [0.1, 0.1]])
    # j_plus = np.array([[1.1, -1.2], [1.2, -1.0]])
    # j_minus = np.array([[0.5, -3.0], [1.0, -1.0]])
    # threshold_E = 0.5
    # threshold_I = 0.5
    # network = ClusteredEI_network(Q, N_E, N_I, p_plus, p_minus, j_plus, j_minus, threshold_E, threshold_I)

    # parameters of Rost
    threshold_E = 1.0
    threshold_I = 1.0
    #p=0.5*np.array([[0.2, 0.2], [0.2, 0.2]])
    #p = np.array([[0.2, 0.5], [0.5, 0.5]])
    p=np.ones((2,2))*0.1
    g=1.2
    Q=20
    #jep=2.0
    Rj=0.75#0.85
    N_E=200#40#200
    N_I=50#10#50
    # network = WeightClusteredEI_Network(Q, N_E, N_I, p, g, jep, Rj, threshold_E, threshold_I)
    pep=6.0#1275
    network = ProbClusteredEI_Network(Q, N_E, N_I, p, g, pep, Rj, threshold_E, threshold_I)


    network.initialize()
    #print(network.state)
    #print(network.weights)
    plt.imshow(network.weights)
    plt.show()

    # run x steps of the network
    steps= 100000
    recording = np.zeros((network.N, steps))
    # use tqdm to show progress bar
    for i in tqdm(range(steps)):
        network.run(2)
        recording[:, i] = network.state
    # plot the recording of the network with flipped y axis

    plt.imshow(recording, interpolation=None, aspect='auto', origin='lower')
    #set colormap to black and white
    plt.set_cmap('binary')
    plt.ylabel("NeuronID")
    plt.xlabel("Time [a.u.]")
    # set title to contain pep and Rj
    plt.title("pep = " + str(pep) + ", Rj = " + str(Rj))
    # set aspect ratio to auto
    #plt.gca().set_aspect('auto')
    plt.show()
