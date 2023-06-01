from ClusteredEI_network import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    p=np.ones((2,2))*0.2
    g=1.2
    Q=10
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