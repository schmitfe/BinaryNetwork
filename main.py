from ClusteredEI_network import *
import BinaryNetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    p = np.ones((2, 2)) * 0.2
    g = 1.2
    Q = 10
    # jep=2.0
    Rj = 0.75  # 0.85

    pep = 6.0  # 1275
    jep = 6.0  # 1.0



    neuron_parameters = {"N_E": 200, "N_I": 50,
                         "threshold_E": 1.0, "threshold_I": 1.0,
                        "tau_theta_E": 50000., "theta_q_E": 10.0,
                         "tau_theta_I": 50000., "theta_q_I": 0.0,
                         }

    #model without SFA, even if theta_q is unequal to zero
    #network = ProbClusteredEI_Network(Q, p, g, pep, Rj, neuron_parameters=neuron_parameters, neuron_model=BinaryNetwork.BinaryNeuronPopulation)
    network = WeightClusteredEI_Network(Q, p, g, jep, Rj, neuron_parameters=neuron_parameters)

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
    plt.show()