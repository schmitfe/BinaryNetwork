# Created by Felix J. Schmitt on 05/30/2023.
# Class for binary networks (state is 0 or 1)

import numpy as np
from numba import jit

class NetworkElement:
    def __init__(self, reference, name="Some Network Element"):
        self.name = name
        self.reference = reference
        self.view = None
    def set_view(self, view):
        self.view = view
    def initialze(self):
        pass

class Neuron(NetworkElement):
    def __init__(self, reference, N=1, name="Some Neuron"):
        super().__init__(reference, name)
        self.N = N
        self.state = None
    def update(self):
        pass
    def set_view(self, view):
        self.state = self.reference.state[view[0]:view[1]]
        self.view = view
    def initialze(self):
        #self.reference
        self.reference.state[self.view[0]:self.view[1]] = np.random.choice([0, 1], size=self.N)

class BinaryNeuronPopulation(Neuron):
    def __init__(self, reference, N=1, threshold=1.0, name="Binary Neuron Population"):
        super().__init__(reference, N, name)
        self.threshold = threshold
    def update(self, weights, state, index=None):
        return np.heaviside(np.sum(weights * state) - self.threshold, 0)

class AdaptiveBinaryNeuronPopulation(Neuron):
    def __init__(self, reference, tau_theta=1, theta_q=0., N=1, threshold=1.0, name="Adaptive Neuron Population"):
        super().__init__(reference, N, name)
        self.threshold = threshold
        self.adaptation = np.zeros(N)
        self.last_update = np.zeros(N)
        self.tau_theta = tau_theta
        self.theta_q = theta_q
    def update(self, weights, state, index):
        self.adaptation[index] = self.threshold + (self.adaptation[index] - self.threshold) * np.exp(-(self.reference.sim_steps-self.last_update[index]) / self.tau_theta)
        value = np.heaviside(np.sum(weights * state) - (self.adaptation[index]), 0)
        if value == 1:
            self.adaptation[index] += self.theta_q
        self.last_update[index] = self.reference.sim_steps
        return value





class BackgroundActivity(Neuron):
    # Neuron which
    def __init__(self, reference, N=1, Activity=0.5, Stochastic=False, name="Background Activity"):
        super().__init__(reference, N, name)
        self.Activity = Activity
        if Stochastic:
            self.update = self.update_stochastic
        else:
            self.update = self.update_deterministic
    def update_stochastic(self, weights=None, state=None, Index=None):
        return np.random.choice([0, 1], 1) * self.update_deterministic(weights, state)
    def update_deterministic(self, weights=None, state=None, Index=None):
        # if activity is a float, set all neurons to this activity
        if isinstance(self.Activity, float):
            return self.Activity
        # if activity is a function, set neurons by this function
        elif callable(self.Activity):
            return self.Activity()
        else:
            return 1.0

    def initialze(self):
        self.state = np.array([self.update() for i in range(self.N)])

class Synapse(NetworkElement):
    def __init__(self, reference, pre, post, name="Some Synapse"):
        super().__init__(reference, name= post.name + " <- " + pre.name)
        self.pre = pre
        self.post = post
        # weights is a matrix of shape (pre.N, post.N)
        self.weights = None
    def set_view(self, view):

        self.weights = self.reference.weights[view[1,0]:view[1,1], view[0,0]:view[0,1]]
        self.view = view
    def initialze(self):
        self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] = np.random.rand(self.post.N, self.pre.N)

class PairwiseBernoulliSynapse(Synapse):
    def __init__(self, reference, pre, post, p=0.5, j=1.0):
        super().__init__(reference, pre, post )
        self.p = p
        self.j = j
    def initialze(self):
        # if p is greater 1, split into two synapses
        p = self.p
        n_iterations = 1
        while p > 1:
            p /= 2
            n_iterations += 1
        if n_iterations > 1:
            print("Warning: p > 1, splitting synapse into " + str(n_iterations) + " synapses")

        for i in range(n_iterations):
            self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] += \
                np.random.choice([0, self.j], size=(self.post.N, self.pre.N), p=[1-p, p])


class AllToAllSynapse(Synapse):
    def __init__(self, reference, pre, post, j=1.0):
        super().__init__(reference, pre, post)
        self.j = j
    def initialze(self):
        self.reference.weights[self.view[1, 0]:self.view[1, 1], self.view[0, 0]:self.view[0, 1]] = \
            np.ones((self.post.N, self.pre.N)) * self.j




class BinaryNetwork:
    def __init__(self, name="Some Binary Network"):
        self.name = name
        self.N = 0
        self.population = []
        self.synapses = []
        self.state = None
        self.weights = None
        self.LUT = None # look up table for the update function
        self.sim_steps = 0

    def add_population(self, population):
        self.population.append(population)
        self.N += population.N
        return population

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def initialize(self, autapse=False):
        self.state = np.zeros(self.N)
        self.weights = np.zeros((self.N, self.N))
        N_start = 0
        for idx, population in enumerate(self.population):
            population.set_view([N_start,N_start + population.N])
            N_start += population.N
            population.initialze()
        self.LUT= np.array([population.view for population in self.population])
        for synapse in self.synapses:
            synapse.set_view(np.array([[synapse.pre.view[0], synapse.pre.view[1]],[synapse.post.view[0], synapse.post.view[1]]]))
            synapse.initialze()
        self.sim_steps = 0
        # set diagonal to zero
        if not autapse:
            np.fill_diagonal(self.weights, 0)

    def update(self):
        # choose a random neuron and update it
        neuron = np.random.randint(self.N)
        # find the population to which the neuron belongs
        population_idx = np.where((self.LUT[:, 0] <= neuron) & (self.LUT[:, 1] > neuron))[0][0]

        # find the index of the neuron in the population
        neuronIDX = neuron - self.LUT[population_idx, 0]
        # update the neuron
        if self.state[neuron] == 0:
            self.state[neuron] = self.population[population_idx].update(self.weights[neuron, :],
                                                                    self.state, neuronIDX)
        else:
            self.state[neuron] = 0
        self.sim_steps += 1

    def run(self, steps=1000):
        for i in range(steps):
            self.update()







