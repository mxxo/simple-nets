# neural net program
# adapted from A.K. Dewdney's New Turing Omnibus
# 2017 Max Orok

# train a simple neural net to convert from rectangular coordinates
# to polar coordinates

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def compute_target(coord_kind, coords):
    # using the formula from: http://mathworld.wolfram.com/DiskPointPicking.html
    # coords = [radius, angle]
    # x-coord
    if coord_kind == 0:
        target = coords[0]*math.cos(coords[1])
    #y-coord
    else:
        target = coords[0]*math.sin(coords[1])

    return target

def create_synapses(num_ports, med_neurons):
    # create the synapse list
    synapselist = []
    # initialize values to random numbers btwn 0,1
    for i in range(0, med_neurons):
        synapselist.append([])
        for j in range(0, num_ports):
            synapselist[i].append(0.1*random.random())

    return synapselist

def estimate_coords(polar_input, synone ,syntwo):

    # get the number of medial neurons,
    # input neurons, and output neurons
    num_median = len(synone)
    num_in = len(synone[0])
    num_out = len(syntwo[0])

    med_input = [0 for x in range(0,num_median)] # input to the medial neurons
    med_out = [0 for x in range(0,num_median)]

    # the first layer of synapse calculation
    for i in range(0,num_median):
        med_input[i] = 0
        med_out[i] = 0
        for j in range(0,num_in):
            med_input[i] = med_input[i] + synone[i][j] * polar_input[j]
        med_out[i] = math.tanh(med_input[i])

    #assign a list for the output variables
    output = [0 for x in range(0, num_out)]
    errors = []
    # the second layer of synapse calculation
    for i in range(0, num_out):
        output[i] = 0
        for j in range(0, num_median):
            output[i] = output[i] + syntwo[j][i] * med_out[j]
        # call the coordinate calculation method
        target = compute_target(i, polar_input)
        this_error = target - output[i]
        errors.append(this_error)

    # clip the lists to prevent overflow
    med_input = np.clip(med_input, -1 , 1)
    med_out = np.clip(med_out, -1, 1)

    back_prop(num_in, num_out, synone, syntwo, med_input, med_out, polar_input, errors)

    # return the average error
    avg_error = math.sqrt(errors[0]**2 + errors[1]**2)
    return avg_error

def back_prop(num_in, num_out, synone, syntwo, medin, medout, p_input, errors):
    # adjust the synapse weighting in response to the error
    # set the rate circa 0.1 (pg. 244 Dewdney)
    rate = 0.01
    med_neurons = len(syntwo)

    # adjust the second synaptic layer
    for i in range(0, num_out):
        for j in range(0, med_neurons):
            syntwo[j][i] = syntwo[j][i] + rate*medout[j]*errors[i]

    # derive the signmoidal signal
    sigma = [0 for x in range(0, med_neurons)]
    sigmoid = [0 for x in range(0, med_neurons)]

    for i in range(0, med_neurons):
        sigma[i] = 0
        for j in range(0, num_out):
            sigma[i] = sigma[i] + errors[j] * syntwo[i][j]
        sigmoid[i] = 1 - np.power(medout[i], 2)
        if abs(medin[i]) > 1:
            print i
            print sigmoid[i]
            print medin[i]

    #adjust the first synaptic layer
    for i in range(0, num_in):
        for j in range(0, med_neurons):
            delta = rate*sigmoid[j]*sigma[j]*p_input[i]
            synone[j][i] = synone[j][i] + delta
    return

###############################################################################
# plot the neural net
in_neurons = 3  # (num inputs + one bias neuron) = (x, y + one bias neuron)
out_neurons = 2 # (num outputs)

med_neurons = 30 # can play around with this, will change your results a lot!

error_list = [] # will be list of list of floats

#initialize synapse layers
synone = create_synapses(in_neurons,med_neurons)
syntwo = create_synapses(out_neurons,med_neurons)

#loop over neural net learning attempts
for i in range(1, 100001):
    if i % 10000 == 0:
        print "iteration number: " + str(i)
    #1 pick a random point
    radius = random.random() # pick a random point on the unit disc
    angle = random.random() * 2*math.pi # radians
    # below is an addition from:
    # https://github.com/tomstuart/neural-network/blob/gh-pages/neural_network.js
    polar_input = [radius, angle, 2]
    #2 call the coordinate conversion function - mutating synone, syntwo
    #3 (within) back propagate error correction - mutating synone, syntwo
    avg_error = estimate_coords(polar_input, synone, syntwo)
    #4 append the error to the error_list
    error_list.append([i, avg_error])

#plot the results
x , y = zip(*error_list)
_ , ax = plt.subplots()
plt.scatter(x,y)
ax.set_xlabel("iteration")
ax.set_ylabel("normalized error")
plt.show()

#debug
    #back propagate to adjust synapse weights
    #print "synone"
    #print synone
    #print "syntwo"
    #print syntwo
    #print "med_input"
    #print med_input
    #print "med_out"
    #print med_out
    #print "polar_input"
    #print polar_input
    #print "errors"
    #print errors
