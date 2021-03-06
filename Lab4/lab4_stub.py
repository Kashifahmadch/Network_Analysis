#kashif Ahmad
#kac160230
import math
import numpy
import zen
import matplotlib.pyplot as plt
plt.ioff()
from numpy import *
import sys
sys.path.append('../zend3js/')
import d3js
from time import sleep
import random

## Generates a small world network
def small_world(n,q,p,G=None,d3=None,x0=300,y0=300):
	'''
	q must be even
	'''
	if G is None:
		G = zen.Graph()
	for i in range(n):
		G.add_node(i)
		if d3:
			x = 200*cos(2*pi*i/n) + x0
			y = 200*sin(2*pi*i/n) + y0
			d3.position_node_(i,x,y)
	# add the regular edges
	for uidx in range(n):
		for vidx in range(uidx+1,uidx+1+q/2):
			vidx = vidx % n
			G.add_edge(uidx,vidx)

	if d3:
		d3.update()
		sleep(3)
		d3.set_interactive(True)

	# add the random edges
	for uidx in range(n):
		for vidx in range(uidx+1,n):
			if not G.has_edge_(uidx,vidx):
				if random.random() <= p:
					G.add_edge_(uidx,vidx)

	return G

## Plots the degree distribution and calculates the power law coefficent
def calc_powerlaw(G,kmin):
	ddist = zen.degree.ddist(G,normalize=False)
	cdist = zen.degree.cddist(G,inverse=True)
	k = numpy.arange(len(ddist))

	#R = G.matrix()
	#R = R.transpose()
	#L = zeros(G.num_nodes)
	#for x in range(G.num_nodes):
	#	temp = 0
	#	for y in range(G.num_nodes):
	#		temp = R[x][y]+temp
	#	L[x] = temp
	
	#print L

	plt.figure(figsize=(8,12))
	plt.subplot(211)
	plt.bar(k,ddist, width=0.8, bottom=0, color='b')

	plt.subplot(212)
	plt.loglog(k,cdist)
	#alpha
	temp1 = k/(kmin-0.5)
	#print temp1
	var1 = 0
	for i in range(len(temp1)):
		if temp1[i] >= 1:
			var1 = log(temp1[i])+ var1
	#print var1
	alpha = 1 + (len(k)/var1)
	
	#alpha = 0 # calculate using Newman (8.6)!
	sigma = ((alpha-1)/(math.sqrt(len(k)))) # calculate using Newman (8.7)!
	print '%1.2f +/- %1.2f' % (alpha,sigma)

	plt.show()

## DIAMETER ===============================================
## Visualize a small world network
G = zen.Graph()
d3 = d3js.D3jsRenderer(G, event_delay=0.1, interactive=False, autolaunch=False)
G = small_world(20,4,0.1,G,d3)
d3.stop_server()

## Convergence of the small world effect
P = [0,0.0001,0.001,0.0025,0.005,0.01,0.05,0.1,0.2,0.4,0.6,0.8]
temp = 0
d = zeros(len(P))

for p in P:
	G = small_world(100,6,p) # don't pass G or d3 here or else it will get slow
	# calculate the diameter and store it for plotting below
	d[temp] = zen.diameter(G)
	temp = temp + 1
## Plot the Convergence
plt.figure(figsize=(8,15))
plt.plot(P,d) # change x and y to your "x" and "y" values
plt.hold(True)
plt.plot([0.0001,1],[(log10(100)),(log10(100))]) # plot the limit
plt.hold(False)
plt.xlabel('prob to make random edge')
plt.ylabel('diameter of small world network')
plt.show()

## POWER LAW ==============================================
G = zen.io.edgelist.read('japanese.edgelist',directed=True,ignore_duplicate_edges=True)
calc_powerlaw(G,2)  # need to change kmin appropriately
G = zen.io.edgelist.read('ca-HepTh.edgelist',directed=False,ignore_duplicate_edges=True)
calc_powerlaw(G,30) # need to change kmin appropriately
G = zen.io.edgelist.read('soc-Epinions1.edgelist',directed=True,ignore_duplicate_edges=True)
calc_powerlaw(G,600) # need to change kmin appropriately

## GIANT COMPONENT ========================================
# taking n to be 101 in every case, that simplifies the equation to c = 100p
c = numpy.arange(0,1,0.05)
p = c/100 
length_temp = zeros(101)
S= zeros(len(c))

for i in range(len(p)):
	G1 = zen.generating.erdos_renyi(101,p[i])
	comp = zen.algorithms.components(G1)
	for j in range(len(comp)):
		length_temp[j] = len(comp[j])
	S[i] = max(length_temp)
plt.figure(figsize=(8,15))
plt.plot(c,S,'b',linewidth=2)
plt.show()




def erdos(n):
	c = numpy.arange(0,1,0.05)
	p = c/n 
	length_temp = zeros(n+1)
	w, h = len(p), 100;
	Matrix = [[0 for x in range(w)] for y in range(h)] 
	S= zeros(len(c))
	for k in range(100):
		for i in range(len(p)):
			G1 = zen.generating.erdos_renyi(n+1,p[i])
			comp = zen.algorithms.components(G1)
			for j in range(len(comp)):
				length_temp[j] = len(comp[j])
			S[i] = max(length_temp)
			Matrix[k][i] = max(length_temp)
	print Matrix	
	for i in range(len(p)):
		for k in range(100):
			S[i] = Matrix[k][i] + S[i]
	S=S/100	
	print S
	plt.figure(figsize=(8,15))
	plt.plot(S,c,'b',linewidth=2)
	plt.show()

erdos(10)
erdos(100)
erdos(1000)

## DEGREE DISTRIBUTIONS OF RANDOM MODELS ==================

## FITTING RANDOM MODELS ==================================
def degree_sequence(G):
	return [degree for degree,freq in enumerate(zen.degree.ddist(G,normalize=False)) for f in range(int(freq))]

def configuration_model(degree_sequence,G=None):
	import numpy.random as numpyrandom
	if G is None:
		G = zen.Graph()

	n = len(degree_sequence)
	for i in range(n):
		G.add_node(i)

	# this is twice the number of edges, needs to be even
	assert mod(sum(degree_sequence),2) == 0, 'The number of edges needs to be even; the sum of degrees is not even.'
	num_edges = sum(degree_sequence)/2

	# the number of edges should be even
	assert mod(num_edges,2) == 0, 'The number of edges needs to be even.'

	stubs = [nidx for nidx,degree in enumerate(degree_sequence) for d in range(degree)]
	stub_pairs = numpyrandom.permutation(num_edges*2)

	self_edges = 0
	multi_edges = 0
	for i in range(num_edges):
		uidx = stubs[stub_pairs[2*i]]
		vidx = stubs[stub_pairs[2*i+1]]
		if uidx == vidx:
			self_edges += 1
		if G.has_edge_(uidx,vidx):
			eidx = G.edge_idx_(uidx,vidx)
			G.set_weight_(eidx, G.weight_(eidx)+1 )
			multi_edges += 1
		else:
			G.add_edge_(uidx,vidx)

	print 'self edges: %i,  multi-edges: %i' % (self_edges,multi_edges)

	return G

## FRIENDSHIP PARADOX =====================================
