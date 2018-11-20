# Kashif Ahmad kac160230
# 1

import zen
import numpy
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy import *

G = zen.io.edgelist.read('hobbit_LotR_characters.edgelist',weighted=True)
#G = zen.io.edgelist.read('hobbit_LotR_characters.edgelist',weighted=True)
A = G.matrix()
N = G.num_nodes

# prints the top five (num) nodes according to the centrality vector v
# v takes the form: v[nidx] is the centrality of node with index nidx
def print_top(G,v, num=5):
	idx_list = [(i,v[i]) for i in range(len(v))]
	idx_list = sorted(idx_list, key = lambda x: x[1], reverse=True)
	for i in range(min(num,len(idx_list))):
		nidx, score = idx_list[i]
		print '  %i. %s (%1.4f)' % (i+1,G.node_object(nidx),score)
		#print '  %i. %s' % (i+1,G.node_object(idx))

# returns the index of the maximum of the array
# if two or more indices have the same max value, the first index is returned
def index_of_max(v):
	return numpy.where(v == max(v))[0]
	
print '\n============================================='
# Degree Centrality
print '\nDegree Centrality:'

print 'The degree centrality matrix is  :' 
R = G.matrix()
R = R.transpose()
L = zeros(G.num_nodes)
for x in range(G.num_nodes):
	temp = 0
	for y in range(G.num_nodes):
		temp = R[x][y]+temp
	L[x] = temp

print L
print_top(G,L)

print '\n============================================='
# Eigenvector Centrality
print '\nEigenvector Centrality (by Zen):'
temp2 = zen.algorithms.centrality.eigenvector_centrality_(G,weighted=True)
print_top(G,temp2)

print '\nEigenvector Centrality (by linear algebra):'
k, v = la.eig(A)
r = numpy.abs(k)
k1_idx = index_of_max(k) # find the index of the largest eigenvalue
# finish printing the top 5 eigenvector centrality characters by linear algebra
#print_top(G,numpy.abs(v[k1_idx]))
g = v[:,k1_idx]
g2 = numpy.abs(g)
print_top(G,g2)

#print v
#print numpy.abs(v)

noi = 60
print '\nConfirming that eigenvector centrality is a steady-state of sorts for node %i (%s):' % (noi,G.node_object(noi))
# compare the eigenvector centrality of node 57 to the sum of the centralities of its neighbors (divided by the largest eigenvalue)
temp3 = G.neighbors_(noi)
sum1 = 0
for f in range(len(temp3)):
	sum1 = temp2[temp3[f]]+sum1

print 'lobelias centrality is' 
print temp2[noi]

print 'sum of lobeliases neighbors centrality is'
print sum1

print 'normalized sum of lobelias neigbors is'
f = sum1/r[k1_idx]
print f

print '\nShowing the convergece of eigenvector centrality...'
num_steps = 10
x = numpy.zeros((G.num_nodes,)) # initial centrality vector
x[76] = 1
cs = numpy.zeros((G.num_nodes,num_steps))
for i in range(num_steps):
	x = x/la.norm(x) # at each step we need to normalize the centrality vector
	for j in range(G.num_nodes):
		cs[j,i] = numpy.dot( x , v[:,j] ) # project x onto each of the eigenvectors
	x = numpy.dot(A,x) # "pass" the centrality one step forward

plt.figure() # this creates a figure to plot in
for i in range(G.num_nodes): # for each eigenvector plot the projection of x onto it over the steps
	if i == k1_idx:
		plt.plot(range(num_steps),abs(cs[i,:]),label='Projection onto v1') # only label the eigenvector v1
	else:
		plt.plot(range(num_steps),abs(cs[i,:]))
plt.ylim([-0.2,1.1]) # this sets the limits for the y axis
plt.legend(loc='best') # this attaches a legend
plt.title('Projection of the centrality vector x onto the eigenvectors of A') # this adds a title
plt.show() # this makes the figure appear

def kats(alpha) : 
#	alpha = 0.99
	ones = numpy.ones((G.num_nodes,1))
	iden =numpy.eye(G.num_nodes)
	temp1 = la.inv(iden - (alpha*R))
	ex = numpy.dot(temp1,ones)
	print_top(G,ex)


print '\n============================================='
# Katz Centrality
print '\nKatz Centrality:'

print 'katz for alpha = 0'
kats(0)

print 'katz for alpha = 0.2'
kats(0.2)

print 'katz for alpha = 0.4'
kats(0.4)

print 'katz for alpha = 0.6'
kats(0.6)

print 'katz for alpha = 0.8'
kats(0.8)


print '\n============================================='
# PageRank

def page_rank(alpha) : 
#	alpha = 0.99

	D = numpy.zeros((G.num_nodes,G.num_nodes))
	for f in range(G.num_nodes):
		D[f][f] = max(L[f],1)	
	ones = numpy.ones((G.num_nodes,1))
	iden =numpy.eye(G.num_nodes)
	D_inv  = la.inv(D)
	R_temp = numpy.dot(R,D_inv)
	temp1 = la.inv(iden - (alpha*R_temp))
	rex = numpy.dot(temp1,ones)
	print_top(G,rex)
		

print '\nPageRank'
print 'pagerank for alpha = 0.8'
page_rank(.8)	
print 'pagerank for alpha = 0.6'
page_rank(0.6)

print 'pagerank for alpha = 1'
page_rank(1)

print '\n============================================='
# Betweenness Centrality
print '\nBetweenness Centrality'
temp_2 = zen.algorithms.centrality.betweenness_centrality_(G,weighted=True)
print_top(G,temp_2)
