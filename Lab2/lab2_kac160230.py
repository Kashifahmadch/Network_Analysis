import zen
import sys
sys.path.append('../zend3js/')
import d3js
from numpy import *
from time import sleep

def square_grid(n,d3,G,x0=100,y0=100,w=50):
	if G is None:
		G = zen.Graph()
	# find the dimensions for the grid that are as close as possible
	num_rows = int(floor(sqrt(n)))
	while n % num_rows != 0:
		num_rows += 1
	num_cols = n/num_rows
	
	# Add all the nodes
	for i in range(n): 
		G.add_node(i)
		
	# Add the edges and position the nodes
	for i in range(num_rows):
		for j in range(num_cols):
			nidx = num_cols*i + j
			d3.position_node_(nidx,x0+i*w,y0+j*w)
			if i < num_rows-1:
				G.add_edge(nidx,nidx+num_cols) # add edge down
			if j < num_cols-1:
				G.add_edge(nidx,nidx+1) # add edge right		

def propagate(G,d3,x,steps,slp=0.5,keep_highlights=False,update_at_end=False):
	interactive = d3.interactive
	d3.set_interactive(False)
	A = G.matrix().T  # adjacency matrix of the network G
	d3.highlight_nodes_(list(where(x>0)[0]))
	d3.update()
	sleep(slp)
	cum_highlighted = sign(x)
	for i in range(steps): # the brains
		x = sign(dot(A,x)) # the brains
		cum_highlighted = sign(cum_highlighted+x)
		if not update_at_end:
			if not keep_highlights:
				d3.clear_highlights()
			d3.highlight_nodes_(list(where(x>0)[0]))
			d3.update()
			sleep(slp)
		#print A.matrix()
	if update_at_end:
		if not keep_highlights:
			d3.clear_highlights()
			d3.highlight_nodes_(list(where(x>0)[0]))
		else:
			d3.highlight_nodes_(list(where(cum_highlighted>0)[0]))
		d3.update()
	d3.set_interactive(interactive)
	if keep_highlights:
		return cum_highlighted
	else:
		return x
	#print A.matrix()
	

# Set up visualizer
G = zen.Graph()
d3 = d3js.D3jsRenderer(G, interactive=False, autolaunch=False)

# ===============================================
# GRID NETWORK
square_grid(144,d3,G,x0=75,y0=75)
d3.update()

x = zeros(G.num_nodes)
x[0] = 1
propagate(G,d3,x,10,slp=1)
sleep(3)

# ===============================================
# DIRECTED NETWORK
d3.clear()
G = zen.io.edgelist.read('lab2.edgelist', directed=True)
A = G.matrix()
B = A
for i in range(10):
	B = A.dot(B)
	print B
d3.set_graph(G)
d3.update()

x = zeros(G.num_nodes)
x[0] = 1
 #code to propagate
propagate(G,d3,x,10,slp=1)
sleep(3)
 #code to find the out-component of node 1
x = zeros(G.num_nodes)
x[1] = 1
propagate(G,d3,x,10,slp=1,keep_highlights=True)
# ===============================================
# E COLI NETWORK
d3.clear()
G = zen.io.edgelist.read('ecoli.edgelist', directed=True)
d3.set_interactive(False)
d3.set_graph(G)
d3.update()
print 'Ecoli has %i nodes.' % G.num_nodes
sleep(2)

# code to find the out-component of node 2
x = zeros(G.num_nodes)
x[2] = 1
t = propagate(G,d3,x,418,slp=1,update_at_end=True)
r = where(t>0)[0]
print 'out component of node 2'
print r
# code to find the out-component of node 16
d3.clear_highlights()
d3.update()
x = zeros(G.num_nodes)
x[16] = 1
g = propagate(G,d3,x,418,slp=1,update_at_end=True)
r = where(g>0)[0]
print 'out component of node 16'
print r
sleep(1)

d = zen.diameter(G)
print 'Ecoli has a diameter of %i.' % d
D,P = zen.algorithms.shortest_path.all_pairs_dijkstra_path_(G)
uidx,vidx = where(D==d)
path = zen.algorithms.shortest_path.pred2path_(uidx[0],vidx[0],P)
print 'the nodes to the shortest path are:'
print path

# code to visualize the path
T = zeros(len(path)-1)
for h in range(len(path)-1):
	T[h] = G.edge_idx_(path[h],path[h+1])
print 'the edges to higlight are, the edges are not being highlighted. I have found the edge inx and inputed them into the highlight funtion'
print T
d3.clear_highlights()
d3.update()
#H=[258,367,170,183]
d3.highlight_edges_(list(T))
sleep(3)
d3.stop_server()

G6 = zen.io.gml.read('pert.gml',weight_fxn = lambda x: x['weight'])

print '#Nodes: %i, #Edges: %i' % (G6.num_nodes,G6.num_edges)
print G6.nodes_()
D,P = zen.algorithms.bellman_ford_path_(G6,0)
print 'The critical activity length is ' 
print D
print 'The sequence is'
print P

T = zen.algorithms.min_cut_(G6, 0, 15, capacity='weight')
print 'min cut set weight'
print T


R = zen.algorithms.min_cut_set_(G6, 0, 15, capacity='weight')
print 'min cut set'
print R


