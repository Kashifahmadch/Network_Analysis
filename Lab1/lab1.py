#Kashif Ahmad kac160230
import zen
import numpy
G= zen.Graph()
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(2,4)
G.add_edge(4,3)
G.add_edge(3,6)
G.add_edge(3,5)
G.add_edge(1,5)
print '#Nodes: %i, #Edges: %i' % (G.num_nodes,G.num_edges)
print 'Has 3-4 edge : %s' %G.has_edge(3,4)
print 'Has 6-4 edge : %s' %G.has_edge(6,4)
print 'The adjacency matrix is :' 
print G.matrix()

A = zen.Graph()
for i in range(7,13): 
	A.add_node(i)
B=A.matrix()
print B==B.transpose()

G2= zen.Graph()
G2.add_edge(2,3,weight=2)
G2.add_edge(1,5,weight=3)
G2.add_edge(1,2)
G2.add_edge(2,4)
G2.add_edge(3,4)
G2.add_edge(3,5)
G2.add_edge(3,6)
G2.add_edge(6,6)

print 'The adjacency matrix for weighted graph is :' 
print G2.matrix()

G3= zen.DiGraph()
G3.add_edge(1,3)
G3.add_edge(4,1)
G3.add_edge(3,2)
G3.add_edge(2,6)
G3.add_edge(6,4)
G3.add_edge(4,5)
G3.add_edge(5,3)
G3.add_edge(6,5)

print 'The adjacency matrix for diagraph is :' 
print G3.matrix()

G4 = zen.io.gml.read('proofwikidefs_la.gml',weight_fxn = lambda x: x['weight'])

R = G4.nodes()

#print R[1]

T = G4.in_neighbors_(4)

#print T[1]

def cocitation(G4):
	cocit=zen.Graph()
		
	for x in range(G4.num_nodes):
		cocit.add_node(G4.node_object(x))
	for i in range(G4.num_nodes):
		for j in range(G4.num_nodes):
			temp1 = G4.in_neighbors_(i)
			temp2 = G4.in_neighbors_(j)
			r=1
			common = []
			for y in range(len(temp1)):
				for z in range(len(temp2)):
					if temp1[y]==temp2[z]:
						common.append(temp1[y])
				#		print temp1[y]						
						r=r+1
						#print r
			#print common
	#return G4_cocitation 
			temp = 0
			for l in common:
               	        	temp=G4.weight_(G4.edge_idx_(l,i))*G4.weight_(G4.edge_idx_(l,j))+temp
            		if temp>0:
				if j>=i:
					if j!=i:  # we dont want to incorporate common incomings of the same node	
                				cocit.add_edge_(i,j,weight=temp)
    	return cocit

G_cocitation=cocitation(G4) 

#print G_cocitation.matrix()

A=G4.matrix()

C1 = numpy.dot(A.transpose(),A)
C2 = G_cocitation.matrix()
Cdiff = C1-C2
print 'Difference between cocitation methods: %i' % Cdiff.sum().sum()

D1 = numpy.dot(A.transpose(),A)
D2 = G_cocitation.matrix()
for q in range(G4.num_nodes):
    D1[q,q]=0
Ddiff = D1-D2
print 'Difference between cocitation methods after the identified change is: %i' % Ddiff.sum().sum()

#print R[1]
temporary = G_cocitation.neighbors('Linear Combination')
#print G_cocitation.neighbors('Linear Combination')
for i in range(len(temporary)):
	print temporary[i]
	print G_cocitation.weight('Linear Combination',temporary[i])

temporary2 = G4.neighbors('Linear Combination')
for i in range(len(temporary2)):
	print temporary2[i]

def has_leaf(graph):
	w=0
	n = graph.num_nodes
	R = graph.nodes()
	for i in range(n):
		temp1 = graph.out_neighbors_(i)
		if len(temp1) == 0 :
			w=100
			return i
	if w==0:
		return -20

H = zen.io.edgelist.read('acyclic1.edgelist',directed=True)

def acyclic(G5):
    n=G5.num_nodes
    nodes=[]
    out=[]
    for node in G5.nodes():
        nodes.append(node)
        out.append(len(G5.out_neighbors(node)))
    while 0 in out:
  	  G5.rm_node(nodes[out.index(0)])
          out=[]
          nodes=[]
          for node in G5.nodes():
          	nodes.append(node)
          	out.append(len(G5.out_neighbors(node)))
   
    r=G5.num_nodes    
    if r==0:
       print 'acyclic network'
    else:
        print 'cyclic network'
  
print 'network 1 is'
acyclic(H)
H2 = zen.io.edgelist.read('acyclic2.edgelist',directed=True)
print 'network 2 is'
acyclic(H2)
H3 = zen.io.edgelist.read('acyclic3.edgelist',directed=True)
print 'network 3 is'
acyclic(H3)

#def nodes_num_neighbors(G5):
#	nodes=[]
#	outgoing=[]
#	for node in G5.nodes():
#		node.append(node)
#		outgoing.append(len(G5.out_neighbors(node))


G6 = zen.io.gml.read('2013-actor-movie-bipartite.gml')
def bipartite(G6):
    new=zen.Graph()
    actors=list(G6.U())
    for l in actors:
        new.add_node(l)
    length=new.num_nodes
   
    for i in range(length):
        for j in range(length):
		temp1=set(G6.neighbors(actors[i]))
		temp2=set(G6.neighbors(actors[j]))
        	if i != j:
			if j > i:	
                		if len(temp1.intersection(temp2)) != 0:
                   	     		new.add_edge(actors[i],actors[j], weight=len(temp1.intersection(temp2)))
   
    return new	
 
temp=bipartite(G6)

print 'Jason Statham: %s'%temp.neighbors('Jason Statham')
print 'WILL FERRELL: %s'% temp.neighbors('Will Ferrell')

print 'Zac Efron: %s'%temp.neighbors('Zac Efron')
print 'Clint Eastwood: %s'% temp.neighbors('Clint Eastwood')

