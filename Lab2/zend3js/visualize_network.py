import sys
import zen
import d3js


if __name__ == "__main__":
	try:
		fname = sys.argv[1]

		G = zen.io.edgelist.read(fname, ignore_duplicate_edges=True)

		d3 = d3js.D3jsRenderer(G, event_delay=0.03, interactive=False,
								node_dstyle=d3js.node_style(size=4), 
								node_hstyle=d3js.node_style(fill='#EB4343'),
								edge_hstyle=d3js.edge_style(stroke='#EB4343',stroke_width=5))

		d3.update()
		d3.stop_server()
	except:
		print 'use like: python visualize_network.py filename.edgelist'
