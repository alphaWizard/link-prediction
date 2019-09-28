from centralities import *
from similarities import *

network = create_graph() 
#join(network,"g")  # for joining the network individually
connect(network,"a","b")
connect(network,"d","b")
connect(network,"c","b")
# connect(network,"b","c")
# connect(network,"b","d")
## add these edges for further verification
connect(network,"c","d")
connect(network,"c","f")
connect(network,"d","e")
connect(network,"d","f")

cns = rooted_pagerank_score(network,'b')
# for a,b,c in cns:
#     print('{} , {} : {}'.format(a,b,c) )

print(cns)
# print(cns[0,5])
