# recherche des composantes connexes
# adapté d'une réponse de jimifiki (2012) posté sur stakoverflow

def find_root(Node, Root):
    '''
    trouver la racine d'un sommet dans un graphe sans cycles
    et la distance entre le sommet et la racine
    '''
    while Node != Root[Node][0]:
        Node = Root[Node][0]
    return(Node, Root[Node][1])

def get_roots(Neighbors):
    '''
    trouver les racines et les composantes connexes dans un graphe
    '''
    Roots = {}

    for Node in Neighbors.keys():
        Roots[Node] = (Node, 0)

    for Node1 in Neighbors: 
        for Node2 in Neighbors[Node1]:
            (Root_Node1, Depth_Node1) = find_root(Node1, Roots) 
            (Root_Node2, Depth_Node2) = find_root(Node2, Roots) 
            if Root_Node1 != Root_Node2: 
                Min = Root_Node1
                Max = Root_Node2 
                if  Depth_Node1 > Depth_Node2: 
                    Min = Root_Node2
                    Max = Root_Node1
                Roots[Max] = (Max, max(Roots[Min][1] + 1, Roots[Max][1]))
                Roots[Min] = (Roots[Max][0], -1) 

    # initialisation
    Family_Nodes = {}
    Family_Edges = {}
    for Node1 in Neighbors: 
        if Roots[Node1][0] == Node1:
            Family_Nodes[Node1] = []
            Family_Edges[Node1] = set()

    # remplissage
    for Node1 in Neighbors: 
        Family_Nodes[find_root(Node1, Roots)[0]].append(Node1)
        for Node2 in Neighbors[Node1]:
            Family_Edges[find_root(Node1, Roots)[0]].add((Node1, Node2)) 

    return(Family_Nodes, Family_Edges)
