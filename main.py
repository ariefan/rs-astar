#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import random
import plotly.express as px
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import csv
import math
import heapq

#######################
# Page configuration
st.set_page_config(
    page_title="Rute terdekat antar Rumah Sakit",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

coords = []
with open('rs_jogja.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row['y'] = float(row['y'])  # Convert y-coordinate to float
        row['x'] = float(row['x'])  # Convert x-coordinate to float
        coords.append(row)

#######################
## Sidebar
#with st.sidebar:
#    st.title('Rute terdekat antar Rumah Sakit')
#
#    dropdown_label = [coord['name'] + '. ' + coord['label'] for coord in coords]
#    dropdown_value = [coord['name'] for coord in coords]
#
#    selected_coord = st.selectbox('Rumah sakit asal', options=dropdown_label)
#    selected_source_value = dropdown_value[dropdown_label.index(selected_coord)]
#
#    selected_coord = st.selectbox('Rumah sakit tujuan', options=dropdown_label)
#    selected_target_value = dropdown_value[dropdown_label.index(selected_coord)]


#######################
# Dashboard Main Panel
col = st.columns((5, 3), gap='medium')

with col[0]:
    import osmnx as ox
    import matplotlib.pyplot as plt
    import networkx as nx
    
    st.title('Rute terdekat antar Rumah Sakit')

    dropdown_label = [coord['name'] + '. ' + coord['label'] for coord in coords]
    dropdown_value = [coord['name'] for coord in coords]

    col1, col2 = st.columns(2)
    with col1:
        selected_coord = st.selectbox('Rumah sakit asal', options=dropdown_label, index=dropdown_value.index('K'))
        selected_source_value = dropdown_value[dropdown_label.index(selected_coord)]
    with col2:
        selected_coord = st.selectbox('Rumah sakit tujuan', options=dropdown_label, index=dropdown_value.index('W'))
        selected_target_value = dropdown_value[dropdown_label.index(selected_coord)]

    # Define the place name
    place_name = "Kota Yogyakarta"

    # Retrieve the graph
    # graph = ox.graph_from_place(place_name, network_type='drive')
    graph = ox.load_graphml('street_yogyakarta.graphml')

    # Add new nodes and edges to the nearest existing nodes
    added_nodes = []
    i = 1
    for coord in coords:
        # Find the nearest existing node in the graph
        nearest_node, nearest_dist = ox.distance.nearest_nodes(graph, coord['x'], coord['y'], return_dist=True)

        # Create a new node with the given coordinates
        new_node_id = i # nearest_node + 1
        coord['id'] = new_node_id
        graph.add_node(new_node_id, x=coord['x'], y=coord['y'], street_count=1, name=coord['name'])

        # Add an edge between the new node and the nearest existing node
        graph.add_edge(new_node_id, nearest_node, length=nearest_dist)

        # Append the new node ID to the list of added nodes
        added_nodes.append(new_node_id)

        i+=1

    # Define Euclidean distance heuristic function
    def euclidean_distance(node1, node2):
        x1, y1 = { graph.nodes[node1]['x'], graph.nodes[node1]['y'] }
        x2, y2 = { graph.nodes[node2]['x'], graph.nodes[node2]['y'] }
        return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)

    # Define A* algorithm function
    def astar(graph, start, goal):
        open_list = [(0, start)]  # Priority queue, tuple: (f_score, node)
        closed_list = set()
        g_scores = {node: math.inf for node in graph.nodes()}
        g_scores[start] = 0
        parents = {}

        while open_list:
            _, current = heapq.heappop(open_list)
            closed_list.add(current)

            if current == goal:
                path = []
                while current in parents:
                    path.insert(0, current)
                    current = parents[current]
                path.insert(0, start)
                return path

            for neighbor in graph.neighbors(current):
                if neighbor in closed_list:
                    continue

                tentative_g_score = g_scores[current] + graph[current][neighbor][0]['length']
                if tentative_g_score < g_scores[neighbor]:
                    parents[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    # f_score = g_scores[neighbor] + heuristic_values[neighbor]
                    f_score = g_scores[neighbor] + euclidean_distance(neighbor, goal)
                    heapq.heappush(open_list, (round(f_score, 3), neighbor))

            # Print step
            # print(f"Node: {current}\nOpen List: {open_list}\nClosed List: {closed_list}\n")

        return None  # No path found 

    # Define your source and target nodes
    source = selected_source_value
    target = selected_target_value

    # Retrieve node IDs for source and target from your coordinates
    start_node = next((item for item in coords if item['name'] == source), None)['id']
    end_node = next((item for item in coords if item['name'] == target), None)['id']

    # Calculate the shortest path using A* algorithm
    graph = graph.to_undirected()
    # shortest_path = nx.astar_path(graph, start_node, end_node, weight='length')
    shortest_path = astar(graph, start_node, end_node)

    # Define node colors and sizes
    node_colors = ['r' if node in added_nodes else 'b' for node in graph.nodes()]
    node_sizes = [30 if node in added_nodes else 5 for node in graph.nodes()]

    # Plot the graph
    fig, ax = ox.plot_graph(graph, node_color=node_colors, node_size=node_sizes, edge_color='gray', bgcolor='w', figsize=(12, 12), close=False, show=False)

    # Annotate added node names
    for node_id in added_nodes:
        x, y = graph.nodes[node_id]['x'], graph.nodes[node_id]['y']
        name = graph.nodes[node_id]['name']
        ax.text(x - 0.0005, y, name, fontsize=12, fontweight='bold', color='brown', ha='right', va='center')

    # Plot the shortest path on the same figure
    ox.plot_graph_route(graph, shortest_path, route_color='green', node_size=3, ax=ax, node_color='k')

    st.pyplot(fig)
    
    st.write('Rute terpendek melewati node: ')
    st.write(shortest_path)


with col[1]:
    st.title('Contoh sederhana A* Algorithm')

    dd_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    dd_value = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    col1, col2 = st.columns(2)
    with col1:
        sel_source_label= st.selectbox('Node Asal', options=dd_label, index=dd_value.index('D'))
        sel_source_value = dd_value[dd_label.index(sel_source_label)]
    with col2:
        sel_target_label = st.selectbox('Node Tujuan', options=dd_label, index=dd_value.index('G'))
        sel_target_value = dd_value[dd_label.index(sel_target_label)]

    # Define Euclidean distance heuristic function
    def euclidean_distance(node1, node2):
        x1, y1 = nodes[node1]
        x2, y2 = nodes[node2]
        return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)

    # Define A* algorithm function
    txt = []
    def astar(graph, start, goal):
        open_list = [(0, start)]  # Priority queue, tuple: (f_score, node)
        closed_list = set()
        g_scores = {node: math.inf for node in graph.nodes()}
        g_scores[start] = 0
        parents = {}

        while open_list:
            _, current = heapq.heappop(open_list)
            closed_list.add(current)

            if current == goal:
                path = []
                while current in parents:
                    path.insert(0, current)
                    current = parents[current]
                path.insert(0, start)
                return path

            for neighbor in graph.neighbors(current):
                if neighbor in closed_list:
                    continue

                tentative_g_score = g_scores[current] + graph[current][neighbor]['weight']
                if tentative_g_score < g_scores[neighbor]:
                    parents[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_score = g_scores[neighbor] + heuristic_values[neighbor]
                    heapq.heappush(open_list, (round(f_score, 3), neighbor))

            # Print step
            txt.append(f"Node: {current}<br>Open List: {open_list}<br>Closed List: {closed_list}")

        return None  # No path found

    def draw_plot(sp):        
        pos = nx.kamada_kawai_layout(G)
        # Define node colors
        edge_colors = ['blue' if (u, v) in zip(sp, sp[1:]) or (v, u) in zip(sp, sp[1:]) else 'black' for u, v, d in G.edges(data=True)]
        edge_widths = [3 if (u, v) in zip(sp, sp[1:]) or (v, u) in zip(sp, sp[1:]) else 1 for u, v, d in G.edges(data=True)]
        node_colors = ['lightblue' if node != source and node != target else 'orange' for node in G.nodes()]

        nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_colors, edge_color=edge_colors, width=edge_widths, font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # Print heuristic values on the plot
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.1, f'h={heuristic_values[node]:.2f}', ha='center', fontsize=10, color='red')
            plt.text(x, y - 0.15, f'{nodes[node]}', ha='center', fontsize=10, color='brown')

        plt.title(f"A* Algorithm: Rute terpendek dari node {source} ke node {target}")
        # display plot
        st.pyplot(plt)

    # Define the graph
    G = nx.Graph()

    nodes = {'A': (0, 0), 'B': (1, 2), 'C': (3, 1), 'D': (2, -1), 'E': (-1, 5), 'F': (4, 2), 'G': (2, 3)}
    edges = [
        ('A', 'B', None),
        ('A', 'C', None),
        ('A', 'E', None),
        ('B', 'C', None),
        ('B', 'D', None),
        ('B', 'F', None),
        ('C', 'G', None),
        ('D', 'A', None),
        ('E', 'G', None),
        ('F', 'C', None),
    ]

    edges = [(u, v, euclidean_distance(u, v)) for u, v, weight in edges]

    # Add nodes and edges to the graph
    G.add_nodes_from(nodes.keys())
    G.add_weighted_edges_from(edges)
    G.to_undirected()

    source = sel_source_value
    target = sel_target_value

    # Calculate heuristic value for each node
    heuristic_values = {node: euclidean_distance(node, target) for node in nodes}

    # Run A* algorithm
    shortest_path = astar(G, source, target)

    draw_plot(shortest_path)

    for t in txt:
        st.write(t, unsafe_allow_html=True)
    st.write("Shortest Path:", shortest_path)

    iteration = 1    
    # st.markdown('#### Iterasi ke:')
    # # Add left and right buttons inline
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     if st.button('←'):
    #         iteration = max(iteration - 1, 0)
    # with col2:
    #     st.write(iteration)
    # with col3:
    #     if st.button('→'):
    #         iteration += 1

    # # Generate random data
    # random_data = pd.DataFrame({
    #     "states": ["State A", "State B", "State C", "State D", "State E"],
    #     "population": [random.randint(1000000, 5000000) for _ in range(5)]
    # })

    # st.markdown('#### Open List')
    # # Display random data
    # st.dataframe(random_data,
    #                 column_order=("states", "population"),
    #                 hide_index=True,
    #                 width=None,
    #                 column_config={
    #                     "states": st.column_config.TextColumn(
    #                         "States",
    #                     ),
    #                     "population": st.column_config.ProgressColumn(
    #                         "Population",
    #                         format="%f",
    #                         min_value=0,
    #                         max_value=max(random_data.population),
    #                     )}
    #                 )

    # st.markdown('#### Closed List')
    # # Display random data
    # st.dataframe(random_data,
    #                 column_order=("states", "population"),
    #                 hide_index=True,
    #                 width=None,
    #                 column_config={
    #                     "states": st.column_config.TextColumn(
    #                         "States",
    #                     ),
    #                     "population": st.column_config.ProgressColumn(
    #                         "Population",
    #                         format="%f",
    #                         min_value=0,
    #                         max_value=max(random_data.population),
    #                     )}
    #                 )

