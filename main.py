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
# Sidebar
with st.sidebar:
    st.title('Rute terdekat antar Rumah Sakit')

    dropdown_label = [coord['name'] + '. ' + coord['label'] for coord in coords]
    dropdown_value = [coord['name'] for coord in coords]

    selected_coord = st.selectbox('Rumah sakit asal', options=dropdown_label)
    selected_source_value = dropdown_value[dropdown_label.index(selected_coord)]

    selected_coord = st.selectbox('Rumah sakit tujuan', options=dropdown_label)
    selected_target_value = dropdown_value[dropdown_label.index(selected_coord)]


#######################
# Dashboard Main Panel
col = st.columns((6.5, 1.5), gap='medium')

with col[0]:    # Define the place name
    import osmnx as ox
    import matplotlib.pyplot as plt
    import networkx as nx

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

    # Define your source and target nodes
    source = selected_source_value
    target = selected_target_value

    # Retrieve node IDs for source and target from your coordinates
    start_node = next((item for item in coords if item['name'] == source), None)['id']
    end_node = next((item for item in coords if item['name'] == target), None)['id']

    # Calculate the shortest path using A* algorithm
    graph = graph.to_undirected()
    shortest_path = nx.astar_path(graph, start_node, end_node, weight='length')

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


with col[1]:
    iteration = 1    
    st.markdown('#### Iterasi ke:')
    # Add left and right buttons inline
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('←'):
            iteration = max(iteration - 1, 0)
    with col2:
        st.write(iteration)
    with col3:
        if st.button('→'):
            iteration += 1

    # Generate random data
    random_data = pd.DataFrame({
        "states": ["State A", "State B", "State C", "State D", "State E"],
        "population": [random.randint(1000000, 5000000) for _ in range(5)]
    })

    st.markdown('#### Open List')
    # Display random data
    st.dataframe(random_data,
                    column_order=("states", "population"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "states": st.column_config.TextColumn(
                            "States",
                        ),
                        "population": st.column_config.ProgressColumn(
                            "Population",
                            format="%f",
                            min_value=0,
                            max_value=max(random_data.population),
                        )}
                    )

    st.markdown('#### Closed List')
    # Display random data
    st.dataframe(random_data,
                    column_order=("states", "population"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "states": st.column_config.TextColumn(
                            "States",
                        ),
                        "population": st.column_config.ProgressColumn(
                            "Population",
                            format="%f",
                            min_value=0,
                            max_value=max(random_data.population),
                        )}
                    )

