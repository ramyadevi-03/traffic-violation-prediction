import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
from modules.blackspot_detector import detect_blackspots

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "route_generated" not in st.session_state:
    st.session_state.route_generated = False

if "route_map" not in st.session_state:
    st.session_state.route_map = None

if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0

if "risk_level" not in st.session_state:
    st.session_state.risk_level = "LOW"

if "distance" not in st.session_state:
    st.session_state.distance = 0

if "travel_time" not in st.session_state:
    st.session_state.travel_time = 0

if "high_risk_count" not in st.session_state:
    st.session_state.high_risk_count = 0

if "medium_risk_count" not in st.session_state:
    st.session_state.medium_risk_count = 0

if "safe_segments" not in st.session_state:
    st.session_state.safe_segments = 0

if "directions" not in st.session_state:
    st.session_state.directions = []

geolocator = Nominatim(user_agent="traffic_ai")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_accident_data():
    df = pd.read_csv("data/nyc_traffic_preprocessed.csv")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    return df


@st.cache_data
def load_blackspots():
    return detect_blackspots("data/nyc_traffic_preprocessed.csv")


@st.cache_resource
def load_graph():
    return ox.load_graphml("road_network.graphml")


# ------------------------------------------------
# GET COORDINATES
# ------------------------------------------------

def get_coordinates(place):

    location = geolocator.geocode(place)

    if location:
        return (location.latitude, location.longitude)

    return None


# ------------------------------------------------
# DISTANCE
# ------------------------------------------------

def calculate_distance(G, route):

    distance = 0

    for u, v in zip(route[:-1], route[1:]):
        edge = G.get_edge_data(u, v)[0]
        distance += edge["length"]

    return distance / 1000


# ------------------------------------------------
# TRAVEL TIME
# ------------------------------------------------

def estimate_time(distance):

    speed = 40
    time_hours = distance / speed

    return time_hours * 60


# ------------------------------------------------
# TURN BY TURN DIRECTIONS
# ------------------------------------------------

def generate_directions(G, route):

    directions = []

    for u, v in zip(route[:-1], route[1:]):

        edge = G.get_edge_data(u, v)[0]

        street = edge.get("name", "Unnamed Road")

        length = edge.get("length", 0) / 1000

        directions.append(
            f"Continue on {street} for {length:.2f} km"
        )

    return directions


# ------------------------------------------------
# SAFE ROUTE GENERATION
# ------------------------------------------------

def generate_safe_route(start_coord, end_coord):

    G = load_graph()

    blackspots = load_blackspots()

    for u, v, k, data in G.edges(keys=True, data=True):
        data["weight"] = data.get("length", 1)

    for _, row in blackspots.iterrows():

        node = ox.distance.nearest_nodes(
            G,
            row["LONGITUDE"],
            row["LATITUDE"]
        )

        for neighbor in G.neighbors(node):

            for key in G[node][neighbor]:

                if row["Risk_Level"] == "HIGH RISK":
                    multiplier = 10

                elif row["Risk_Level"] == "MEDIUM RISK":
                    multiplier = 6

                else:
                    multiplier = 3

                G[node][neighbor][key]["weight"] *= multiplier

    orig = ox.distance.nearest_nodes(G, start_coord[1], start_coord[0])
    dest = ox.distance.nearest_nodes(G, end_coord[1], end_coord[0])

    route = nx.shortest_path(G, orig, dest, weight="weight")

    return G, route, blackspots


# ------------------------------------------------
# PLOT ROUTE
# ------------------------------------------------

def plot_route(G, route, start_coord, end_coord, blackspots, show_heatmap=True):

    m = folium.Map(location=start_coord, zoom_start=12)

    route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

    folium.PolyLine(
        route_coords,
        color="blue",
        weight=6,
        opacity=0.8
    ).add_to(m)

    folium.Marker(
        start_coord,
        popup="Start",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        end_coord,
        popup="Destination",
        icon=folium.Icon(color="red")
    ).add_to(m)

    for _, row in blackspots.iterrows():

        if row["Risk_Level"] == "HIGH RISK":
            color = "red"

        elif row["Risk_Level"] == "MEDIUM RISK":
            color = "orange"

        else:
            color = "green"

        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['Risk_Level']} ({row['Accident_Count']} accidents)"
        ).add_to(m)

    df = load_accident_data()

    heat_data = df[["LATITUDE", "LONGITUDE"]].values.tolist()

    if show_heatmap:
        HeatMap(
            heat_data,
            radius=12,
            blur=15
        ).add_to(m)

    high = 0
    medium = 0

    for _, row in blackspots.iterrows():

        for lat, lon in route_coords:

            dist = np.sqrt(
                (lat - row["LATITUDE"]) ** 2 +
                (lon - row["LONGITUDE"]) ** 2
            )

            if dist < 0.01:

                if row["Risk_Level"] == "HIGH RISK":
                    high += 1

                elif row["Risk_Level"] == "MEDIUM RISK":
                    medium += 1

    safe = len(route_coords) - (high + medium)

    risk_score = min((high * 4) + (medium * 2), 100)

    if risk_score > 70:
        level = "HIGH RISK"

    elif risk_score > 30:
        level = "MEDIUM RISK"

    else:
        level = "LOW RISK"

    return m, risk_score, level, high, medium, safe


# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------

def show_safe_route():

    st.title("🚦 AI Safe Route Recommendation")

    start = st.text_input("Start Location")

    destination = st.text_input("Destination")

    travel_time_option = st.selectbox(
        "Preferred Travel Time",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    show_heatmap = st.checkbox("Show Accident Heatmap", True)

    if st.button("Find Safest Route"):

        start_coord = get_coordinates(start)

        end_coord = get_coordinates(destination)

        if start_coord and end_coord:

            with st.spinner("Analyzing safest route..."):

                G, route, blackspots = generate_safe_route(
                    start_coord,
                    end_coord
                )

                distance = calculate_distance(G, route)

                travel_time = estimate_time(distance)

                directions = generate_directions(G, route)

                route_map, score, level, high, medium, safe = plot_route(
                    G,
                    route,
                    start_coord,
                    end_coord,
                    blackspots,
                    show_heatmap
                )

                if travel_time_option == "Evening":
                    score += 10

                if travel_time_option == "Night":
                    score += 5

                st.session_state.route_map = route_map
                st.session_state.risk_score = score
                st.session_state.risk_level = level
                st.session_state.distance = distance
                st.session_state.travel_time = travel_time
                st.session_state.high_risk_count = high
                st.session_state.medium_risk_count = medium
                st.session_state.safe_segments = safe
                st.session_state.directions = directions
                st.session_state.route_generated = True

        else:
            st.error("Location not found")

    if st.session_state.route_generated:

        col1, col2 = st.columns([1, 2])

        with col1:

            st.subheader("Route Summary")

            st.metric("Distance", f"{st.session_state.distance:.2f} km")

            st.metric(
                "Travel Time",
                f"{st.session_state.travel_time:.1f} minutes"
            )

            st.metric("Risk Score", st.session_state.risk_score)

            if st.session_state.risk_level == "HIGH RISK":
                st.error("🔴 High Risk Route")

            elif st.session_state.risk_level == "MEDIUM RISK":
                st.warning("🟠 Medium Risk Route")

            else:
                st.success("🟢 Safe Route")

            st.subheader("Risk Analytics")

            st.metric("High Risk Areas", st.session_state.high_risk_count)

            st.metric("Medium Risk Areas", st.session_state.medium_risk_count)

            st.metric("Safe Segments", st.session_state.safe_segments)

            st.subheader("AI Safety Advice")

            if st.session_state.high_risk_count > 5:
                st.warning("Avoid peak hours due to many accident zones.")

            elif st.session_state.medium_risk_count > 10:
                st.info("Drive cautiously in medium risk areas.")

            else:
                st.success("Route considered relatively safe.")

        with col2:

            map_data = st_folium(
                st.session_state.route_map,
                width=900,
                height=650,
                returned_objects=["last_clicked"],
                key="safe_route_map"
            )

            if map_data and map_data["last_clicked"]:

                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]

                st.info(f"Clicked Location: {lat:.4f}, {lon:.4f}")

        st.subheader("Turn-by-Turn Directions")

        for step in st.session_state.directions[:10]:
            st.write("➡", step)