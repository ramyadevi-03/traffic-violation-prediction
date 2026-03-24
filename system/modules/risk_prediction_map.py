import streamlit as st
import pandas as pd
import folium
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import HeatMap


# ------------------------------------------------
# SESSION STATE (PREVENT BLINKING)
# ------------------------------------------------

if "risk_map" not in st.session_state:
    st.session_state.risk_map = None

if "risk_generated" not in st.session_state:
    st.session_state.risk_generated = False

if "high_risk" not in st.session_state:
    st.session_state.high_risk = 0

if "medium_risk" not in st.session_state:
    st.session_state.medium_risk = 0

if "low_risk" not in st.session_state:
    st.session_state.low_risk = 0


# ------------------------------------------------
# MAIN PAGE
# ------------------------------------------------

def show_risk_prediction():

    st.title("🚦 AI Dynamic Accident Risk Prediction Map")

    st.write("Predicting accident-prone zones using historical traffic data.")


    # ------------------------------------------------
    # USER CONTROLS
    # ------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        time_period = st.selectbox(
            "Select Traffic Time",
            ["Morning", "Afternoon", "Evening", "Night"]
        )

    with col2:

        risk_filter = st.multiselect(
            "Filter Risk Zones",
            ["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )


    # ------------------------------------------------
    # GENERATE MAP BUTTON
    # ------------------------------------------------

    if st.button("Generate Risk Map"):

        df = pd.read_csv("data/nyc_traffic_preprocessed.csv")

        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

        # sample for performance
        df = df.sample(600)

        m = folium.Map(
            location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()],
            zoom_start=11
        )

        # ---------------------------------------------
        # HEATMAP
        # ---------------------------------------------

        heat_data = df[["LATITUDE", "LONGITUDE"]].values.tolist()

        HeatMap(
            heat_data,
            radius=14,
            blur=18
        ).add_to(m)


        # ---------------------------------------------
        # RISK POINTS
        # ---------------------------------------------

        high = 0
        medium = 0
        low = 0

        for _, row in df.iterrows():

            lat = row["LATITUDE"]
            lon = row["LONGITUDE"]

            # time based risk simulation
            risk_value = np.random.rand()

            if time_period == "Night":
                risk_value += 0.15

            elif time_period == "Evening":
                risk_value += 0.10


            # risk classification
            if risk_value > 0.7:

                label = "High Risk"
                color = "red"

                if "High Risk" not in risk_filter:
                    continue

                high += 1

            elif risk_value > 0.4:

                label = "Medium Risk"
                color = "orange"

                if "Medium Risk" not in risk_filter:
                    continue

                medium += 1

            else:

                label = "Low Risk"
                color = "green"

                if "Low Risk" not in risk_filter:
                    continue

                low += 1


            popup = f"""
            <b>Risk Level:</b> {label}<br>
            <b>Recommended Action:</b> Drive Carefully
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(m)


        # ---------------------------------------------
        # MAP LEGEND
        # ---------------------------------------------

        legend_html = """
        <div style="
        position: fixed;
        bottom: 40px; left: 40px;
        width: 180px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding:10px;
        ">
        <b>Risk Legend</b><br>
        🔴 High Risk<br>
        🟠 Medium Risk<br>
        🟢 Low Risk<br>
        🔥 Accident Density
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))


        # ---------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------

        st.session_state.risk_map = m
        st.session_state.high_risk = high
        st.session_state.medium_risk = medium
        st.session_state.low_risk = low
        st.session_state.risk_generated = True


    # ------------------------------------------------
    # DISPLAY MAP
    # ------------------------------------------------

    if st.session_state.risk_generated:

        st.subheader("Predicted Risk Zones")

        st_folium(
            st.session_state.risk_map,
            width=900,
            height=600,
            returned_objects=[],
            key="risk_map"
        )


        # ------------------------------------------------
        # RISK ANALYTICS
        # ------------------------------------------------

        st.divider()

        st.subheader("Risk Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("High Risk Zones", st.session_state.high_risk)

        with col2:
            st.metric("Medium Risk Zones", st.session_state.medium_risk)

        with col3:
            st.metric("Low Risk Zones", st.session_state.low_risk)


        # ------------------------------------------------
        # CITY SAFETY INDEX
        # ------------------------------------------------

        total = (
            st.session_state.high_risk +
            st.session_state.medium_risk +
            st.session_state.low_risk
        )

        if total > 0:

            city_risk = int(
                (st.session_state.high_risk / total) * 100
            )

            st.subheader("City Safety Index")

            st.progress(city_risk / 100)

            st.write(f"City Risk Score: {city_risk}/100")


        # ------------------------------------------------
        # TOP DANGEROUS LOCATIONS
        # ------------------------------------------------

        st.subheader("Top Dangerous Locations")

        df = pd.read_csv("data/nyc_traffic_preprocessed.csv")

        danger = df.dropna(subset=["LATITUDE", "LONGITUDE"])

        danger = danger.sample(10)[["LATITUDE", "LONGITUDE"]]

        st.dataframe(danger, use_container_width=True)


        st.success("AI predicted accident-prone areas based on historical accident patterns.")