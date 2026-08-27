import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============================================
# CARGAR MODELOS
# ============================================

@st.cache_resource
def cargar_modelos():
    with open('modelo_xgb.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('modelo_kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return xgb, kmeans, scaler, encoders

xgb, kmeans, scaler, encoders = cargar_modelos()

# ============================================
# INTERFAZ
# ============================================

st.set_page_config(page_title="Predictor de Cancelaciones Hoteleras",
                   page_icon="🏨", layout="centered")

st.title("🏨 Predictor de Cancelaciones Hoteleras")
st.markdown("Introduce los datos de una nueva reserva para obtener la probabilidad de cancelación y la recomendación de acción.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Tipo de hotel", ["City Hotel", "Resort Hotel"])
    lead_time = st.slider("Días de antelación", 0, 737, 90)
    total_nights = st.slider("Noches de estancia", 1, 20, 3)
    adr = st.number_input("Precio por noche (€)", 0.0, 1000.0, 100.0)
    adults = st.number_input("Número de adultos", 1, 10, 2)
    deposit_type = st.selectbox("Tipo de depósito",
                                ["Sin depósito", "No reembolsable", "Reembolsable"])

with col2:
    distribution_channel = st.selectbox("Canal de distribución",
                                        ["Directo", "Corporativo", "Agencia de viajes", "GDS"])
    customer_type = st.selectbox("Tipo de cliente", [
        "Individual (sin contrato)",
        "Empresa con contrato",
        "Grupo de particulares",
        "Grupo organizado"
    ])
    meal = st.selectbox("Régimen alimenticio", [
        "Solo alojamiento (sin comidas)",
        "Alojamiento y desayuno",
        "Media pensión (desayuno y cena)",
        "Pensión completa (todas las comidas)"
    ])
    market_segment = st.selectbox("Segmento de mercado",
                                  ["Directo", "Corporativo", "Agencia online",
                                   "Agencia offline", "Grupos", "Complementario", "Aviación"])
    is_repeated_guest = st.checkbox("¿Es cliente repetidor?")
    required_car_parking = st.checkbox("¿Necesita aparcamiento?")

st.divider()

col3, col4 = st.columns(2)
with col3:
    previous_cancellations = st.number_input("Cancelaciones previas", 0, 26, 0)
    previous_bookings_not_canceled = st.number_input("Reservas previas completadas", 0, 72, 0)
with col4:
    booking_changes = st.number_input("Cambios en la reserva", 0, 21, 0)
    total_special_requests = st.number_input("Peticiones especiales", 0, 5, 0)
    days_in_waiting_list = st.number_input("Días en lista de espera", 0, 391, 0)

st.divider()

# ============================================
# PREDICCIÓN
# ============================================

if st.button("Predecir", use_container_width=True, type="primary"):

    # Mapeo español → inglés
    deposit_map = {"Sin depósito": "No Deposit",
                   "No reembolsable": "Non Refund",
                   "Reembolsable": "Refundable"}
    channel_map = {"Directo": "Direct",
                   "Corporativo": "Corporate",
                   "Agencia de viajes": "TA/TO",
                   "GDS": "GDS"}
    customer_map = {"Individual (sin contrato)": "Transient",
                    "Empresa con contrato": "Contract",
                    "Grupo de particulares": "Transient-Party",
                    "Grupo organizado": "Group"}
    meal_map = {"Solo alojamiento (sin comidas)": "SC",
                "Alojamiento y desayuno": "BB",
                "Media pensión (desayuno y cena)": "HB",
                "Pensión completa (todas las comidas)": "FB"}
    segment_map = {"Directo": "Direct",
                   "Corporativo": "Corporate",
                   "Agencia online": "Online TA",
                   "Agencia offline": "Offline TA/TO",
                   "Grupos": "Groups",
                   "Complementario": "Complementary",
                   "Aviación": "Aviation"}

    # Codificar con encoders reales
    hotel_enc = encoders['hotel'].transform([hotel])[0]
    deposit_enc = encoders['deposit_type'].transform([deposit_map[deposit_type]])[0]
    channel_enc = encoders['distribution_channel'].transform([channel_map[distribution_channel]])[0]
    customer_enc = encoders['customer_type'].transform([customer_map[customer_type]])[0]

    # NOTA: 'meal' y 'market_segment' no tienen encoder guardado en encoders.pkl,
    # por lo que se codifican manualmente replicando el orden alfabético que usa
    # LabelEncoder por defecto (confirmado con los 4 encoders reales del pickle).
    # Antes de la entrega final, verificar con sorted(df['meal'].unique()) en el
    # notebook que este mapeo coincide exactamente con el usado en el entrenamiento.
    meal_enc = {"BB": 0, "FB": 1, "HB": 2, "SC": 3}[meal_map[meal]]
    segment_enc = {"Aviation": 0, "Complementary": 1, "Corporate": 2,
                   "Direct": 3, "Groups": 4, "Offline TA/TO": 5, "Online TA": 6}[segment_map[market_segment]]

    # Clustering
    # IMPORTANTE: el orden de estas columnas debe coincidir EXACTAMENTE con
    # scaler.feature_names_in_ (verificado contra el scaler.pkl real):
    # [..., hotel, distribution_channel, customer_type, deposit_type]
    cluster_input = np.array([[
        lead_time, total_nights, adr,
        int(is_repeated_guest), previous_cancellations,
        previous_bookings_not_canceled, booking_changes,
        total_special_requests, int(required_car_parking),
        hotel_enc, channel_enc, customer_enc, deposit_enc
    ]])
    cluster_scaled = scaler.transform(cluster_input)
    cluster = kmeans.predict(cluster_scaled)[0]

    # Predicción
    model_input = pd.DataFrame([[
        lead_time, total_nights, adr,
        int(is_repeated_guest), previous_cancellations,
        previous_bookings_not_canceled, booking_changes,
        total_special_requests, int(required_car_parking),
        hotel_enc, channel_enc, customer_enc, deposit_enc,
        meal_enc, segment_enc, days_in_waiting_list, adults, cluster
    ]], columns=[
        'lead_time', 'total_nights', 'adr', 'is_repeated_guest',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'total_of_special_requests',
        'required_car_parking_spaces', 'hotel', 'distribution_channel',
        'customer_type', 'deposit_type', 'meal', 'market_segment',
        'days_in_waiting_list', 'adults', 'cluster'
    ])

    prob_cancelacion = xgb.predict_proba(model_input)[0][1] * 100

    # Mostrar resultado
    st.markdown("## 📊 Resultado")

    if prob_cancelacion >= 70:
        st.error(f"## 🔴 {prob_cancelacion:.1f}% de probabilidad de cancelación")
    elif prob_cancelacion >= 40:
        st.warning(f"## 🟡 {prob_cancelacion:.1f}% de probabilidad de cancelación")
    else:
        st.success(f"## 🟢 {prob_cancelacion:.1f}% de probabilidad de cancelación")

    # Segmento de cliente (conecta la predicción con la fase de clustering del TFM)
    nombres_cluster = {
        0: "Cliente Fiel",
        1: "Turista Estándar",
        2: "Cancelador Seguro",
        3: "Vacacionista Planificado",
    }
    st.caption(f"Segmento de cliente asignado: **{nombres_cluster.get(cluster, cluster)}**")

    # Recomendación
    st.markdown("### 💡 Recomendación")
    if prob_cancelacion >= 70:
        if deposit_type == "Sin depósito":
            recomendacion = "**Riesgo muy alto.** Considera solicitar depósito o cambiar a política no reembolsable para asegurar el ingreso."
        elif deposit_type == "Reembolsable":
            recomendacion = "**Riesgo muy alto.** Considera cambiar a política no reembolsable. Valora el overbooking controlado."
        else:
            recomendacion = "**Riesgo muy alto.** El depósito no reembolsable ya está aplicado. Considera overbooking controlado o contactar al cliente para confirmar intención."
        st.error(recomendacion)

    elif prob_cancelacion >= 40:
        recomendacion = "**Riesgo medio.** Envía confirmación de reserva e incentivo de permanencia 30 días antes de la llegada."
        st.warning(recomendacion)

    else:
        recomendacion = "**Riesgo bajo.** Reserva estable. Oportunidad de upselling: spa, pensión completa, actividades."
        st.success(recomendacion)
