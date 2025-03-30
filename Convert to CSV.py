import requests
import pandas as pd
import xml.etree.ElementTree as ET
import pytz
from datetime import datetime
from dotenv import load_dotenv
import os


# Ruta del archivo XML
# xml_file = "xml_files/rss-2025-03-28_16-04-56.xml"

load_dotenv()
RSS_URL = os.getenv("RSS_FEED_URL")

# Fetch the XML data
response = requests.get(RSS_URL)
response.raise_for_status()  # Raise an exception for HTTP errors

# Parse the XML data
xml_file = ET.fromstring(response.content)

print(xml_file)

# Analizar el archivo XML
tree = ET.ElementTree(xml_file)
root = tree.getroot()

# Función para recorrer todos los nodos
def parse_element(element, parent_tag=""):
    data = {}
    for child in element:
        tag_name = f"{parent_tag}.{child.tag}" if parent_tag else child.tag
        if len(child) > 0:
            # Si tiene hijos, llamar a parse_element de forma recursiva
            data.update(parse_element(child, tag_name))
        else:
            # Si no tiene hijos, agregar el texto del nodo
            data[tag_name] = child.text if child.text is not None else ""
    return data

# Extraer información de cada nodo <port>
data = []
for port in root.findall(".//port"):
    port_data = parse_element(port)
    data.append(port_data)

# Convertir la lista en un DataFrame de pandas
df = pd.DataFrame(data)
df.head(5)

def pivot_col(datafram, piv_columns, id_cols, col_name, col_value):
    # Reestructurar el DataFrame usando melt
    # print(datafram.columns)
    piv_df = datafram.melt(
        id_vars=id_cols,
        value_vars=piv_columns.keys(),
        var_name=col_name,
        value_name=col_value
    )

    # Mapear los valores de la columna 'lane_type' usando el diccionario
    piv_df[col_name] = piv_df[col_name].map(piv_columns)
    return piv_df

# Crear la columna 'time_mdt' convirtiendo las horas a la zona horaria MDT
def convert_to_mdt(row):
    try:
        # Obtener el nombre válido de la zona horaria
        local_tz_name = time_zone_mapping.get(row['time_zone'], None)
        if not local_tz_name:
            return None  # Si no hay mapeo, devolver None
        
        # Crear un objeto datetime con la zona horaria original
        local_tz = pytz.timezone(local_tz_name)
        mdt_tz = pytz.timezone('US/Mountain')
        local_time = local_tz.localize(datetime.combine(datetime.today(), row['time']))
        mdt_time = local_time.astimezone(mdt_tz)
        return mdt_time.strftime('%H:%M:%S')  # Devolver la hora en formato de 24 horas como string
    except Exception as e:
        print(f"Error al convertir la hora: {e}")
        return None


# Filtrar las columnas que empiezan con "commercial", "passenger" y "pedestrian"
#pivot_columns = [col for col in df.columns if col.startswith(('commercial', 'passenger', 'pedestrian'))]

#pivot las columnas de automation_type
pivot_columns = {
    'commercial_automation_type': 'commercial',
    'passenger_automation_type': 'passenger',
    'pedestrain_automation_type': 'pedestrian'
}

id_columns=['port_number', 'border', 'port_name', 'crossing_name', 'hours', 'date', 'port_status', 'construction_notice']
column_name='lane_type'
column_value='automation_type'
df1=pivot_col(df, pivot_columns, id_columns, column_name, column_value)


#pivot las columnas de maximum_lanes
pivot_columns = {
    'commercial_vehicle_lanes.maximum_lanes': 'commercial',
    'passenger_vehicle_lanes.maximum_lanes': 'passenger',
    'pedestrian_lanes.maximum_lanes': 'pedestrian'
}

id_columns=['port_number']
column_value='max_lanes'
df2=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

#pivot las columnas de standard_lanes.update_time
pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.update_time': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.update_time': 'passenger',
    'pedestrian_lanes.standard_lanes.update_time': 'pedestrian'
}

column_value='update_time'
df3=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

# Add the new column 'lane_subtype' with the value 'standard'
df3['lane_subtype'] = 'standard'

#pivot las columnas de standard_lanes.lanes_open
pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.lanes_open': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.lanes_open': 'passenger',
    'pedestrian_lanes.standard_lanes.lanes_open': 'pedestrian'
}

column_value='lanes_open'
df4=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

#pivot las columnas de standard_lanes.operational_status
pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.operational_status': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.operational_status': 'passenger',
    'pedestrian_lanes.standard_lanes.operational_status': 'pedestrian'
}

column_value='operational_status'
df5=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

#pivot las columnas de standard_lanes.delay_minutes
pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.delay_minutes': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.delay_minutes': 'passenger',
    'pedestrian_lanes.standard_lanes.delay_minutes': 'pedestrian'
}

column_value='delay_minutes'
df6=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

# Realizar el "left join" entre df1 y df2 utilizando las columnas 'port_number' y 'lane_type' como claves
merged_df = pd.merge(df1, df2, on=['port_number', 'lane_type'], how='left')
merged_df.head(5)

# Realizar el "left join" entre el resultado anterior y df3 utilizando las mismas claves
merged1_df = pd.merge(merged_df, df3, on=['port_number', 'lane_type'], how='left')

# Realizar el "left join" entre el resultado anterior y df3 utilizando las mismas claves
merged2_df = pd.merge(merged1_df, df4, on=['port_number', 'lane_type'], how='left')

# Realizar el "left join" entre el resultado anterior y df3 utilizando las mismas claves
merged3_df = pd.merge(merged2_df, df5, on=['port_number', 'lane_type'], how='left')

# Realizar el "left join" entre el resultado anterior y df3 utilizando las mismas claves
merged4_df = pd.merge(merged3_df, df6, on=['port_number', 'lane_type'], how='left')


# Crear el DataFrame df7 con las columnas commercial_vehicle_lanes.FAST_lanes
df7 = df[['port_number', 
          'commercial_vehicle_lanes.FAST_lanes.operational_status', 
          'commercial_vehicle_lanes.FAST_lanes.update_time', 
          'commercial_vehicle_lanes.FAST_lanes.delay_minutes', 
          'commercial_vehicle_lanes.FAST_lanes.lanes_open']].copy()

# Renombrar las columnas 
df7.rename(columns={
    'commercial_vehicle_lanes.FAST_lanes.operational_status': 'operational_status',
    'commercial_vehicle_lanes.FAST_lanes.update_time': 'update_time',
    'commercial_vehicle_lanes.FAST_lanes.delay_minutes': 'delay_minutes',
    'commercial_vehicle_lanes.FAST_lanes.lanes_open': 'lanes_open'
}, inplace=True)

# Filtrar las filas de merged_df donde 'lane_type' sea 'commercial'
merged_df_commercial = merged_df[merged_df['lane_type'] == 'commercial']

# Unir df7 con merged_df_commercial utilizando el campo 'port_number'
df8 = pd.merge(merged_df_commercial, df7, on='port_number', how='left')

# Insertar la columna 'lane_subtype' con el valor 'FAST lanes' después de la columna 'update_time'
df8.insert(df8.columns.get_loc('update_time') + 1, 'lane_subtype', 'FAST lanes')

# Unir los DataFrames merged4_df y df8 en el DataFrame final_df
final_df = pd.concat([merged4_df, df8], ignore_index=True)

#print(df.columns)

# Crear el DataFrame con las columnas passenger_vehicle_lanes.NEXUS_SENTRI_lane
df7 = df[['port_number', 
          'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.operational_status', 
          'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.update_time', 
          'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.delay_minutes', 
          'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.lanes_open']].copy()

# Renombrar las columnas 
df7.rename(columns={
    'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.operational_status': 'operational_status',
    'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.update_time': 'update_time',
    'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.delay_minutes': 'delay_minutes',
    'passenger_vehicle_lanes.NEXUS_SENTRI_lanes.lanes_open': 'lanes_open'
}, inplace=True)

# Filtrar las filas de merged_df donde 'lane_type' sea 'passenger'
merged_df_passenger = merged_df[merged_df['lane_type'] == 'passenger']

# Unir df7 con merged_df_passenger utilizando el campo 'port_number'
df8 = pd.merge(merged_df_passenger, df7, on='port_number', how='left')

# Insertar la columna 'lane_subtype' con el valor 'FAST lanes' después de la columna 'update_time'
df8.insert(df8.columns.get_loc('update_time') + 1, 'lane_subtype', 'NEXUS SENTRI lane')

# Unir los DataFrames merged4_df y df8 en el DataFrame final_df
final_df = pd.concat([final_df, df8], ignore_index=True)

# Crear el DataFrame df7 con las columnas passenger_vehicle_lanes.ready_lanes
df7 = df[['port_number', 
          'passenger_vehicle_lanes.ready_lanes.operational_status', 
          'passenger_vehicle_lanes.ready_lanes.update_time', 
          'passenger_vehicle_lanes.ready_lanes.delay_minutes', 
          'passenger_vehicle_lanes.ready_lanes.lanes_open']].copy()

# Renombrar las columnas 
df7.rename(columns={
    'passenger_vehicle_lanes.ready_lanes.operational_status': 'operational_status',
    'passenger_vehicle_lanes.ready_lanes.update_time': 'update_time',
    'passenger_vehicle_lanes.ready_lanes.delay_minutes': 'delay_minutes',
    'passenger_vehicle_lanes.ready_lanes.lanes_open': 'lanes_open'
}, inplace=True)

# Filtrar las filas de merged_df donde 'lane_type' sea 'passenger'
#merged_df_passenger = merged_df[merged_df['lane_type'] == 'passenger']

# Unir df7 con merged_df_passenger utilizando el campo 'port_number'
df8 = pd.merge(merged_df_passenger, df7, on='port_number', how='left')

# Insertar la columna 'lane_subtype' con el valor 'FAST lanes' después de la columna 'update_time'
df8.insert(df8.columns.get_loc('update_time') + 1, 'lane_subtype', 'ready_lanes')

# Unir los DataFrames merged4_df y df8 en el DataFrame final_df
final_df = pd.concat([final_df, df8], ignore_index=True)

# Crear el DataFrame df7 con las columnas pedestrian_lanes.ready_lanes
df7 = df[['port_number', 
          'pedestrian_lanes.ready_lanes.operational_status', 
          'pedestrian_lanes.ready_lanes.update_time', 
          'pedestrian_lanes.ready_lanes.delay_minutes', 
          'pedestrian_lanes.ready_lanes.lanes_open']].copy()

# Renombrar las columnas 
df7.rename(columns={
    'pedestrian_lanes.ready_lanes.operational_status': 'operational_status',
    'pedestrian_lanes.ready_lanes.update_time': 'update_time',
    'pedestrian_lanes.ready_lanes.delay_minutes': 'delay_minutes',
    'pedestrian_lanes.ready_lanes.lanes_open': 'lanes_open'
}, inplace=True)

# Filtrar las filas de merged_df donde 'lane_type' sea 'passenger'
merged_df_passenger = merged_df[merged_df['lane_type'] == 'pedestrian']

# Unir df7 con merged_df_passenger utilizando el campo 'port_number'
df8 = pd.merge(merged_df_passenger, df7, on='port_number', how='left')

# Insertar la columna 'lane_subtype' con el valor 'FAST lanes' después de la columna 'update_time'
df8.insert(df8.columns.get_loc('update_time') + 1, 'lane_subtype', 'ready_lanes')

# Unir los DataFrames merged4_df y df8 en el DataFrame final_df
final_df = pd.concat([final_df, df8], ignore_index=True)

# Reemplazar "At Noon" por "At 12:00 pm" en la columna 'update_time'
final_df['update_time'] = final_df['update_time'].str.replace(r'At Noon', 'At 12:00 pm', regex=True)

# Dividir el campo 'update_time' en 'hora' (formato 24 horas) y 'zona_horaria'
final_df[['time', 'time_zone']] = final_df['update_time'].str.extract(r'At (\d{1,2}:\d{2} (?:am|pm)) (\w{3})')

# Convertir la columna 'hora' al formato de 24 horas
final_df['time'] = pd.to_datetime(final_df['time'], format='%I:%M %p').dt.time

# Mapeo de abreviaturas de zonas horarias a nombres válidos de pytz
time_zone_mapping = {
    'EDT': 'US/Eastern',
    'EST': 'US/Eastern',
    'CDT': 'US/Central',
    'CST': 'US/Central',
    'MDT': 'US/Mountain',
    'MST': 'US/Mountain',
    'PDT': 'US/Pacific',
    'PST': 'US/Pacific'
}

final_df['time_mdt'] = final_df.apply(convert_to_mdt, axis=1)

# Verificar el resultado
#print(final_df.head())

# Guardar el DataFrame en un archivo CSV
csv_file = "csv_files/rss-2025-03-28_16-04-56.csv"
final_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Los datos se han guardado en {csv_file}")