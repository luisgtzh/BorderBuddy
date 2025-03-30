import pandas as pd
import xml.etree.ElementTree as ET

# Ruta del archivo XML
xml_file = "xml_files/rss-2025-03-28_16-04-56.xml"

# Analizar el archivo XML
tree = ET.parse(xml_file)
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

# Filtrar las columnas que empiezan con "commercial", "passenger" y "pedestrian"
#pivot_columns = [col for col in df.columns if col.startswith(('commercial', 'passenger', 'pedestrian'))]

#pivot las columnas de automation_type
pivot_columns = {
    'commercial_automation_type': 'commercial',
    'passenger_automation_type': 'passenger',
    'pedestrain_automation_type': 'pedestrian'
}

id_columns=['port_number', 'border', 'port_name', 'crossing_name', 'hours', 'date', 'port_status']
column_name='lane_type'
column_value='automation_type'
df1=pivot_col(df, pivot_columns, id_columns, column_name, column_value)
#df.head(5)

#pivot las columnas de maximum_lanes
pivot_columns = {
    'commercial_vehicle_lanes.maximum_lanes': 'commercial',
    'passenger_vehicle_lanes.maximum_lanes': 'passenger',
    'pedestrian_lanes.maximum_lanes': 'pedestrian'
}

id_columns=['port_number']
column_name='lane_type'
column_value='max_lanes'
df2=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

#pivot las columnas de standard_lanes.update_time
pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.update_time': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.update_time': 'passenger',
    'pedestrian_lanes.standard_lanes.update_time': 'pedestrian'
}

id_columns=['port_number']
column_name='lane_type'
column_value='update_time'
df3=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

# Add the new column 'lane_subtype' with the value 'standard'
df3['lane_subtype'] = 'standard'

# Dividir el campo 'update_time' en 'hora' (formato 24 horas) y 'zona_horaria'
df3[['hora', 'zona_horaria']] = df3['update_time'].str.extract(r'At (\d{1,2}:\d{2} (?:am|pm)) (\w{3})')

# Convertir la columna 'time_24h' a formato de hora
#df3['time_24h'] = pd.to_datetime(df3['time_24h'], format='%H:%M:%S').dt.time

print(df3.head())

# Realizar el "left join" entre df1 y df2 utilizando las columnas 'port_number' y 'lane_type' como claves
merged_df = pd.merge(df1, df2, on=['port_number', 'lane_type'], how='left')

# Realizar el "left join" entre el resultado anterior y df3 utilizando las mismas claves
final_df = pd.merge(merged_df, df3, on=['port_number', 'lane_type'], how='left')

# Guardar el DataFrame en un archivo CSV
csv_file = "csv_files/rss-2025-03-28_16-04-56.csv"
final_df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Los datos se han guardado en {csv_file}")