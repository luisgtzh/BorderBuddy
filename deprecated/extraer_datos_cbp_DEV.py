import requests
import pandas as pd
import xml.etree.ElementTree as ET
import pytz
from datetime import datetime
from dotenv import load_dotenv
import os
import psycopg2


# Ruta del archivo XML
# xml_file = "xml_files/rss-2025-03-28_16-04-56.xml"

load_dotenv()
RSS_URL = os.getenv("RSS_FEED_URL")

# Detalles de conexión a la base de datos
USER = os.getenv("SUPABASE_USER_DEV")
PASSWORD = os.getenv("SUPABASE_PASSWORD_DEV")
HOST = os.getenv("SUPABASE_HOST_DEV")
PORT = os.getenv("SUPABASE_PORT_DEV")
DBNAME = os.getenv("SUPABASE_DBNAME_DEV")

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


# --- Efficiently handle special lane types (FAST, NEXUS_SENTRI, ready_lanes) ---
special_lanes = [
    # (lane_type, prefix, key, subtype)
    ('commercial', 'commercial_vehicle_lanes', 'FAST_lanes', 'FAST lanes'),
    ('passenger', 'passenger_vehicle_lanes', 'NEXUS_SENTRI_lanes', 'NEXUS SENTRI lane'),
    ('passenger', 'passenger_vehicle_lanes', 'ready_lanes', 'ready_lanes'),
    ('pedestrian', 'pedestrian_lanes', 'ready_lanes', 'ready_lanes')
]
special_dfs = []
for lane_type, prefix, key, subtype in special_lanes:
    base_cols = [
        'port_number',
        f'{prefix}.{key}.operational_status',
        f'{prefix}.{key}.update_time',
        f'{prefix}.{key}.delay_minutes',
        f'{prefix}.{key}.lanes_open'
    ]
    # Only keep columns that exist in df
    cols = [c for c in base_cols if c in df.columns]
    if len(cols) < 2:
        continue
    temp = df[cols].copy()
    rename_dict = {
        f'{prefix}.{key}.operational_status': 'operational_status',
        f'{prefix}.{key}.update_time': 'update_time',
        f'{prefix}.{key}.delay_minutes': 'delay_minutes',
        f'{prefix}.{key}.lanes_open': 'lanes_open'
    }
    temp.rename(columns=rename_dict, inplace=True)
    temp['lane_type'] = lane_type
    temp['lane_subtype'] = subtype
    special_dfs.append(temp)

# Concatenate all DataFrames
final_df = pd.concat([merged4_df] + special_dfs, ignore_index=True)

# Reemplazar "At Noon" por "At 12:00 pm" en la columna 'update_time'
final_df['update_time'] = final_df['update_time'].str.replace(r'At Noon', 'At 12:00 pm', regex=True)

# Dividir el campo 'update_time' en 'hora' (formato 24 horas) y 'zona_horaria'
final_df[['time', 'time_zone']] = final_df['update_time'].str.extract(r'At (\d{1,2}:\d{2} (?:am|pm)) (\w{3})')

# Fix invalid 12-hour times like '0:02 am' to '12:02 am'
final_df['time'] = final_df['time'].str.replace(r'^0:', '12:', regex=True)

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

# Replace 'NaT' with None for the 'time' and 'time_mdt' columns
final_df['time'] = final_df['time'].apply(lambda x: None if pd.isnull(x) else x)
final_df['time_mdt'] = final_df['time_mdt'].apply(lambda x: None if pd.isnull(x) else x)

# Replace empty strings with None
final_df.replace("", None, inplace=True)

# Replace NaN values with None
final_df = final_df.where(pd.notnull(final_df), None)

# Verificar el resultado
print(final_df.head())

# Guardar el DataFrame en un archivo CSV
#csv_file = "csv_files/rss-2025-03-28_16-04-56.csv"
#final_df.to_csv(csv_file, index=False, encoding="utf-8")

#data_to_insert = final_df.to_dict(orient="records")

# --- Bulk Insert to Database ---
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Conexión exitosa a la base de datos.")
    cursor = connection.cursor()

    # --- Transaction 1: Insert into cruces_fronterizos ---
    try:
        table_name = "cruces_fronterizos"
        columns = list(final_df.columns)
        insert_sql = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({', '.join(['%s'] * len(columns))});"
        )
        records = final_df.values.tolist()
        cursor.executemany(insert_sql, records)  # Bulk insert
        connection.commit()
        print("Datos insertados en cruces_fronterizos.")
    except Exception as e:
        connection.rollback()
        print(f"Error al insertar datos en cruces_fronterizos: {e}")

    # --- Transaction 2: Update cruces table ---
    try:
        # Getpuertos table from Supabase 
        puertos_df = pd.read_sql('SELECT id, port_name, border FROM puertos', connection)
        puertos_df['port_name'] = puertos_df['port_name'].astype(str).str.strip()
        puertos_df['border'] = puertos_df['border'].astype(str).str.strip()
        if 'port_name' in final_df.columns and 'border' in final_df.columns:
            final_df['port_name'] = final_df['port_name'].astype(str).str.strip()
            final_df['border'] = final_df['border'].astype(str).str.strip()
            final_df = pd.merge(final_df, puertos_df[['id', 'port_name', 'border']], on=['port_name', 'border'], how='left')
            final_df.rename(columns={'id': 'port_id'}, inplace=True)
        else:
            final_df['port_id'] = None

        cruces_df = pd.read_sql('SELECT * FROM cruces', connection)
        for col in ['crossing_name', 'port_id', 'lane_type', 'lane_subtype']:
            cruces_df[col] = cruces_df[col].astype(str).str.strip().str.lower()
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).str.strip().str.lower()
            else:
                final_df[col] = None
        # Remove rows from final_df where port_id is missing (None or nan)
        final_df = final_df[final_df['port_id'].notnull() & (final_df['port_id'].astype(str).str.lower() != 'none')]
        cruces_df['merge_key'] = cruces_df['crossing_name'] + '|' + cruces_df['port_id'] + '|' + cruces_df['lane_type'] + '|' + cruces_df['lane_subtype']
        final_df['merge_key'] = final_df['crossing_name'] + '|' + final_df['port_id'] + '|' + final_df['lane_type'] + '|' + final_df['lane_subtype']
        cruces_df.set_index('merge_key', inplace=True)
        final_df.set_index('merge_key', inplace=True)
        update_cols = ['lanes_open', 'delay_minutes', 'time', 'time_zone', 'max_lanes', 'update_time']
        common_keys = cruces_df.index.intersection(final_df.index)
        # Ensure update columns are object dtype to avoid dtype warnings
        for col in update_cols:
            if col in cruces_df.columns:
                cruces_df[col] = cruces_df[col].astype('object')
        # Merge logic: for each cell, use value from final_df if not null/empty, else keep cruces_df value
        for key in common_keys:
            for col in update_cols:
                if col in final_df.columns:
                    val = final_df.at[key, col]
                    if val is not None and (not (isinstance(val, float) and pd.isnull(val))) and str(val).strip() != '':
                        cruces_df.at[key, col] = val
        # Convert columns that should be integers to Int64 (nullable integer)
        int_columns = ['lanes_open', 'delay_minutes', 'max_lanes']
        for col in int_columns:
            if col in cruces_df.columns:
                cruces_df[col] = pd.to_numeric(cruces_df[col], errors='coerce').astype('Int64')
        cruces_df.reset_index(drop=True, inplace=True)
        # Use a temp table for safe update
        temp_table = 'cruces_temp_update'
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table};")
        cursor.execute(f"CREATE TABLE {temp_table} (LIKE cruces INCLUDING ALL);")
        from io import StringIO
        sio = StringIO()
        cruces_df.to_csv(sio, sep='\t', header=False, index=False, na_rep='\\N')
        sio.seek(0)
        cursor.copy_from(sio, temp_table, null='\\N')
        cursor.execute('BEGIN;')
        cursor.execute('DELETE FROM cruces;')
        cursor.execute(f'INSERT INTO cruces SELECT * FROM {temp_table};')
        cursor.execute(f'DROP TABLE {temp_table};')
        cursor.execute('COMMIT;')
        print('cruces table updated successfully (merged with non-null values from final_df).')
    except Exception as e:
        connection.rollback()
        print(f"Error al actualizar cruces: {e}")

    cursor.close()
    connection.close()
    print("Conexión cerrada.")
except Exception as e:
    print(f"Error general de conexión o ejecución: {e}")