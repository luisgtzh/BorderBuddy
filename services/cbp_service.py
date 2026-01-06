import os
from datetime import datetime
import xml.etree.ElementTree as ET

import pandas as pd
import psycopg2
import pytz
import requests
from dotenv import load_dotenv

"""
Service to pull CBP RSS data, normalize it, and update database tables.
Functionality matches the original script while improving readability and structure.
"""

load_dotenv()
RSS_URL = os.getenv("RSS_FEED_URL")
ENV = (os.getenv("ENV") or "dev").lower()

# Detalles de conexión a la base de datos
USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
HOST = os.getenv("SUPABASE_HOST")
PORT = os.getenv("SUPABASE_PORT")
DBNAME = os.getenv("SUPABASE_DBNAME")

CONTEXT_COLUMNS = [
    "port_number",
    "border",
    "port_name",
    "crossing_name",
    "hours",
    "date",
    "port_status",
    "construction_notice",
]

AUTOMATION_PIVOT = {
    "commercial_automation_type": "commercial",
    "passenger_automation_type": "passenger",
    "pedestrain_automation_type": "pedestrian",
}

MAX_LANES_PIVOT = {
    "commercial_vehicle_lanes.maximum_lanes": "commercial",
    "passenger_vehicle_lanes.maximum_lanes": "passenger",
    "pedestrian_lanes.maximum_lanes": "pedestrian",
}

UPDATE_TIME_PIVOT = {
    "commercial_vehicle_lanes.standard_lanes.update_time": "commercial",
    "passenger_vehicle_lanes.standard_lanes.update_time": "passenger",
    "pedestrian_lanes.standard_lanes.update_time": "pedestrian",
}

LANES_OPEN_PIVOT = {
    "commercial_vehicle_lanes.standard_lanes.lanes_open": "commercial",
    "passenger_vehicle_lanes.standard_lanes.lanes_open": "passenger",
    "pedestrian_lanes.standard_lanes.lanes_open": "pedestrian",
}

OPERATIONAL_STATUS_PIVOT = {
    "commercial_vehicle_lanes.standard_lanes.operational_status": "commercial",
    "passenger_vehicle_lanes.standard_lanes.operational_status": "passenger",
    "pedestrian_lanes.standard_lanes.operational_status": "pedestrian",
}

DELAY_MINUTES_PIVOT = {
    "commercial_vehicle_lanes.standard_lanes.delay_minutes": "commercial",
    "passenger_vehicle_lanes.standard_lanes.delay_minutes": "passenger",
    "pedestrian_lanes.standard_lanes.delay_minutes": "pedestrian",
}

SPECIAL_LANES = [
    ("commercial", "commercial_vehicle_lanes", "FAST_lanes", "FAST lanes"),
    ("passenger", "passenger_vehicle_lanes", "NEXUS_SENTRI_lanes", "NEXUS SENTRI lane"),
    ("passenger", "passenger_vehicle_lanes", "ready_lanes", "ready_lanes"),
    ("pedestrian", "pedestrian_lanes", "ready_lanes", "ready_lanes"),
]

TIME_ZONE_MAPPING = {
    "EDT": "US/Eastern",
    "EST": "US/Eastern",
    "CDT": "US/Central",
    "CST": "US/Central",
    "MDT": "US/Mountain",
    "MST": "US/Mountain",
    "PDT": "US/Pacific",
    "PST": "US/Pacific",
}


def fetch_xml_root(url: str) -> ET.Element:
    """Fetch RSS XML and return the root element."""
    response = requests.get(url)
    response.raise_for_status()
    xml_file = ET.fromstring(response.content)
    if ENV != "prod":
        print(xml_file)
    return xml_file


# Función para recorrer todos los nodos
def parse_element(element: ET.Element, parent_tag: str = "") -> dict:
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


def parse_ports_to_dataframe(root: ET.Element) -> pd.DataFrame:
    """Parse all <port> elements into a flattened DataFrame."""
    data = []
    for port in root.findall(".//port"):
        port_data = parse_element(port)
        data.append(port_data)
    return pd.DataFrame(data)


def pivot_col(datafram: pd.DataFrame, piv_columns: dict, id_cols: list, col_name: str, col_value: str) -> pd.DataFrame:
    # Reestructurar el DataFrame usando melt
    piv_df = datafram.melt(
        id_vars=id_cols,
        value_vars=piv_columns.keys(),
        var_name=col_name,
        value_name=col_value,
    )

    # Mapear los valores de la columna 'lane_type' usando el diccionario
    piv_df[col_name] = piv_df[col_name].map(piv_columns)
    return piv_df


def build_standard_lane_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build the base dataframe for standard lanes across automation, capacity, and status metrics."""
    id_columns = ["port_number", "border", "port_name", "crossing_name", "hours", "date", "port_status", "construction_notice"]
    column_name = "lane_type"

    df1 = pivot_col(df, AUTOMATION_PIVOT, id_columns, column_name, "automation_type")
    df2 = pivot_col(df, MAX_LANES_PIVOT, ["port_number"], column_name, "max_lanes")
    df3 = pivot_col(df, UPDATE_TIME_PIVOT, ["port_number"], column_name, "update_time")
    df3["lane_subtype"] = "standard"
    df4 = pivot_col(df, LANES_OPEN_PIVOT, ["port_number"], column_name, "lanes_open")
    df5 = pivot_col(df, OPERATIONAL_STATUS_PIVOT, ["port_number"], column_name, "operational_status")
    df6 = pivot_col(df, DELAY_MINUTES_PIVOT, ["port_number"], column_name, "delay_minutes")

    merged_df = (
        df1.merge(df2, on=["port_number", column_name], how="left")
        .merge(df3, on=["port_number", column_name], how="left")
        .merge(df4, on=["port_number", column_name], how="left")
        .merge(df5, on=["port_number", column_name], how="left")
        .merge(df6, on=["port_number", column_name], how="left")
    )
    return merged_df


# Crear la columna 'time_mdt' convirtiendo las horas a la zona horaria MDT
def convert_to_mdt(row: pd.Series):
    try:
        # Obtener el nombre válido de la zona horaria
        local_tz_name = TIME_ZONE_MAPPING.get(row["time_zone"], None)
        if not local_tz_name:
            return None  # Si no hay mapeo, devolver None

        # Crear un objeto datetime con la zona horaria original
        local_tz = pytz.timezone(local_tz_name)
        mdt_tz = pytz.timezone("US/Mountain")
        local_time = local_tz.localize(datetime.combine(datetime.today(), row["time"]))
        mdt_time = local_time.astimezone(mdt_tz)
        return mdt_time.strftime("%H:%M:%S")  # Devolver la hora en formato de 24 horas como string
    except Exception as e:
        print(f"Error al convertir la hora: {e}")
        return None


def build_special_lane_dataframes(df: pd.DataFrame) -> list:
    """Build dataframes for special lane types (FAST, NEXUS_SENTRI, ready_lanes)."""
    special_dfs = []
    context_cols = [c for c in CONTEXT_COLUMNS if c in df.columns]
    for lane_type, prefix, key, subtype in SPECIAL_LANES:
        metric_cols = [
            f"{prefix}.{key}.operational_status",
            f"{prefix}.{key}.update_time",
            f"{prefix}.{key}.delay_minutes",
            f"{prefix}.{key}.lanes_open",
        ]
        cols = context_cols + [c for c in metric_cols if c in df.columns]
        if len(cols) <= len(context_cols):
            continue
        temp = df[cols].copy()
        rename_dict = {
            f"{prefix}.{key}.operational_status": "operational_status",
            f"{prefix}.{key}.update_time": "update_time",
            f"{prefix}.{key}.delay_minutes": "delay_minutes",
            f"{prefix}.{key}.lanes_open": "lanes_open",
        }
        temp.rename(columns=rename_dict, inplace=True)
        temp["lane_type"] = lane_type
        temp["lane_subtype"] = subtype
        if "automation_type" not in temp.columns:
            temp["automation_type"] = None
        special_dfs.append(temp)
    return special_dfs


def normalize_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Apply time normalization and fill required fields."""
    for col in ["construction_notice", "operational_status"]:
        if col not in final_df.columns:
            final_df[col] = None

    final_df["update_time"] = final_df["update_time"].str.replace(r"At Noon", "At 12:00 pm", regex=True)
    final_df[["time", "time_zone"]] = final_df["update_time"].str.extract(r"At (\d{1,2}:\d{2} (?:am|pm)) (\w{3})")
    final_df["time"] = final_df["time"].str.replace(r"^0:", "12:", regex=True)
    final_df["time"] = pd.to_datetime(final_df["time"], format="%I:%M %p").dt.time
    final_df["time_mdt"] = final_df.apply(convert_to_mdt, axis=1)
    final_df["time"] = final_df["time"].apply(lambda x: None if pd.isnull(x) else x)
    final_df["time_mdt"] = final_df["time_mdt"].apply(lambda x: None if pd.isnull(x) else x)
    final_df.replace("", None, inplace=True)
    final_df = final_df.where(pd.notnull(final_df), None)
    return final_df


def insert_cruces_fronterizos(connection, cursor, final_df: pd.DataFrame):
    """Insert rows into cruces_fronterizos when running in prod."""
    if ENV != "prod":
        print(f"ENV='{ENV}' detectado; se omite inserción en cruces_fronterizos.")
        return

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


def update_cruces_table(connection, cursor, final_df: pd.DataFrame):
    """Update cruces table using non-null values from the latest feed."""
    try:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'cruces'
            ORDER BY ordinal_position
            """
        )
        cruces_columns = [row[0] for row in cursor.fetchall()]

        puertos_df = pd.read_sql("SELECT id, port_name, border FROM puertos", connection)
        puertos_df["port_name"] = puertos_df["port_name"].astype(str).str.strip()
        puertos_df["border"] = puertos_df["border"].astype(str).str.strip()
        if "port_name" in final_df.columns and "border" in final_df.columns:
            final_df["port_name"] = final_df["port_name"].astype(str).str.strip()
            final_df["border"] = final_df["border"].astype(str).str.strip()
            final_df = pd.merge(final_df, puertos_df[["id", "port_name", "border"]], on=["port_name", "border"], how="left")
            final_df.rename(columns={"id": "port_id"}, inplace=True)
        else:
            final_df["port_id"] = None

        cruces_df = pd.read_sql("SELECT * FROM cruces", connection)
        for col in cruces_columns:
            if col not in cruces_df.columns:
                cruces_df[col] = None
        cruces_df = cruces_df[[c for c in cruces_columns if c in cruces_df.columns]]
        for col in ["crossing_name", "port_id", "lane_type", "lane_subtype"]:
            cruces_df[col] = cruces_df[col].astype(str).str.strip().str.lower()
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).str.strip().str.lower()
            else:
                final_df[col] = None
        final_df = final_df[final_df["port_id"].notnull() & (final_df["port_id"].astype(str).str.lower() != "none")]
        cruces_df["merge_key"] = cruces_df["crossing_name"] + "|" + cruces_df["port_id"] + "|" + cruces_df["lane_type"] + "|" + cruces_df["lane_subtype"]
        final_df["merge_key"] = final_df["crossing_name"] + "|" + final_df["port_id"] + "|" + final_df["lane_type"] + "|" + final_df["lane_subtype"]
        cruces_df.set_index("merge_key", inplace=True)
        final_df.set_index("merge_key", inplace=True)
        update_cols = ["lanes_open", "delay_minutes", "time", "time_zone", "max_lanes", "update_time", "construction_notice", "operational_status"]
        update_cols = [col for col in update_cols if col in cruces_columns]
        common_keys = cruces_df.index.intersection(final_df.index)
        for col in update_cols:
            cruces_df[col] = cruces_df[col].astype("object")
        for key in common_keys:
            for col in update_cols:
                if col in final_df.columns:
                    val = final_df.at[key, col]
                    if val is not None and (not (isinstance(val, float) and pd.isnull(val))) and str(val).strip() != "":
                        cruces_df.at[key, col] = val
        int_columns = ["lanes_open", "delay_minutes", "max_lanes"]
        for col in int_columns:
            if col in cruces_df.columns:
                cruces_df[col] = pd.to_numeric(cruces_df[col], errors="coerce").astype("Int64")
        cruces_df.reset_index(drop=True, inplace=True)
        cruces_df = cruces_df.reindex(columns=cruces_columns)
        cruces_df = cruces_df.where(pd.notnull(cruces_df), None)
        temp_table = "cruces_temp_update"
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table};")
        cursor.execute(f"CREATE TABLE {temp_table} (LIKE cruces INCLUDING ALL);")
        from io import StringIO

        sio = StringIO()
        cruces_df.to_csv(sio, header=False, index=False, na_rep="\\N")
        sio.seek(0)
        copy_sql = f"COPY {temp_table} ({', '.join(cruces_columns)}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        cursor.copy_expert(copy_sql, sio)
        cursor.execute("BEGIN;")
        cursor.execute("DELETE FROM cruces;")
        cursor.execute(f"INSERT INTO cruces SELECT * FROM {temp_table};")
        cursor.execute(f"DROP TABLE {temp_table};")
        cursor.execute("COMMIT;")
        print("cruces table updated successfully (merged with non-null values from final_df).")
    except Exception as e:
        connection.rollback()
        print(f"Error al actualizar cruces (overwrite non-null values): {e}")


def run():
    xml_root = fetch_xml_root(RSS_URL)
    tree = ET.ElementTree(xml_root)
    root = tree.getroot()

    df = parse_ports_to_dataframe(root)
    standard_df = build_standard_lane_dataframe(df)
    special_dfs = build_special_lane_dataframes(df)
    final_df = pd.concat([standard_df] + special_dfs, ignore_index=True)
    final_df = normalize_dataframe(final_df)

    if ENV != "prod":
        print(final_df.head())

    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME,
        )
        print("Conexión exitosa a la base de datos.")
        cursor = connection.cursor()

        insert_cruces_fronterizos(connection, cursor, final_df)
        update_cruces_table(connection, cursor, final_df)

        cursor.close()
        connection.close()
        print("Conexión cerrada.")
    except Exception as e:
        print(f"Error general de conexión o ejecución: {e}")


if __name__ == "__main__":
    run()
