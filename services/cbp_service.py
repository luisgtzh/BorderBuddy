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


def _first_non_empty(series: pd.Series):
    for value in series:
        if value is None or pd.isnull(value):
            continue
        value_str = str(value).strip()
        if value_str:
            return value
    return None


def _normalized_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def attach_port_ids(df: pd.DataFrame, connection) -> pd.DataFrame:
    puertos_df = pd.read_sql("SELECT id, port_name, border FROM puertos", connection)
    puertos_df["port_name"] = puertos_df["port_name"].astype(str).str.strip()
    puertos_df["border"] = puertos_df["border"].astype(str).str.strip()

    df = df.copy()
    if "port_name" in df.columns and "border" in df.columns:
        df["port_name"] = df["port_name"].astype(str).str.strip()
        df["border"] = df["border"].astype(str).str.strip()
        df = pd.merge(df, puertos_df[["id", "port_name", "border"]], on=["port_name", "border"], how="left")
        df.rename(columns={"id": "port_id"}, inplace=True)
    else:
        df["port_id"] = None
    return df


def build_crossing_groups_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"port_id", "crossing_name"}
    if not required_columns.issubset(final_df.columns):
        return pd.DataFrame(columns=["port_id", "crossing_name", "hours", "construction_notice", "operational_status"])

    group_columns = [c for c in ["port_id", "crossing_name", "hours", "construction_notice", "port_status"] if c in final_df.columns]
    group_df = final_df[group_columns].copy()
    group_df = group_df[group_df["port_id"].notnull()]
    group_df["crossing_name_norm"] = _normalized_series(group_df["crossing_name"])
    group_df = group_df[group_df["crossing_name_norm"] != ""]

    agg_map = {"crossing_name": _first_non_empty}
    if "hours" in group_df.columns:
        agg_map["hours"] = _first_non_empty
    if "construction_notice" in group_df.columns:
        agg_map["construction_notice"] = _first_non_empty
    if "port_status" in group_df.columns:
        agg_map["port_status"] = _first_non_empty

    group_df = group_df.groupby(["port_id", "crossing_name_norm"], as_index=False).agg(agg_map)
    group_df.rename(columns={"port_status": "operational_status"}, inplace=True)
    for col in ["hours", "construction_notice", "operational_status"]:
        if col not in group_df.columns:
            group_df[col] = None
    group_df = group_df.where(pd.notnull(group_df), None)
    return group_df


def upsert_crossing_groups_table(connection, cursor, group_df: pd.DataFrame):
    if group_df.empty:
        print("No crossing group data to update.")
        return

    existing_df = pd.read_sql("SELECT id, port_id, crossing_name FROM crossing_groups", connection)
    if existing_df.empty:
        existing_df = pd.DataFrame(columns=["id", "port_id", "crossing_name", "crossing_name_norm"])
    else:
        existing_df["crossing_name_norm"] = _normalized_series(existing_df["crossing_name"])

    group_df = group_df.copy()
    group_df["crossing_name_norm"] = _normalized_series(group_df["crossing_name"])

    merged = group_df.merge(
        existing_df[["id", "port_id", "crossing_name_norm"]],
        on=["port_id", "crossing_name_norm"],
        how="left",
    )
    merged = merged.where(pd.notnull(merged), None)
    update_rows = merged[merged["id"].notnull()]
    insert_rows = merged[merged["id"].isnull()]

    if not update_rows.empty:
        update_sql = """
            UPDATE crossing_groups
            SET crossing_name = %s,
                hours = COALESCE(%s, hours),
                construction_notice = COALESCE(%s, construction_notice),
                operational_status = COALESCE(%s, operational_status)
            WHERE id = %s
        """
        update_values = [
            (
                row["crossing_name"],
                row.get("hours"),
                row.get("construction_notice"),
                row.get("operational_status"),
                row["id"],
            )
            for _, row in update_rows.iterrows()
        ]
        cursor.executemany(update_sql, update_values)

    if not insert_rows.empty:
        insert_sql = """
            INSERT INTO crossing_groups (
                port_id, crossing_name, hours, construction_notice, operational_status
            )
            VALUES (%s, %s, %s, %s, %s)
        """
        insert_values = [
            (
                row["port_id"],
                row["crossing_name"],
                row.get("hours"),
                row.get("construction_notice"),
                row.get("operational_status"),
            )
            for _, row in insert_rows.iterrows()
        ]
        cursor.executemany(insert_sql, insert_values)

    connection.commit()
    print("crossing_groups table updated successfully.")


def clear_lane_construction_notices(connection, cursor):
    try:
        cursor.execute(
            """
            UPDATE lanes AS l
            SET construction_notice = NULL
            FROM crossing_groups AS cg
            WHERE l.crossing_group_id = cg.id
              AND l.construction_notice IS NOT NULL
              AND cg.construction_notice IS NOT NULL
              AND trim(lower(l.construction_notice)) = trim(lower(cg.construction_notice))
            """
        )
        connection.commit()
        print("Lane construction notices cleared where redundant.")
    except Exception as e:
        connection.rollback()
        print(f"Error clearing lane construction notices: {e}")


def attach_crossing_group_ids(final_df: pd.DataFrame, connection) -> pd.DataFrame:
    if "port_id" not in final_df.columns or "crossing_name" not in final_df.columns:
        final_df = final_df.copy()
        final_df["crossing_group_id"] = None
        return final_df

    groups_df = pd.read_sql("SELECT id, port_id, crossing_name FROM crossing_groups", connection)
    if groups_df.empty:
        final_df = final_df.copy()
        final_df["crossing_group_id"] = None
        return final_df

    groups_df["crossing_name_norm"] = _normalized_series(groups_df["crossing_name"])
    final_df = final_df.copy()
    final_df["crossing_name_norm"] = _normalized_series(final_df["crossing_name"])
    final_df = final_df.merge(
        groups_df[["id", "port_id", "crossing_name_norm"]],
        on=["port_id", "crossing_name_norm"],
        how="left",
    )
    final_df.rename(columns={"id": "crossing_group_id"}, inplace=True)
    final_df.drop(columns=["crossing_name_norm"], inplace=True)
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


def update_lanes_table(connection, cursor, final_df: pd.DataFrame):
    """Update lanes table using non-null values from the latest feed."""
    try:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'lanes'
            ORDER BY ordinal_position
            """
        )
        lanes_columns = [row[0] for row in cursor.fetchall()]

        if "port_id" not in final_df.columns:
            final_df = attach_port_ids(final_df, connection)
        if "crossing_group_id" not in final_df.columns:
            final_df = final_df.copy()
            final_df["crossing_group_id"] = None

        lanes_df = pd.read_sql("SELECT * FROM lanes", connection)
        for col in lanes_columns:
            if col not in lanes_df.columns:
                lanes_df[col] = None
        lanes_df = lanes_df[[c for c in lanes_columns if c in lanes_df.columns]]
        for col in ["crossing_name", "port_id", "lane_type", "lane_subtype"]:
            lanes_df[col] = lanes_df[col].astype(str).str.strip().str.lower()
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).str.strip().str.lower()
            else:
                final_df[col] = None
        final_df = final_df[final_df["port_id"].notnull() & (final_df["port_id"].astype(str).str.lower() != "none")]
        lanes_df["merge_key"] = lanes_df["crossing_name"] + "|" + lanes_df["port_id"] + "|" + lanes_df["lane_type"] + "|" + lanes_df["lane_subtype"]
        final_df["merge_key"] = final_df["crossing_name"] + "|" + final_df["port_id"] + "|" + final_df["lane_type"] + "|" + final_df["lane_subtype"]
        lanes_df.set_index("merge_key", inplace=True)
        final_df.set_index("merge_key", inplace=True)
        update_cols = [
            "lanes_open",
            "delay_minutes",
            "time",
            "time_zone",
            "max_lanes",
            "update_time",
            "operational_status",
            "hours",
            "crossing_group_id",
        ]
        update_cols = [col for col in update_cols if col in lanes_columns]
        common_keys = lanes_df.index.intersection(final_df.index)
        for col in update_cols:
            lanes_df[col] = lanes_df[col].astype("object")
        for key in common_keys:
            for col in update_cols:
                if col in final_df.columns:
                    val = final_df.at[key, col]
                    if val is not None and (not (isinstance(val, float) and pd.isnull(val))) and str(val).strip() != "":
                        lanes_df.at[key, col] = val
        int_columns = ["lanes_open", "delay_minutes", "max_lanes"]
        for col in int_columns:
            if col in lanes_df.columns:
                lanes_df[col] = pd.to_numeric(lanes_df[col], errors="coerce").astype("Int64")
        lanes_df.reset_index(drop=True, inplace=True)
        lanes_df = lanes_df.reindex(columns=lanes_columns)
        lanes_df = lanes_df.where(pd.notnull(lanes_df), None)
        temp_table = "lanes_temp_update"
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table};")
        cursor.execute(f"CREATE TABLE {temp_table} (LIKE lanes INCLUDING ALL);")
        from io import StringIO

        sio = StringIO()
        lanes_df.to_csv(sio, header=False, index=False, na_rep="\\N")
        sio.seek(0)
        copy_sql = f"COPY {temp_table} ({', '.join(lanes_columns)}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        cursor.copy_expert(copy_sql, sio)
        cursor.execute("BEGIN;")
        cursor.execute("DELETE FROM lanes;")
        cursor.execute(f"INSERT INTO lanes SELECT * FROM {temp_table};")
        cursor.execute(f"DROP TABLE {temp_table};")
        cursor.execute("COMMIT;")
        print("lanes table updated successfully (merged with non-null values from final_df).")
    except Exception as e:
        connection.rollback()
        print(f"Error al actualizar lanes (overwrite non-null values): {e}")


def run():
    xml_root = fetch_xml_root(RSS_URL)
    tree = ET.ElementTree(xml_root)
    root = tree.getroot()

    df = parse_ports_to_dataframe(root)
    standard_df = build_standard_lane_dataframe(df)
    special_dfs = build_special_lane_dataframes(df)
    final_df = pd.concat([standard_df] + special_dfs, ignore_index=True)
    normalized_df = normalize_dataframe(final_df)

    if ENV != "prod":
        print(normalized_df.head())

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

        insert_cruces_fronterizos(connection, cursor, normalized_df)
        final_df = attach_port_ids(normalized_df, connection)
        group_df = build_crossing_groups_dataframe(final_df)
        upsert_crossing_groups_table(connection, cursor, group_df)
        final_df = attach_crossing_group_ids(final_df, connection)
        update_lanes_table(connection, cursor, final_df)
        clear_lane_construction_notices(connection, cursor)

        cursor.close()
        connection.close()
        print("Conexión cerrada.")
    except Exception as e:
        print(f"Error general de conexión o ejecución: {e}")


if __name__ == "__main__":
    run()
