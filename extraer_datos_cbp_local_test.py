import requests
import pandas as pd
import xml.etree.ElementTree as ET
import pytz
from datetime import datetime
import os

# For local testing, allow user to specify XML file or use RSS URL
def get_xml_content():
    xml_path = input("Enter path to XML file for local test (leave blank to use RSS_FEED_URL): ").strip()
    if xml_path:
        with open(xml_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        from dotenv import load_dotenv
        load_dotenv()
        RSS_URL = os.getenv("RSS_FEED_URL")
        response = requests.get(RSS_URL)
        response.raise_for_status()
        return response.content

xml_content = get_xml_content()
xml_file = ET.fromstring(xml_content)

# Analizar el archivo XML
root = ET.ElementTree(xml_file).getroot()

def parse_element(element, parent_tag=""):
    data = {}
    for child in element:
        tag_name = f"{parent_tag}.{child.tag}" if parent_tag else child.tag
        if len(child) > 0:
            data.update(parse_element(child, tag_name))
        else:
            data[tag_name] = child.text if child.text is not None else ""
    return data

data = []
for port in root.findall(".//port"):
    port_data = parse_element(port)
    data.append(port_data)

df = pd.DataFrame(data)
CONTEXT_COLUMNS = [
    'port_number', 'border', 'port_name', 'crossing_name',
    'hours', 'date', 'port_status', 'construction_notice'
]

def read_csv_with_fallback(path, **kwargs):
    """Try a handful of encodings so local CSVs with ANSI characters still load."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise last_err  # Bubble up the final error if all encodings fail

def pivot_col(datafram, piv_columns, id_cols, col_name, col_value):
    piv_df = datafram.melt(
        id_vars=id_cols,
        value_vars=piv_columns.keys(),
        var_name=col_name,
        value_name=col_value
    )
    piv_df[col_name] = piv_df[col_name].map(piv_columns)
    return piv_df

# --- Standard lanes pivots ---
pivot_columns = {
    'commercial_automation_type': 'commercial',
    'passenger_automation_type': 'passenger',
    'pedestrain_automation_type': 'pedestrian'
}
id_columns=['port_number', 'border', 'port_name', 'crossing_name', 'hours', 'date', 'port_status', 'construction_notice']
column_name='lane_type'
column_value='automation_type'
df1=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

pivot_columns = {
    'commercial_vehicle_lanes.maximum_lanes': 'commercial',
    'passenger_vehicle_lanes.maximum_lanes': 'passenger',
    'pedestrian_lanes.maximum_lanes': 'pedestrian'
}
id_columns=['port_number']
column_value='max_lanes'
df2=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.update_time': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.update_time': 'passenger',
    'pedestrian_lanes.standard_lanes.update_time': 'pedestrian'
}
column_value='update_time'
df3=pivot_col(df, pivot_columns, id_columns, column_name, column_value)
df3['lane_subtype'] = 'standard'

pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.lanes_open': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.lanes_open': 'passenger',
    'pedestrian_lanes.standard_lanes.lanes_open': 'pedestrian'
}
column_value='lanes_open'
df4=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.operational_status': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.operational_status': 'passenger',
    'pedestrian_lanes.standard_lanes.operational_status': 'pedestrian'
}
column_value='operational_status'
df5=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

pivot_columns = {
    'commercial_vehicle_lanes.standard_lanes.delay_minutes': 'commercial',
    'passenger_vehicle_lanes.standard_lanes.delay_minutes': 'passenger',
    'pedestrian_lanes.standard_lanes.delay_minutes': 'pedestrian'
}
column_value='delay_minutes'
df6=pivot_col(df, pivot_columns, id_columns, column_name, column_value)

merged_df = pd.merge(df1, df2, on=['port_number', 'lane_type'], how='left')
merged1_df = pd.merge(merged_df, df3, on=['port_number', 'lane_type'], how='left')
merged2_df = pd.merge(merged1_df, df4, on=['port_number', 'lane_type'], how='left')
merged3_df = pd.merge(merged2_df, df5, on=['port_number', 'lane_type'], how='left')
merged4_df = pd.merge(merged3_df, df6, on=['port_number', 'lane_type'], how='left')

# --- Special lanes ---
special_lanes = [
    ('commercial', 'commercial_vehicle_lanes', 'FAST_lanes', 'FAST lanes'),
    ('passenger', 'passenger_vehicle_lanes', 'NEXUS_SENTRI_lanes', 'NEXUS SENTRI lane'),
    ('passenger', 'passenger_vehicle_lanes', 'ready_lanes', 'ready_lanes'),
    ('pedestrian', 'pedestrian_lanes', 'ready_lanes', 'ready_lanes')
]
special_dfs = []
context_cols = [c for c in CONTEXT_COLUMNS if c in df.columns]
for lane_type, prefix, key, subtype in special_lanes:
    metric_cols = [
        f'{prefix}.{key}.operational_status',
        f'{prefix}.{key}.update_time',
        f'{prefix}.{key}.delay_minutes',
        f'{prefix}.{key}.lanes_open'
    ]
    cols = context_cols + [c for c in metric_cols if c in df.columns]
    if len(cols) <= len(context_cols):
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
    if 'automation_type' not in temp.columns:
        temp['automation_type'] = None
    special_dfs.append(temp)

final_df = pd.concat([merged4_df] + special_dfs, ignore_index=True)

final_df['update_time'] = final_df['update_time'].str.replace(r'At Noon', 'At 12:00 pm', regex=True)
final_df[['time', 'time_zone']] = final_df['update_time'].str.extract(r'At (\d{1,2}:\d{2} (?:am|pm)) (\w{3})')
final_df['time'] = final_df['time'].str.replace(r'^0:', '12:', regex=True)
final_df['time'] = pd.to_datetime(final_df['time'], format='%I:%M %p').dt.time

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

def convert_to_mdt(row):
    try:
        local_tz_name = time_zone_mapping.get(row['time_zone'], None)
        if not local_tz_name:
            return None
        local_tz = pytz.timezone(local_tz_name)
        mdt_tz = pytz.timezone('US/Mountain')
        local_time = local_tz.localize(datetime.combine(datetime.today(), row['time']))
        mdt_time = local_time.astimezone(mdt_tz)
        return mdt_time.strftime('%H:%M:%S')
    except Exception as e:
        print(f"Error al convertir la hora: {e}")
        return None

final_df['time_mdt'] = final_df.apply(convert_to_mdt, axis=1)
final_df['time'] = final_df['time'].apply(lambda x: None if pd.isnull(x) else x)
final_df['time_mdt'] = final_df['time_mdt'].apply(lambda x: None if pd.isnull(x) else x)
final_df.replace("", None, inplace=True)
final_df = final_df.where(pd.notnull(final_df), None)

print(final_df.head())

# Save to CSV for local inspection
def get_csv_path():
    default = 'csv_files/rss_local_test.csv'
    path = input(f"Enter CSV output path (default: {default}): ").strip()
    return path if path else default

csv_file = get_csv_path()
final_df.to_csv(csv_file, index=False, encoding="utf-8")
print(f"CSV saved to {csv_file}")

# --- Add port_id to final_df by merging with puertos.csv ---
puertos_csv_path = '../Supabase/puertos.csv'
puertos_df = read_csv_with_fallback(puertos_csv_path)
puertos_df['port_name'] = puertos_df['port_name'].astype(str).str.strip()
puertos_df['border'] = puertos_df['border'].astype(str).str.strip()
if 'port_name' in final_df.columns and 'border' in final_df.columns:
    final_df['port_name'] = final_df['port_name'].astype(str).str.strip()
    final_df['border'] = final_df['border'].astype(str).str.strip()
    final_df = pd.merge(final_df, puertos_df[['id', 'port_name', 'border']], on=['port_name', 'border'], how='left')
    final_df.rename(columns={'id': 'port_id'}, inplace=True)
else:
    # If port_name/border are missing, fill with None for compatibility
    final_df['port_id'] = None

# --- Simulate cruces table update for local test ---
cruces_csv_path = '../Supabase/cruces.csv'
cruces_df = read_csv_with_fallback(cruces_csv_path)
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
# Only update rows where the merge key exists in both DataFrames
common_keys = cruces_df.index.intersection(final_df.index)
# Ensure update columns are objects so pandas lets us mix str/numbers safely
for col in update_cols:
    if col in cruces_df.columns:
        cruces_df[col] = cruces_df[col].astype('object')
for key in common_keys:
    for col in update_cols:
        if col in final_df.columns:
            val = final_df.at[key, col]
            if val is not None and (not (isinstance(val, float) and pd.isnull(val))) and str(val).strip() != '':
                cruces_df.at[key, col] = val
# Align numeric fields with the production script behaviour
int_columns = ['lanes_open', 'delay_minutes', 'max_lanes']
for col in int_columns:
    if col in cruces_df.columns:
        cruces_df[col] = pd.to_numeric(cruces_df[col], errors='coerce').astype('Int64')
cruces_df.reset_index(drop=True, inplace=True)
updated_cruces_csv = 'csv_files/cruces_updated_local_test.csv'
cruces_df.to_csv(updated_cruces_csv, index=False, encoding='utf-8')
print(f"Updated cruces table saved to {updated_cruces_csv}")
