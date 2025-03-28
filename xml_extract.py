import requests
import xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
RSS_URL = os.getenv("RSS_FEED_URL")

# Fetch the XML data
response = requests.get(RSS_URL)
response.raise_for_status()  # Raise an exception for HTTP errors

# Parse the XML data
root = ET.fromstring(response.content)

data = []
# Loop through each port element
for port in root.findall('.//port'):
    # Gather basic port info
    port_data = {
        'port_number': port.find('port_number').text if port.find('port_number') is not None else None,
        'port_name': port.find('port_name').text if port.find('port_name') is not None else None,
        'crossing_name': port.find('crossing_name').text if port.find('crossing_name') is not None else None,
        'hours': port.find('hours').text if port.find('hours') is not None else None,
        'date': port.find('date').text if port.find('date') is not None else None,
        'port_status': port.find('port_status').text if port.find('port_status') is not None else None
    }

    # Example: parse passenger lanes (standard_lanes) data
    passenger_lanes = port.find('passenger_vehicle_lanes')
    if passenger_lanes is not None:
        standard_lanes = passenger_lanes.find('standard_lanes')
        if standard_lanes is not None:
            port_data['passenger_status'] = standard_lanes.find('operational_status').text if standard_lanes.find('operational_status') is not None else None
            port_data['passenger_delay'] = standard_lanes.find('delay_minutes').text if standard_lanes.find('delay_minutes') is not None else None
            port_data['passenger_lanes_open'] = standard_lanes.find('lanes_open').text if standard_lanes.find('lanes_open') is not None else None

    data.append(port_data)

# Create a pandas DataFrame and save to Excel
df = pd.DataFrame(data)
print(df.head())  # Display the first few rows of the DataFrame
#excel_file = 'bwt_data_new.xlsx'
#df.to_excel(excel_file, index=False)
#print(f"Data has been saved to {excel_file}")