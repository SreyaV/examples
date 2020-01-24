import os 
import json
print(os.environ)

data = {
    "subscription_id": "ad203158-bc5d-4e72-b764-2607833a71dc",
    "resource_group": "akannava",
    "workspace_name": "akannava"
}

with open('config.json', 'w') as outfile:
    json.dump(data, outfile)