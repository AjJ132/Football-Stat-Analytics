import json

def extract_keys_for_position(json_file_path, position):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Set to store unique keys
    unique_keys = set()

    # Iterate through each player in the data
    for player in data:
        if player.get('position') == position:
            # Add all keys from this player to the set
            unique_keys.update(player.keys())

    return sorted(list(unique_keys))

# Example usage
file_path = 'temp.json'  # Replace with your actual JSON file path
position = 'WR'  # Change this to 'WR' when you have data with WR positions

distinct_keys = extract_keys_for_position(file_path, position)

print(f"Distinct keys for position '{position}':")
for key in distinct_keys:
    print(key)