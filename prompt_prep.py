import pandas as pd

# Re-load after code state reset
csv_path = "data/input.csv"
df = pd.read_csv(csv_path)

# Define new Reddit-style prompts
new_prompts = [
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]


# Replace existing prompts with the new ones
num_prompts = min(len(df), len(new_prompts))
df.loc[:num_prompts - 1, 'Prompt'] = new_prompts[:num_prompts]

# Clear unused prompts if any
if len(df) > num_prompts:
    df.loc[num_prompts:, 'Prompt'] = ""

# Save updated CSV
output_path = "data/input.csv"
df.to_csv(output_path, index=False)

output_path
