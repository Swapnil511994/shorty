import pandas as pd

# Re-load after code state reset
csv_path = "data/input.csv"
df = pd.read_csv(csv_path)

# Define new Reddit-style prompts
new_prompts = [
    "Write a shocking true confession that starts innocently but ends with an unexpected twist.",
    "Describe a moment when someone realized they were the villain in their own story.",
    "Tell a workplace revenge story that ends with poetic justice in under 30 seconds.",
    "Narrate a 'Today I F*ed Up' moment that spirals out of control hilariously.",
    "Craft a story where someone regrets saying ‘yes’ to something small.",
    "Write a short 'Am I The A**hole?' scenario with a clear plot and moral dilemma.",
    "Generate a chilling but short paranormal encounter that leaves us questioning reality.",
    "Create a story of a kid innocently exposing a family secret during dinner.",
    "Describe a cringey first date story that ends in the most unexpected way.",
    "Write a 'petty revenge' story where the outcome is surprisingly wholesome.",
    
]
# "Generate a high-school drama that involves betrayal, a secret, and karma.",
#     "Write a story about someone eavesdropping and learning something they shouldn’t have.",
#     "Describe a neighbor war that escalated over something ridiculous.",
#     "Write a short tale where a lie told to protect someone ends up destroying trust.",
#     "Create a story where a harmless prank turns into a life lesson."

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
