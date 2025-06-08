import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load the model to GPU
model = ChatterboxTTS.from_pretrained(device="cuda")

# Narration text (200 words)
text = (
    "In the ever-evolving world of gaming, few moments are as electrifying as a perfectly executed pentakill. "
    "Today, the spotlight is on a legendary team-up that left the entire Rift in awe. Ezreal, with his precision shots, "
    "joined forces with the fiery spirit of Jinx, who brought chaos and rockets to the battlefield. Alongside them were Ahri, "
    "the charming mage whose orbs danced through enemies; Yasuo, the master swordsman whose wind technique sliced through the frontlines; "
    "and Teemo—yes, even Teemo—who laid traps and took down opponents with calculated precision.\n\n"
    "As the game reached its final moments, tension surged. The enemy pushed forward, confident in their lead. "
    "But this dream team had other plans. In a stunning turnaround, each member unleashed their ultimates in perfect sync. "
    "One by one, the enemies fell—stunned, silenced, and wiped from the map. The crowd erupted. A pentakill had just decided the fate of the match.\n\n"
    "From desperation to domination, this epic moment reminded everyone why League of Legends is more than a game—it’s a battle of strategy, "
    "timing, and unmatched synergy. Victory wasn't just earned. It was carved into history, one Nexus explosion at a time."
)

# Generate default voice narration
wav_default = model.generate(text)
ta.save("test-1.wav", wav_default, model.sr)

# Generate voice-cloned narration using your sample
AUDIO_PROMPT_PATH = "audio/sample/sample.wav"
wav_cloned = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav_cloned, model.sr)
