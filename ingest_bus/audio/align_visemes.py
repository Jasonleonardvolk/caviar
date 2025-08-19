import os
import json
import subprocess

PHONEME_TO_VISEME = {
    "AA": 1, "AE": 1, "AH": 1, "AO": 1, "AW": 2, "AY": 2,
    "B": 3, "CH": 4, "D": 5, "DH": 5, "EH": 1, "ER": 6, "EY": 1,
    "F": 7, "G": 5, "HH": 8, "IH": 1, "IY": 1, "JH": 4, "K": 5,
    "L": 6, "M": 3, "N": 5, "NG": 5, "OW": 1, "OY": 2, "P": 3,
    "R": 6, "S": 4, "SH": 4, "T": 5, "TH": 7, "UH": 1, "UW": 1,
    "V": 7, "W": 2, "Y": 2, "Z": 4, "ZH": 4
}

ALIGNMENT_OUT = "output/align.json"

def run_mfa(audio_path: str, text: str, output_json: str):
    with open("temp.trans.txt", "w") as f:
        f.write(f"enola {text}\n")
    subprocess.run([
        "mfa", "align", audio_path, "temp.trans.txt",
        "english", "output", "--output_format", "json"
    ])
    os.rename("output/enola/align.json", output_json)
    print(f"✅ Alignment written to {output_json}")

def convert_to_viseme_sequence(json_path: str, out_path: str):
    with open(json_path) as f:
        data = json.load(f)
    visemes = []
    for segment in data["words"]:
        for p in segment["phones"]:
            phoneme = p["phone"].split("_")[0]
            start = float(p["start_time"])
            end = float(p["end_time"])
            viseme_id = PHONEME_TO_VISEME.get(phoneme, 0)
            visemes.append({"viseme": viseme_id, "start": start, "end": end})
    with open(out_path, "w") as f:
        json.dump(visemes, f, indent=2)
    print(f"✅ Viseme map saved to {out_path}")

if __name__ == "__main__":
    run_mfa("output/enola.wav", "ENOLA reporting. Let's investigate this.", ALIGNMENT_OUT)
    convert_to_viseme_sequence(ALIGNMENT_OUT, "output/viseme_map.json")