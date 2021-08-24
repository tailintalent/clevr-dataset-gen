"""
Generates relation templates from the template.json file and saves them in
OUTPUT_DIR.
"""

import os
from pathlib import Path
import pyjson5


# List of relations to generate templates for.
# The corresponding placeholder variable <Z>, <M>, etc. is included because
# we have to also template the constraint that this variable must be null.
RELATIONS = {
    "same_shape": "Z",
    "same_material": "M",
    "same_size": "S",
    "same_color": "C"
}

OBJECTS = [
    ["large", "blue", "metal", "cylinder"],
    ["small", "red", "metal", "sphere"],
    ["large", "green", "rubber", "cube"],
    ["small", "brown", "rubber", "cylinder"],
    ["large", "purple", "metal", "sphere"],
    ["small", "cyan", "rubber", "cube"]
]

OUTPUT_DIR = Path("../question_generation/babyarc_easy/")

if __name__ == "__main__":
    print("Generating relations")
    template = Path("babyarc_easy_template.json").read_text()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Delete all files in the output directory before generating new templates
    for file in OUTPUT_DIR.iterdir():
        file.unlink()

    print("Saving templates to {}".format(OUTPUT_DIR))

    for obj in OBJECTS:
        size, color, material, shape = obj

        constraints = [
            {
                "params": [
                    "<Z>", size
                ],
                "type": "EQ"
            },
            {
                "params": [
                    "<C>", color
                ],
                "type": "EQ"
            },
            {
                "params": [
                    "<M>", material
                ],
                "type": "EQ"
            },
            {
                "params": [
                    "<S>", shape
                ],
                "type": "EQ"
            }
        ]

        for relation, tag in RELATIONS.items():
            tag = "<" + tag + ">"
            print("Generating relations for {}".format(relation))

            templated_file = template \
                    .replace("%RELATION%", relation) \
                    .replace("%CONSTRAINTS%", pyjson5.encode(constraints))
            
            with open(OUTPUT_DIR / "{}-{}.json".format(relation, pyjson5.encode(obj)), "w") as f:
                # Use pyjson5 to decode & encode to strip comments, which the
                # default json library used in the question generator doesn't
                # support.
                f.write(pyjson5.encode(pyjson5.decode(templated_file)))
