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

OUTPUT_DIR = Path("../question_generation/babyarc/")

if __name__ == "__main__":
    print("Generating relations")
    template = Path("template.json").read_text()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Delete all files in the output directory before generating new templates
    for file in OUTPUT_DIR.iterdir():
        file.unlink()

    print("Saving templates to {}".format(OUTPUT_DIR))

    for relation, tag in RELATIONS.items():
        tag = "<" + tag + ">"
        for relation2, tag2 in RELATIONS.items():
            print("Generating relations for {} and {}".format(relation, relation2))

            tag2 = "<" + tag2 + ">"

            templated_file = template.replace(
                "%RELATION%", relation).replace("%RELATION_TAG%", tag).replace("%RELATION2%", relation2).replace("%RELATION_TAG2%", tag2)

            with open(OUTPUT_DIR / "{}-{}.json".format(relation, relation2), "w") as f:
                # Use pyjson5 to decode & encode to strip comments, which the
                # default json library used in the question generator doesn't
                # support.
                f.write(pyjson5.encode(pyjson5.decode(templated_file)))
