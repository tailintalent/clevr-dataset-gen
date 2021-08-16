from pathlib import Path
import json
from typing import List
import re
import os
import pickle
import gzip

from tqdm import tqdm
from torchvision import transforms
import torch
from PIL import Image


RELATIONS = {
    "same_shape": "Z",
    "same_material": "M",
    "same_size": "S",
    "same_color": "C"
}

def reglob(path, exp, invert=False):
    """glob.glob() style searching which uses regex

    :param exp: Regex expression for filename
    :param invert: Invert match to non matching files
    """

    m = re.compile(exp)

    if invert is False:
        res = [f for f in os.listdir(path) if m.search(f)]
    else:
        res = [f for f in os.listdir(path) if not m.search(f)]

    res = map(lambda x: "%s/%s" % ( path, x, ), res)
    return res


class ClevrRelationDataset(torch.utils.data.Dataset):
    """
    CLEVR with relations dataset. This dataset generates "fake" ARC tasks using
    the CLEVR object engine.

    Each sample is structured as follows:
    - count: # of tasks
    - inputs: 5x input images of shape
        - data: 3x320x240 tensor
        - mask: Nx320x240 tensor of masks, where N is the number of objects
        - question: question metadata, including program
    - outputs: 5x output images, each a 3x320x240 tensor of the single object
      selected as the refer node.
    - test_input: input image of the same shape as inputs
    - test_output: output image of the same shape as outputs
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    MIN_EXAMPLES_REQUIRED = 6
    
    image_cache = dict()
    
    def load_image(self, filename: str):
        if filename not in self.image_cache:
            self.image_cache[filename] = self.transform(Image.open(filename))
        return self.image_cache[filename]

    def __init__(self, image_dir: str = None, question_dir: str = None, pickle_file: str = None):
        """
        Initializes the dataset. Pass in either image_dir and question_dir 
        (from which tasks will be synthesized) or pass in the path to a
        pickle_file previously saved by pickle().
        """
        
        if pickle_file is not None:
            with gzip.GzipFile(pickle_file, 'r') as f:
                self.tasks = pickle.load(f)
        else:
            image_dir, question_dir = Path(image_dir), Path(question_dir)
            questions = json.loads((question_dir / "questions.json").read_text())

            tasks = dict()

            image_cache = dict()
            
            # Assemble the questions into tasks
            for idx, question in enumerate(tqdm(questions["questions"])):
                template_filename = question["template_filename"]
                question_family_index = question["question_family_index"]
                program = question["program"]
                input_filename = image_dir / question["image_filename"]
                mask_filenames = list(sorted(reglob(str(image_dir), f"{question['image']}_[0-9]+.png")))
                task_str = self.get_unique_task_string(program)

                if task_str in tasks and input_filename in tasks[task_str]["images"]:
                    # Skip, since we have already seen this input image (for diversity!)
                    continue

                input = self.load_image(input_filename)
                masks = [self.load_image(filename) for filename in mask_filenames]
                refer_node_mask = masks[question["answer"]]

                if task_str not in tasks:
                    tasks[task_str] = {
                        "count": 0,
                        "inputs": [],
                        "outputs": [],
                        "task_str": task_str,
                        "questions": [],
                        "images": []
                    }

                tasks[task_str]["count"] += 1
                tasks[task_str]["questions"].append(question)
                tasks[task_str]["inputs"].append({
                    "image": input,
                    "mask": masks,
                    "question": question
                })
                tasks[task_str]["outputs"].append(refer_node_mask)
                tasks[task_str]["images"].append(input_filename)
                
                if idx % 100 == 0:
                    print(len({k:v for k, v in tasks.items() if v["count"] >= self.MIN_EXAMPLES_REQUIRED}))

            # Only keep tasks with at least 5 items
            tasks = {k:v for k, v in tasks.items() if v["count"] >= self.MIN_EXAMPLES_REQUIRED}
            
            # Make the last example a test input/output
            for task in tasks:
                task["test_input"] = task["inputs"].pop()
                task["test_output"] = task["outputs"].pop()

            self.tasks = list(tasks.values())
        
        print("Loaded", len(self.tasks), "tasks.")
    
    def pickle(self, filename: str):
        with gzip.GzipFile(filename, 'w', compresslevel=1) as f:
            pickle.dump(self.tasks, f)
    
    @staticmethod
    def get_unique_task_string(program: List[str]):
        """
        Parses the program for a given question and returns a unique string that identifies the 
        babyARC task that it embodies.

        This function is somewhat hacky in that it doesn't deal with the AST directly, but it
        works for the generated babyARC template programs.
        """
        inputs = []
        object_str = []
        for node in program:
            # Generate a new object str every time we see a new "scene" (which implies
            # a new object)
            if node["type"] == "scene":
                if len(object_str) != 0:
                    inputs.append(",".join(object_str))
                    object_str = []
                continue

            # If we're not at a scene, then we're in the middle of an object
            if node["type"].startswith("filter_"):
                # This node filters some property of the input. Let's consider it.
                object_str.append(node["type"][7:] + "=" + node["value_inputs"][0])
        inputs.append(",".join(object_str))
        relations = [node["type"]
                        for node in program if node["type"] in RELATIONS]
        
        assert len(inputs) == len(relations)
        combined = zip(relations, inputs)
        return "+".join(sorted([relation + "-" + input for relation, input in combined]))


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]
