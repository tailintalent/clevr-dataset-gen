from pathlib import Path
import json
from typing import List
import re
import os
import pickle
import gzip
from copy import deepcopy

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
    
    def load_image(self, filename: str) -> torch.Tensor:
        if filename not in self.image_cache:
            self.image_cache[filename] = self.transform(Image.open(filename))
        return self.image_cache[filename]

    def __init__(self,
                 image_dir: str = None,
                 question_dir: str = None,
                 file: str = None,
                 output_type: str = "full-color",
                 is_easy_dataset: bool = False,
                 stop_at: int = None):
        """
        Initializes the dataset. Pass in either image_dir and question_dir 
        (from which tasks will be synthesized) or pass in the path to a
        file previously saved by save().

        Args:
            output_type: what type of output to give as the output image. If
            full-color, the target will be a full-color rendition of the
            selected object [3x320x240]. If mask only, it will be a mask only
            [1x320x240].
            is_easy_dataset: marks whether this dataset is the "easy" version of CLEVR.
            If so, this uses a special processing algorithm that splits each
            task by 6 to generate more tasks than there are unique task ids.
            stop_at: idx of question to stop at. Useful if you don't want to load
            the whole dataset (which may take a long time)
        
        Documentation on the data format in images/:

        - CLEVR_new_000000.png: the input image
        - CLEVR_new_000000_#.png: full color renders of individual objects
        - CLEVR_new_000000_mask_#.png: masks of individual objects. The
          background is RGB color #404040, while the foreground is some color
          that isn't #404040.
        - CLEVR_new_000000_mask_aio.png: masks of all objects, with occlusion
        """

        assert output_type == "full-color" or output_type == "mask-only"
        
        if file is not None:
            self.tasks = torch.load(file)
        else:
            image_dir, question_dir = Path(image_dir), Path(question_dir)
            questions = json.loads((question_dir / "questions.json").read_text())

            tasks = dict()

            # Assemble the questions into tasks
            for idx, question in enumerate(tqdm(questions["questions"])):
                program = question["program"]
                input_filename = image_dir / question["image_filename"]
                task_str = self.get_unique_task_string(program)

                if task_str in tasks and input_filename in tasks[task_str]["images"]:
                    # Skip, since we have already seen this input image (for diversity!)
                    continue
                
                if not is_easy_dataset and task_str in tasks and tasks[task_str]["count"] >= self.MIN_EXAMPLES_REQUIRED:
                    # Add examples until reaching MIN_EXAMPLES_REQUIRED, unless we are
                    # using easy dataset and can reuse tasks
                    continue

                # Answer idx is the index of a selected object in the image
                answer_object_idx = question["answer"]
                assert type(answer_object_idx) == int

                input = self.load_image(input_filename)
                output_full_color = self.load_image(f"{image_dir}/{question['image']}_{answer_object_idx}.png")
                output_mask_only = self.load_mask_image(f"{image_dir}/{question['image']}_mask_{answer_object_idx}.png")

                if task_str not in tasks:
                    tasks[task_str] = {
                        "count": 0,
                        "inputs": [],
                        "outputs_full_color": [],
                        "outputs_mask_only": [],
                        "task_str": task_str,
                        "questions": [],
                        "images": []
                    }
                
                tasks[task_str]["count"] += 1
                tasks[task_str]["questions"].append(question)
                tasks[task_str]["inputs"].append({
                    "image": input,
                    "question": question
                })
                tasks[task_str]["outputs_full_color"].append(output_full_color)
                tasks[task_str]["outputs_mask_only"].append(output_mask_only)
                tasks[task_str]["images"].append(input_filename)
                
                if stop_at is not None and idx > stop_at:
                    break

            # Only keep tasks with at least MIN_EXAMPLES items
            tasks = {k:v for k, v in tasks.items() if v["count"] >= self.MIN_EXAMPLES_REQUIRED}
            
            self.tasks = list(tasks.values())

            if is_easy_dataset:
                # Extra processing to split tasks by increments of 6, to augment
                # the dataset further
                new_tasks = []
                for task in self.tasks:
                    for start_idx in range(0, task["count"], self.MIN_EXAMPLES_REQUIRED):
                        end_idx = start_idx + self.MIN_EXAMPLES_REQUIRED
                        if end_idx >= task["count"]:
                            break

                        new_tasks.append({
                            "count": self.MIN_EXAMPLES_REQUIRED,
                            "inputs": task["inputs"][start_idx:end_idx],
                            "outputs_full_color": task["outputs_full_color"][start_idx:end_idx],
                            "outputs_mask_only": task["outputs_mask_only"][start_idx:end_idx],
                            "task_str": task["task_str"],
                            "questions": task["questions"][start_idx:end_idx],
                            "images": task["images"][start_idx:end_idx],
                        })
                print("Dataset extra processing concluded.")
                self.tasks = new_tasks
        
        self.output_type = output_type
        print("Loaded", len(self.tasks), "tasks.")
    
    def save(self, filename: str):
        torch.save(self.tasks, filename)
    
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

    def load_mask_image(self, filename: str):
        image = self.load_image(filename)
        image = (image - 0x40/255) != 0 # Remove background and binarize
        # Delete alpha layer and OR the RGB layers together
        image = image[0:3].sum(axis=0, keepdims=True).to(torch.bool)
        return image

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = deepcopy(self.tasks[idx])

        # Handle output_type
        if not self.output_type == "full-color":
            del task["outputs_full_color"] # To save space
        
        task["outputs"] = task["outputs_mask_only"]

        # Delete variable-length lists in the dataset that break collating when
        # batch_size > 1
        del task["images"]
        del task["questions"]
        for inp in task["inputs"]:
            del inp["question"]
        
        # Make the last example a test input/output
        task["test_input"] = task["inputs"].pop()
        task["test_output"] = task["outputs"].pop()

        return task


def create_full_dataset(output_type: str = "mask-only"):
    """
    Loads the full dataset from disk and splits it into a train/val/test split, making up
    2/3, 1/6, and 1/6 of the dataset, respectively.

    Returns:
        train_set, val_set, test_set - a tuple of torch.util.data.Datasets
    """
    dataset = ClevrRelationDataset(
        file="/dfs/user/tailin/.results/CLEVR_relation/relations-dataset-2021-08-18-608-tasks.pt",
        output_type=output_type)
    return torch.utils.data.random_split(
        dataset,
        [len(dataset) * 2//3, len(dataset) * 1//6, len(dataset) * 1//6 + 1],
        generator=torch.Generator().manual_seed(42))


def create_easy_dataset(output_type: str = "mask-only"):
    dataset = ClevrRelationDataset(
        file="/dfs/user/tailin/.results/CLEVR_relation/relations-dataset-easy-2021-09-16-461-tasks2.pt",
        output_type=output_type, is_easy_dataset=True)
    return torch.utils.data.random_split(
        dataset,
        [len(dataset) * 2//3, len(dataset) * 1//6, len(dataset) * 1//6 + 2],
        generator=torch.Generator().manual_seed(42))
    