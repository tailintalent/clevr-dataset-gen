#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict
import json
import numpy as np
import imageio
from numbers import Number
import pdb
import random
import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from reasoning.pytorch_net.util import Dictionary, first_item, to_cpu_recur, try_call, Printer, transform_dict, MineDataset, is_diagnose, reduce_tensor, get_hashing, pdump, pload, remove_elements, loss_op_core, filter_kwargs, to_Variable, gather_broadcast, get_pdict, COLOR_LIST, set_seed, Zip, Early_Stopping, init_args, make_dir, str2bool, get_filename_short, get_machine_name, get_device, record_data, plot_matrices, filter_filename, get_next_available_key, to_np_array, to_Variable, get_filename_short, write_to_config, Dictionary, Batch, to_cpu
from reasoning.util import color_dict, clip_grad, identity_fun, seperate_concept, to_one_hot, onehot_to_RGB, get_root_dir, get_module_parameters, assign_embedding_value, get_hashing, to_device_recur, visualize_matrices, repeat_n, mask_iou_score, shrink, get_obj_from_mask


# In[ ]:


def get_mask_filename(image_filaname, image_filenames, include_aio=False):
    string = image_filename.split(".png")[0]
    if include_aio:
        return [filename for filename in image_filenames if "mask" in filename and string in filename]
    else:
        return [filename for filename in image_filenames if "mask" in filename and string in filename and not "aio" in filename]

def get_image(dirname, image_filename, resize=None, is_square=True, antialias=False):
    img = torch.FloatTensor(imageio.imread(dirname + "images/" + image_filename).transpose(2,0,1)[:3]/255)
    if is_square:
        img = img[...,40:-40]
    if resize is None:
        return img
    else:
        return F.interpolate(img[None], size=resize, mode='bilinear' if antialias else "nearest", antialias=antialias)[0]

def get_mask(mask_raw):
    return ((mask_raw - 64/255) > 1e-5).any(0)[None].float()

def get_image_and_mask(
    chosen_filename,
    obj_id,
    n_objs,
    dirname,
    image_filenames,
    resize=None,
    is_square=True,
    isplot=False,
    check_square_oob="None",
):
    img = get_image(dirname, chosen_filename+".png", resize=resize, is_square=is_square, antialias=True)
    if isplot:
        visualize_matrices([img], use_color_dict=False)
    masks = []
    is_valid = True
    if check_square_oob != "None":
        assert check_square_oob == "all"
        mask_list = []
        for id in range(n_objs):
            mask_filename = chosen_filename + f"_mask_{id}.png"
            mask_raw = get_image(dirname, mask_filename, resize=None, is_square=False, antialias=False)
            mask = get_mask(mask_raw)
            if (mask[..., :40] > 0).any() or (mask[..., -40:] > 0).any():
                is_valid = False
                break
            # processing:
            if is_square:
                mask_raw = mask_raw[...,40:-40]
            if resize is not None:
                mask_raw = F.interpolate(mask_raw[None], size=resize, mode="nearest", antialias=False)[0]
            mask = get_mask(mask_raw)
            mask_list.append(mask)

    if is_valid:
        mask = mask_list[obj_id]
        if isplot:
            plot_matrices(mask, images_per_row=6)
    else:
        img, mask, mask_list = None, None, None
    return img, mask, mask_list


def get_clevr_concept_data_core(filter_dict, dirname, resize=(60,60), n_examples=None, image_filenames=None, isplot=False):
    assert len(filter_dict) == 1
    concept = get_cap(first_item(filter_dict))
    json_filenames = sorted(filter_filename(dirname + "scenes"))
    chosen_filenames = []
    data_list = []
    for k, json_filename in enumerate(json_filenames):
        meta = json.load(open(dirname + "scenes/" + json_filename))
        objects = meta["objects"]
        objs_valid_list = []
        for i, obj in enumerate(objects):
            is_chosen = True
            for key, value in filter_dict.items():
                if obj[key] != value:
                    is_chosen = False
                    break
            if is_chosen:
                objs_valid_list.append(i)
        if len(objs_valid_list) > 0:
            if isplot:
                print(f"{k}:")
            chosen_filename = json_filename.split(".json")[0]
            obj_id = np.random.choice(objs_valid_list)
            img, mask, mask_list = get_image_and_mask(
                chosen_filename, obj_id, 
                n_objs=len(objects),
                dirname=dirname,
                image_filenames=image_filenames,
                resize=resize,
                check_square_oob="all",
                isplot=isplot,
            )
            if mask is None:
                continue
            chosen_filenames.append((chosen_filename, obj_id))
            obj_spec = get_obj_spec(objects)
            node_id_map = OrderedDict({
                f"obj_{i}": i for i in range(len(mask_list))
            })
            id_object_mask = OrderedDict({i: mask_ele for i, mask_ele in enumerate(mask_list)})
            info = Dictionary({
                "dirname": dirname,
                "chosen_filename": chosen_filename,
                "obj_id": obj_id,
                "meta": meta,
                "obj_spec": obj_spec,
                "node_id_map": node_id_map,
                "id_object_mask": id_object_mask,
            })
            if isplot:
                print(obj_spec)
                plot_matrices([ele[0] for ele in mask_list], images_per_row=6)
            data = (
                img,
                (mask,),
                concept,
                info,
            )
            data_list.append(data)
            if len(data_list) % 100 == 0 or len(data_list) == n_examples:
                print(len(data_list))
        if n_examples is not None and len(data_list) >= n_examples:
            break
    return data_list


def get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=None, dirname=None):
    if isinstance(canvas_size, Number):
        canvas_size = (canvas_size, canvas_size)
    modes = mode.split("+")
    n_examples_ele = int(np.ceil(n_examples / len(modes)))
    data_list_all = []
    if dirname is None:
        dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    image_filenames = sorted(filter_filename(dirname + "images"))
    for mode_ele in modes:
        print(f"mode: {mode_ele}:")
        filter_dict = {MAP_DICT[mode_ele][0]: MAP_DICT[mode_ele][1]}
        data_list = get_clevr_concept_data_core(
            filter_dict,
            dirname,
            resize=canvas_size,
            n_examples=n_examples_ele,
            image_filenames=image_filenames,
            isplot=False,
        )
        data_list_all += data_list
    data_list_all = data_list_all[:n_examples]
    random.shuffle(data_list_all)
    return data_list_all


def get_cap(string):
    return string[0].upper() + string[1:]


def get_obj_spec(objects):
    obj_spec = []
    for i, obj in enumerate(objects):
        types = f'{get_cap(obj["color"])}+{get_cap(obj["shape"])}+{get_cap(obj["size"])}+{get_cap(obj["material"])}'
        obj_spec_ele = [(f"obj_{i}", f"{types}_[-1]"), "Attr"]
        obj_spec.append(obj_spec_ele)
    return obj_spec


MAP_DICT = {
    "Red": ("color", "red"),
    "Green": ("color", "green"),
    "Blue": ("color", "blue"),
    "Cube": ("shape", "cube"),
    "Cylinder": ("shape", "cylinder"),
    "Large": ("size", "large"),
    "Small": ("size", "small"),
}


# In[ ]:


# if __name__ == "__main__":
#     mode = "Red+Green+Blue+Cube+Cylinder+Large+Small"
#     dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
#     data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=440, dirname=dirname)
#     pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_{64}_ex_{20000}_1.p")


# In[ ]:


# filter_dict = {"color": "red"}
# dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
# resize=(60,60)
# n_examples=100
# image_filenames = sorted(filter_filename(dirname + "images"))
# isplot=True


# In[ ]:


if __name__ == "__main__":
    mode = "Red+Green+Blue+Cube+Cylinder+Large+Small"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=25000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_{64}_ex_{25000}_1.p")


# In[ ]:


if __name__ == "__main__":
    mode = "Red+Green+Blue+Cube+Cylinder+Large+Small"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-60000-130000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=25000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_{64}_ex_{25000}_2.p")


# In[ ]:


if __name__ == "__main__":
    mode = "Red+Green+Blue+Cube+Cylinder+Large+Small"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-130000-150000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=5000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_{64}_ex_{5000}_3.p")


# In[ ]:


if __name__ == "__main__":
    isplot = True
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    json_filenames = sorted(filter_filename(dirname + "scenes"))
    image_filenames = sorted(filter_filename(dirname + "images"))
    filter_dict = {
        "color": "red",  # "red", "green", "blue",
        # "shape": "cube", # "cube", "cylinder"
        # "size": "big",   # "big", "small"
    }
    data_list = get_clevr_concept_data_core(filter_dict, dirname, resize=(64,64), n_examples=100, isplot=False)


# In[ ]:


"""
Steps:
1. Able to load image, mask and corresponding json property (10min)
2. Write a function that given any concept, prepare concept dataset (1h)
3. Write a function that given any relation, prepare relation dataset (0.5h)
4. Training concept dataset, begin by testing overfitting (1h)
5. Train relation dataset, begin by overfitting (0.5h)
6. Validation dataset
7. Baseline
"""

