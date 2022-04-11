#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict
import json
from itertools import product
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from networkx import DiGraph
from networkx import line_graph
import imageio
from numbers import Number
from itertools import product
import pdb
import random
import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from reasoning.pytorch_net.util import get_matching_mapping, remove_duplicates, to_line_graph, draw_nx_graph, get_nx_graph, to_nx_graph, get_all_graphs, Dictionary, first_item, to_cpu_recur, try_call, Printer, transform_dict, MineDataset, is_diagnose, reduce_tensor, get_hashing, pdump, pload, remove_elements, loss_op_core, filter_kwargs, to_Variable, gather_broadcast, get_pdict, COLOR_LIST, set_seed, Zip, Early_Stopping, init_args, make_dir, str2bool, get_filename_short, get_machine_name, get_device, record_data, plot_matrices, filter_filename, get_next_available_key, to_np_array, to_Variable, get_filename_short, write_to_config, Dictionary, Batch, to_cpu
from reasoning.util import color_dict, clip_grad, identity_fun, seperate_concept, to_one_hot, onehot_to_RGB, get_root_dir, get_module_parameters, assign_embedding_value, get_hashing, to_device_recur, visualize_matrices, repeat_n, mask_iou_score, shrink, get_obj_from_mask


# ### Helper functions:

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
    if not isinstance(obj_id, list) and not isinstance(obj_id, tuple):
        obj_id = [obj_id]
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
        masks = [mask_list[id] for id in obj_id]
        if isplot:
            plot_matrices([ele[0] for ele in masks], images_per_row=6)
    else:
        img, masks, mask_list = None, None, None
    return img, masks, mask_list


def get_all_relations(objects):
    Dict = {"SameColor": "color", "SameShape": "shape", "SameSize": "size"}
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i < j:
                for relation in ["SameColor", "SameShape", "SameSize"]:
                    key = Dict[relation]
                    if obj1[key] == obj2[key]:
                        relations.append((i, j, relation))
    return relations


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
            img, masks, mask_list = get_image_and_mask(
                chosen_filename, obj_id, 
                n_objs=len(objects),
                dirname=dirname,
                image_filenames=image_filenames,
                resize=resize,
                check_square_oob="all",
                isplot=isplot,
            )
            if masks is None:
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
                tuple(masks),
                concept,
                info,
            )
            data_list.append(data)
            if len(data_list) % 100 == 0 or len(data_list) == n_examples:
                print(len(data_list))
        if n_examples is not None and len(data_list) >= n_examples:
            break
    return data_list


def get_clevr_relation_data_core(relation, dirname, resize=(64,64), n_examples=None, image_filenames=None, isplot=False):
    json_filenames = sorted(filter_filename(dirname + "scenes"))
    chosen_filenames = []
    data_list = []
    Dict = {"SameColor": "color", "SameShape": "shape", "SameSize": "size"}
    for k, json_filename in enumerate(json_filenames):
        meta = json.load(open(dirname + "scenes/" + json_filename))
        objects = meta["objects"]
        pairs = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i < j:
                    key = Dict[relation]
                    if obj1[key] == obj2[key]:
                        pairs.append((i, j))
        if len(pairs) > 0:
            if isplot:
                print(f"{k}:")
            chosen_filename = json_filename.split(".json")[0]
            obj_ids = pairs[np.random.choice(len(pairs))]
            img, masks, mask_list = get_image_and_mask(
                chosen_filename,
                obj_ids, 
                n_objs=len(objects),
                dirname=dirname,
                image_filenames=image_filenames,
                resize=resize,
                check_square_oob="all",
                isplot=isplot,
            )
            if masks is None:
                continue
            chosen_filenames.append((chosen_filename, obj_ids))
            obj_spec = get_obj_spec(objects)
            node_id_map = OrderedDict({
                f"obj_{i}": i for i in range(len(mask_list))
            })
            id_object_mask = OrderedDict({i: mask_ele for i, mask_ele in enumerate(mask_list)})
            relations = get_all_relations(objects)
            info = Dictionary({
                "dirname": dirname,
                "chosen_filename": chosen_filename,
                "obj_id": obj_ids,
                "meta": meta,
                "obj_spec": obj_spec,
                "node_id_map": node_id_map,
                "id_object_mask": id_object_mask,
                "relations": relations,
            })
            if isplot:
                print(obj_spec)
                plot_matrices([ele[0] for ele in mask_list], images_per_row=6)
            data = (
                img,
                tuple(masks),
                relation,
                info,
            )
            data_list.append(data)
            if len(data_list) % 100 == 0 or len(data_list) == n_examples:
                print(len(data_list))
        if n_examples is not None and len(data_list) >= n_examples:
            break
    return data_list


def get_multigraph_from_obj(objects):
    """
    Args:
        objects, e.g.:
      [{'rotation': 260.9647025538182,
        'size': 'small',
        'material': 'metal',
        'shape': 'cube',
        '3d_coords': [1.4607938528060913, 0.5913009643554688, 0.3499999940395355],
        'color': 'green',
        'pixel_coords': [205, 122, 10.810774803161621]},
        {'rotation': 308.63693063835143,
        'size': 'small',
        'material': 'metal',
        'shape': 'cube',
        '3d_coords': [0.1465933918952942, 1.0083972215652466, 0.3499999940395355],
        'color': 'red',
        'pixel_coords': [186, 104, 11.9275484085083]},
        {'rotation': 261.01205500179,
        'size': 'small',
        'material': 'rubber',
        'shape': 'cube',
        '3d_coords': [-1.4735305309295654, 1.666679859161377, 0.3499999940395355],
        'color': 'green',
        'pixel_coords': [170, 84, 13.383231163024902]}]
    
    Returns:
        graph:
            [(0, ('green', 'cube', 'small')),
             (1, ('red', 'cube', 'small')),
             (2, ('green', 'cube', 'small')),
             ((0, 1), ('SameShape', 'SameSize')),
             ((0, 2), ('SameColor', 'SameShape', 'SameSize')),
             ((1, 2), ('SameShape', 'SameSize')),
            ]
    """
    RELATIONS = ["SameColor", "SameShape", "SameSize"]
    Dict = {"SameColor": "color", "SameShape": "shape", "SameSize": "size"}
    graph = []
    for i, obj in enumerate(objects):
        features = (get_cap(obj["color"]), get_cap(obj["shape"]), get_cap(obj["size"]))
        graph.append((i, features))
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i < j:
                features = []
                for relation in RELATIONS:
                    key = Dict[relation]
                    if obj1[key] == obj2[key]:
                        features.append(relation)
                graph.append(((i, j), tuple(features)))
    return graph


def get_clevr_graph_data_core(graph_key, dirname, resize=(64,64), n_examples=None, image_filenames=None, isplot=False):
    graph = GRAPH_DICT[graph_key]
    json_filenames = sorted(filter_filename(dirname + "scenes"))
    chosen_filenames = []
    data_list = []
    relations = ["SameColor", "SameShape", "SameSize"]
    Dict = {"SameColor": "color", "SameShape": "shape", "SameSize": "size"}
    for k, json_filename in enumerate(json_filenames):
        meta = json.load(open(dirname + "scenes/" + json_filename))
        objects = meta["objects"]
        graph_ex_raw = get_multigraph_from_obj(objects)
        graph_ex_all = get_all_graphs(graph_ex_raw)
        is_valid = False
        reverse_mappings = []
        for graph_ex in graph_ex_all:
            reverse_mapping = get_matching_mapping(graph_ex, graph)
            if len(reverse_mapping) > 0:
                is_valid = True
                reverse_mappings.append(reverse_mapping)
        List = []
        for reverse_mapping in reverse_mappings:
            List += [tuple([(key, value) for key, value in ele.items()]) for ele in reverse_mapping]
        List = remove_duplicates(List)
        reverse_mappings = [dict(ele) for ele in List]
        if is_valid:
            obj_ids = list(reverse_mappings[0].values())
            all_obj_ids = remove_duplicates([tuple(set(list(reverse_mappings[k].values()))) for k in range(len(reverse_mappings))])
            if isplot:
                print(f"{k}:")
            chosen_filename = json_filename.split(".json")[0]
            img, masks, mask_list = get_image_and_mask(
                chosen_filename,
                np.arange(len(objects)).tolist(), 
                n_objs=len(objects),
                dirname=dirname,
                image_filenames=image_filenames,
                resize=resize,
                check_square_oob="all",
                isplot=isplot,
            )
            if masks is None:
                continue
            chosen_filenames.append((chosen_filename, obj_ids))
            obj_spec = get_obj_spec(objects)
            node_id_map = OrderedDict({
                f"obj_{i}": i for i in range(len(mask_list))
            })
            id_object_mask = OrderedDict({i: mask_ele for i, mask_ele in enumerate(mask_list)})
            relations = get_all_relations(objects)
            info = Dictionary({
                "dirname": dirname,
                "chosen_filename": chosen_filename,
                "obj_id": obj_ids,
                "all_obj_ids": all_obj_ids,
                "graph_key": graph_key,
                "graph": graph,
                "reverse_mappings": reverse_mappings,
                "meta": meta,
                "obj_spec": obj_spec,
                "node_id_map": node_id_map,
                "id_object_mask": id_object_mask,
                "relations": relations,
            })
            if isplot:
                print(obj_spec)
                print("reverse_mappings: ", reverse_mappings)
                print("all_obj_ids:", all_obj_ids)
                plot_matrices([ele[0] for ele in mask_list], images_per_row=6)
            data = (
                img,
                tuple(masks),
                graph,
                info,
            )
            data_list.append(data)
            if len(data_list) % 100 == 0 or len(data_list) == n_examples:
                print(len(data_list))
        if n_examples is not None and len(data_list) >= n_examples:
            print(f"Scanned {k} examples")
            break
    return data_list


def get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=None, dirname=None, isplot=False):
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
            isplot=isplot,
        )
        data_list_all += data_list
    data_list_all = data_list_all[:n_examples]
    random.shuffle(data_list_all)
    return data_list_all


def get_clevr_relation_data(mode, canvas_size=(64,64), n_examples=None, dirname=None, isplot=False):
    if isinstance(canvas_size, Number):
        canvas_size = (canvas_size, canvas_size)
    modes = mode.split("+")
    n_examples_ele = int(np.ceil(n_examples / len(modes)))
    data_list_all = []
    if dirname is None:
        dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    image_filenames = sorted(filter_filename(dirname + "images"))
    for relation in modes:
        print(f"mode: {relation}:")
        data_list = get_clevr_relation_data_core(
            relation,
            dirname,
            resize=canvas_size,
            n_examples=n_examples_ele,
            image_filenames=image_filenames,
            isplot=isplot,
        )
        data_list_all += data_list
    data_list_all = data_list_all[:n_examples]
    random.shuffle(data_list_all)
    return data_list_all


def get_clevr_graph_data(mode, canvas_size=(64,64), n_examples=None, dirname=None, isplot=False):
    if isinstance(canvas_size, Number):
        canvas_size = (canvas_size, canvas_size)
    modes = mode.split("+")
    n_examples_ele = int(np.ceil(n_examples / len(modes)))
    data_list_all = []
    if dirname is None:
        dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    image_filenames = sorted(filter_filename(dirname + "images"))
    for graph_key in modes:
        print(f"mode: {graph_key}:")
        data_list = get_clevr_graph_data_core(
            graph_key,
            dirname,
            resize=canvas_size,
            n_examples=n_examples_ele,
            image_filenames=image_filenames,
            isplot=isplot,
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


# ### Inference:

# In[ ]:


GRAPH_DICT = {
    "Graph1": [
        (0, "Red"),
        (1, ""),
        (2, ""),
        ((0,1), "SameColor"),
        ((1,2), "SameShape"),
    ],
    "Graph2": [
        (0, "Large"),
        (1, ""),
        (2, ""),
        ((0,1), "SameSize"),
        ((0,2), "SameColor"),   
    ],
    "Graph3": [
        (0, "Cube"),
        (1, ""),
        (2, ""),
        ((0,1), "SameShape"),
        ((1,2), "SameSize")
    ],
}


# In[ ]:


if __name__ == "__main__":
    mode = "Graph1+Graph2+Graph3"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-test-mpi-0-50000/"
    data_list = get_clevr_graph_data(mode, canvas_size=(64,64), n_examples=200, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_graph_Graph1+Graph2+Graph3_{64}_ex_{200}_4.p")


# ### Relation:

# In[ ]:


if __name__ == "__main__":
    mode = "SameColor+SameShape+SameSize"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    data_list = get_clevr_relation_data(mode, canvas_size=(64,64), n_examples=25000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_relation_{64}_ex_{25000}_1.p")


# In[ ]:


if __name__ == "__main__":
    mode = "SameColor+SameShape+SameSize"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-60000-130000/"
    data_list = get_clevr_relation_data(mode, canvas_size=(64,64), n_examples=30000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_relation_{64}_ex_{30000}_2.p")


# In[ ]:


if __name__ == "__main__":
    mode = "SameColor+SameShape+SameSize"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-130000-150000/"
    data_list = get_clevr_relation_data(mode, canvas_size=(64,64), n_examples=11000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_relation_{64}_ex_{11000}_3.p")


# ### Concept:

# In[ ]:


if __name__ == "__main__":
    mode = "SameColor+SameShape+SameSize"
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


# ### 3-concepts:

# In[ ]:


if __name__ == "__main__":
    mode = "Red+Cube+Large"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-0-60000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=20000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_Red+Cube+Large_{64}_ex_{20000}_1.p")


# In[ ]:


if __name__ == "__main__":
    mode = "Red+Cube+Large"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-60000-130000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=20000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_Red+Cube+Large_{64}_ex_{20000}_2.p")


# In[ ]:


if __name__ == "__main__":
    mode = "Red+Cube+Large"
    dirname = "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-v2-mpi-130000-150000/"
    data_list = get_clevr_concept_data(mode, canvas_size=(64,64), n_examples=4000, dirname=dirname)
    pdump(data_list, "/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/" + f"data_list_canvas_Red+Cube+Large_{64}_ex_{4000}_3.p")


# In[ ]:





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

