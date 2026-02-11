import os
import pickle
import trimesh
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def py_file_to_mesh_file(py_path):
    try:
        with open(py_path, 'r') as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        mesh.export(py_path[:-3] + '.stl')
    except:
        pass


def run_split(path, py_paths, split):
    pool = Pool(32)
    list(tqdm(pool.imap(py_file_to_mesh_file, py_paths), total=len(py_paths)))
    pool.close()
    pool.join()

    annotations = list()
    for py_path in py_paths:
        mesh_path = py_path[:-3] + '.stl'
        if os.path.exists(mesh_path):
            annotations.append(dict(
                py_path=py_path[len(path) + 1:],
                mesh_path=mesh_path[len(path) + 1:]
            ))
    with open(os.path.join(path, f'{split}.pkl'), 'wb') as f:
        pickle.dump(annotations, f)
    print(split, len(annotations))


def run(path):
    run_split(path, glob(os.path.join(path, 'train', '*/*.py')), 'train')
    run_split(path, glob(os.path.join(path, 'val', '*.py')), 'val')


if __name__ == '__main__':
    run('./cad-recode-v1.5')
