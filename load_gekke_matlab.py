import numpy as np
import h5py


def copy_refs(refs, f):
    s = refs.shape
    dest = np.empty((s[2], s[1], s[0]), dtype=object)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                dest[k, j, i] = f[refs[i, j, k]][:]
    return dest

def load_matlab_met_gekke_nested_shit(filename, structname, matrixname):
    with h5py.File(filename, "r") as f:
        struct = f[structname]
        array = copy_refs(struct[matrixname], f)
        x=3
        return array


if __name__ == "__main__":
    data = load_matlab_met_gekke_nested_shit("savine.mat", "TPD", "T")
    print(data)
   
