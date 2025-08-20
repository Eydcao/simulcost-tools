import numpy as np
import struct
import mmap
import os
import re

# Global block variable (to mimic MATLAB 'global block')
block = {}

def get_data_sdf(filename):
    global block

    class H:
        pass

    h = H()
    h.ID_LENGTH = 32
    h.ENDIANNESS = 16911887
    h.VERSION = 1
    h.REVISION = 4
    h.MAGIC = b'SDF1'

    class BLOCKTYPE:
        SCRUBBED = -1
        NULL = 0
        PLAIN_MESH = 1
        POINT_MESH = 2
        PLAIN_VARIABLE = 3
        POINT_VARIABLE = 4
        CONSTANT = 5
        ARRAY = 6
        RUN_INFO = 7
        SOURCE = 8
        STITCHED_TENSOR = 9
        STITCHED_MATERIAL = 10
        STITCHED_MATVAR = 11
        STITCHED_SPECIES = 12
        SPECIES = 13
        PLAIN_DERIVED = 14
        POINT_DERIVED = 15
        CONTIGUOUS_TENSOR = 16
        CONTIGUOUS_MATERIAL = 17
        CONTIGUOUS_MATVAR = 18
        CONTIGUOUS_SPECIES = 19
        CPU_SPLIT = 20
        STITCHED_OBSTACLE_GROUP = 21
        UNSTRUCTURED_MESH = 22
        STITCHED = 23
        CONTIGUOUS = 24
        LAGRANGIAN_MESH = 25
        STATION = 26
        STATION_DERIVED = 27
        DATABLOCK = 28
        NAMEVALUE = 29

    h.BLOCKTYPE = BLOCKTYPE

    class DATATYPE:
        NULL = 0
        INTEGER4 = 1
        INTEGER8 = 2
        REAL4 = 3
        REAL8 = 4
        REAL16 = 5
        CHARACTER = 6
        LOGICAL = 7
        OTHER = 8

    h.DATATYPE = DATATYPE
    h.filename = filename

    # Open file
    try:
        fid = open(filename, 'rb')
    except FileNotFoundError:
        print('bad filename')
        return 'fail'

    h.fid = fid

    # Read file header
    sdf_magic = fid.read(4)
    endianness = struct.unpack('i', fid.read(4))[0]
    version = struct.unpack('i', fid.read(4))[0]
    revision = struct.unpack('i', fid.read(4))[0]
    code_name = fid.read(h.ID_LENGTH).decode('utf-8').strip()
    first_block_location = struct.unpack('q', fid.read(8))[0]
    summary_location = struct.unpack('q', fid.read(8))[0]
    summary_size = struct.unpack('i', fid.read(4))[0]
    nblocks = struct.unpack('i', fid.read(4))[0]
    h.block_header_length = struct.unpack('i', fid.read(4))[0]
    step = struct.unpack('i', fid.read(4))[0]
    time = struct.unpack('d', fid.read(8))[0]
    jobid1 = struct.unpack('i', fid.read(4))[0]
    jobid2 = struct.unpack('i', fid.read(4))[0]
    string_length = struct.unpack('i', fid.read(4))[0]
    code_io_version = struct.unpack('i', fid.read(4))[0]

    q = {'step': step, 'time': time}

    # Block processing
    b = {}
    b['block_start'] = first_block_location
    blocklist = []

    gridtype = 0

    for n in range(nblocks):
        fid.seek(b['block_start'], 0)
        b['next_block_location'] = struct.unpack('Q', fid.read(8))[0]
        b['data_location'] = struct.unpack('Q', fid.read(8))[0]
        b['id'] = fid.read(h.ID_LENGTH).decode('utf-8').strip()
        b['data_length'] = struct.unpack('Q', fid.read(8))[0]
        b['blocktype'] = struct.unpack('I', fid.read(4))[0]
        b['datatype'] = struct.unpack('I', fid.read(4))[0]
        b['ndims'] = struct.unpack('I', fid.read(4))[0]
        b['name'] = fid.read(string_length).decode('utf-8').strip()
        b['mesh_id'] = ''
        b['var'] = None
        b['map'] = None

        block = b.copy()

        if block['blocktype'] == h.BLOCKTYPE.PLAIN_MESH:
            block['var'] = get_plain_mesh_sdf(h)
            gridtype = h.BLOCKTYPE.PLAIN_MESH
        elif block['blocktype'] == h.BLOCKTYPE.LAGRANGIAN_MESH:
            block['var'] = get_lagrangian_mesh_sdf(h)
            gridtype = h.BLOCKTYPE.LAGRANGIAN_MESH
        elif block['blocktype'] == h.BLOCKTYPE.POINT_MESH:
            block['var'] = get_point_mesh_sdf(h)
        elif block['blocktype'] == h.BLOCKTYPE.PLAIN_VARIABLE:
            block['var'] = get_plain_variable_sdf(h)
        elif block['blocktype'] == h.BLOCKTYPE.POINT_VARIABLE:
            block['var'] = get_point_variable_sdf(h)
        elif block['blocktype'] == h.BLOCKTYPE.CONSTANT:
            block['var'] = get_constant_sdf(h)

        blocklist.append(block)
        b['block_start'] = b['next_block_location']

    # Build output structure
    for b in blocklist:
        # Split names like MATLAB does
        parts = [re.sub(r'\W','',s) for s in b['name'].replace(' ','_').split('/')]
        add = b['blocktype'] in [h.BLOCKTYPE.PLAIN_MESH, h.BLOCKTYPE.LAGRANGIAN_MESH,
                                 h.BLOCKTYPE.POINT_MESH, h.BLOCKTYPE.PLAIN_VARIABLE,
                                 h.BLOCKTYPE.POINT_VARIABLE, h.BLOCKTYPE.CONSTANT]

        # Attach mesh grid if needed
        hasgrid = 0
        if b['blocktype'] == h.BLOCKTYPE.PLAIN_VARIABLE:
            hasgrid = gridtype
        elif b['blocktype'] == h.BLOCKTYPE.POINT_VARIABLE:
            hasgrid = h.BLOCKTYPE.POINT_MESH

        got = False
        grid = None
        gname = ''
        if hasgrid:
            for g in blocklist:
                if g['blocktype'] == hasgrid and b['mesh_id'] == g['id']:
                    grid = g['var']
                    gname = g['name'].replace(' ','_').replace('/','.')
                    got = True
                    break

        if add:
            target = q
            for p in parts[:-1]:
                if p not in target:
                    target[p] = {}
                target = target[p]
            target[parts[-1]] = b['var']
            if got:
                target[parts[-1]]['grid'] = grid
                target[parts[-1]]['grid']['name'] = gname

    fid.close()
    return q


# -----------------------------
# Helper functions

def get_plain_mesh_sdf(h):
    global block
    with open(h.filename, 'rb') as f:
        f.seek(block['block_start'] + h.block_header_length)
        mults = np.fromfile(f, dtype=np.float64, count=block['ndims'])
        labels = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        units = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        geometry = np.fromfile(f, dtype=np.int32, count=1)[0]
        extents = np.fromfile(f, dtype=np.float64, count=2*block['ndims'])
        npts = np.fromfile(f, dtype=np.int32, count=block['ndims'])

    q = {'labels': labels, 'units': units}

    typestring = get_numpy_dtype(block['datatype'])

    nelements = sum(npts)
    typesize = int(block['data_length'] / nelements)

    f = open(h.filename, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    offset = int(block['data_location'])

    tags = list('xyzabcdefghijklmn')
    for n in range(block['ndims']):
        tagname = tags[n]
        num_elements = npts[n]
        data = np.frombuffer(mm[offset:offset + typesize*num_elements], dtype=typestring, count=num_elements)
        q[tagname] = data
        offset += typesize * num_elements

    mm.close()
    f.close()
    return q


def get_lagrangian_mesh_sdf(h):
    global block
    with open(h.filename, 'rb') as f:
        f.seek(block['block_start'] + h.block_header_length)
        mults = np.fromfile(f, dtype=np.float64, count=block['ndims'])
        labels = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        units = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        geometry = np.fromfile(f, dtype=np.int32, count=1)[0]
        extents = np.fromfile(f, dtype=np.float64, count=2*block['ndims'])
        npts = np.fromfile(f, dtype=np.int32, count=block['ndims'])

    q = {'labels': labels, 'units': units}

    typestring = get_numpy_dtype(block['datatype'])
    nelements = np.prod(npts)
    typesize = int(block['data_length'] / block['ndims'] / nelements)


    f = open(h.filename, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    offset = int(block['data_location'])

    tags = list('xyzabcdefghijklmn')
    for n in range(block['ndims']):
        tagname = tags[n]
        shape = tuple(npts)
        data = np.frombuffer(mm[offset:], dtype=typestring, count=nelements).reshape(shape)
        q[tagname] = data
        offset += typesize * nelements

    mm.close()
    f.close()
    return q


def get_point_mesh_sdf(h):
    global block
    with open(h.filename, 'rb') as f:
        f.seek(block['block_start'] + h.block_header_length)
        mults = np.fromfile(f, dtype=np.float64, count=block['ndims'])
        labels = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        units = [f.read(h.ID_LENGTH).decode('utf-8').strip() for _ in range(block['ndims'])]
        geometry = np.fromfile(f, dtype=np.int32, count=1)[0]
        extents = np.fromfile(f, dtype=np.float64, count=2*block['ndims'])
        npart = np.fromfile(f, dtype=np.int64, count=1)[0]

    q = {'labels': labels, 'units': units}
    typestring = get_numpy_dtype(block['datatype'])
    nelements = block['ndims'] * npart
    typesize = int(block['data_length'] / nelements)

    f = open(h.filename, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    offset = int(block['data_location'])

    tags = list('xyzabcdefghijklmn')
    for n in range(block['ndims']):
        tagname = tags[n]
        data = np.frombuffer(mm[offset:offset + typesize*npart], dtype=typestring, count=npart)
        q[tagname] = data
        offset += typesize * npart

    mm.close()
    f.close()
    return q


def get_plain_variable_sdf(h):
    global block
    fid = h.fid
    fid.seek(block['block_start'] + h.block_header_length)
    mult = struct.unpack('d', fid.read(8))[0]
    units = fid.read(h.ID_LENGTH).decode('utf-8').strip()
    block['mesh_id'] = fid.read(h.ID_LENGTH).decode('utf-8').strip()
    npts = np.fromfile(fid, dtype=np.int32, count=block['ndims'])
    stagger = struct.unpack('i', fid.read(4))[0]

    typestring = {3:'float32',4:'float64',1:'int32',2:'int64'}.get(block['datatype'],'float64')

    offset = block['data_location']

    with open(h.filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm[offset:offset+block['data_length']], dtype=typestring)
        mm.close()

    return {'data': data}


def get_point_variable_sdf(h):
    global block
    fid = h.fid
    fid.seek(block['block_start'] + h.block_header_length)
    mult = struct.unpack('d', fid.read(8))[0]
    units = fid.read(h.ID_LENGTH).decode('utf-8').strip()
    block['mesh_id'] = fid.read(h.ID_LENGTH).decode('utf-8').strip()
    npart = struct.unpack('q', fid.read(8))[0]

    typestring = {3:'float32',4:'float64',1:'int32',2:'int64'}.get(block['datatype'],'float64')

    offset = block['data_location']

    with open(h.filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm[offset:offset+block['data_length']], dtype=typestring)
        mm.close()

    return {'data': data}


def get_constant_sdf(h):
    global block
    fid = h.fid
    fid.seek(block['block_start'] + h.block_header_length)

    typestring = {3:'float32',4:'float64',1:'int32',2:'int64'}.get(block['datatype'],'float64')
    val = np.fromfile(fid, dtype=typestring, count=1)[0]
    return val

def get_numpy_dtype(datatype):
    if datatype == 3:  # REAL4
        return np.float32
    elif datatype == 4:  # REAL8
        return np.float64
    elif datatype == 1:  # INTEGER4
        return np.int32
    elif datatype == 2:  # INTEGER8
        return np.int64
    else:
        return np.float64  # fallback

