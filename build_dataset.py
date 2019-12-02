import sys
import shutil
import numpy as np

root_path = sys.argv[1]
if not shutil.os.path.isdir(root_path):
    raise ValueError("The informed path does not exist. Please try to run again with a valid folder path.")

faces = dict()
for face in [faces for faces in shutil.os.listdir(root_path) if shutil.os.path.isdir(root_path+'/'+faces)]:
    faces[face] = np.array([filename for filename in shutil.os.listdir(root_path+'/'+face) if shutil.os.path.isfile(root_path+'/'+face+'/'+filename) and filename.split('_')[-1] in ['open.pgm', 'sunglasses.pgm']])
    np.random.shuffle(faces[face])

if not shutil.os.path.isdir(root_path+'/Train'):
    shutil.os.mkdir(root_path+'/Train')
if not shutil.os.path.isdir(root_path+'/Test'):
    shutil.os.mkdir(root_path+'/Test')

for face in faces:
    print(len(faces[face]))
    if not shutil.os.path.isdir(root_path+'/Train/'+face):
        shutil.os.mkdir(root_path+'/Train/'+face)
    for filename in faces[face][0:26]:
        shutil.copyfile(root_path+'/'+face+'/'+filename, root_path+'/Train/'+face+'/'+filename)
    if not shutil.os.path.isdir(root_path+'/Test/'+face):
        shutil.os.mkdir(root_path+'/Test/'+face)
    for filename in faces[face][26:]:
        shutil.copyfile(root_path+'/'+face+'/'+filename, root_path+'/Test/'+face+'/'+filename)
