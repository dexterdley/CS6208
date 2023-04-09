import os
import wget
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import pandas as pd
from tqdm import tqdm
import pickle


""" Helper functions"""
def draw_landmarks(img, landmarks):
  image = img.copy()
  for point in landmarks:
    x, y = point[0].item(), point[1].item()
    image = cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
  return image

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def delaunay_triangulation(points):
    convexhull = cv2.convexHull(points) #create outline from points
  
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)

    subdiv.insert(points.tolist()) #points need to be in list type
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
    
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
    
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
    
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
    
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    
    return indexes_triangles

def draw_edges(img, triangles, landmarks):
    image = img.copy()
    for t in triangles:
        pt1 = landmarks[t[0]]
        pt2 = landmarks[t[1]]
        pt3 = landmarks[t[2]]

        cv2.line(image, pt1, pt2, (0, 0, 255), 2)
        cv2.line(image, pt2, pt3, (0, 0, 255), 2)
        cv2.line(image, pt1, pt3, (0, 0, 255), 2)

    return image

def get_edges_and_weights(triangles, landmarks):
  edges = []
  edges_weight = []
  for t in triangles:
      e1 = [ t[0], t[1] ]
      e2 = [ t[1], t[2] ]
      e3 = [ t[0], t[2] ]
      if sorted(e1) not in edges:
        edges.append(sorted(e1))
      if sorted(e2) not in edges:
        edges.append(sorted(e2))
      if sorted(e3) not in edges:
        edges.append(sorted(e3))
  #print( edges )

  for edge in edges: #edge weight
    pt1 = landmarks[edge[0]]
    pt2 = landmarks[edge[1]]
    d = np.sqrt( (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1]) **2 )
    edges_weight.append(d)

  return np.stack(edges), np.array(edges_weight)
""" End of helper functions"""

data_root = "datasets"
base_url = "https://graal.ift.ulaval.ca/public/celeba/"

file_list = [
    "img_align_celeba.zip",
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
    "list_eval_partition.txt",
]

# Path to folder with the dataset
dataset_folder = f"{data_root}/celeba"
os.makedirs(dataset_folder, exist_ok=True)

for file in file_list:
    url = f"{base_url}/{file}"
    if not os.path.exists(f"{dataset_folder}/{file}"):
        wget.download(url, f"{dataset_folder}/{file}")

#Declare dataset here

dataframe = {}
dataframe['nodes'] = {}
dataframe['edge_index'] = {}
dataframe['edge_weight'] = {}
dataframe['y'] = {}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

partition = 'train'
use_DLIB = True

if partition == 'train':
	dataset = torchvision.datasets.CelebA(data_root, split="train", target_type=["attr", "landmarks"], transform=None)

elif partition == 'valid':
	dataset = torchvision.datasets.CelebA(data_root, split="valid", target_type=["attr", "landmarks"], transform=None)

elif partition == 'test':
	dataset = torchvision.datasets.CelebA(data_root, split="test", target_type=["attr", "landmarks"], transform=None)


for i in tqdm(range(dataset.__len__()) ,desc="Extracting: " + partition):
  img, items = dataset.__getitem__(i)
  image = np.array(img)

  if use_DLIB == False:
    attr, landmarks = items
    landmarks = np.array(landmarks.reshape(5,2))

    triangles = delaunay_triangulation(landmarks)
    nodes = np.mean(image[landmarks[:,0], landmarks[:,1]] , axis=1)#take mean of each pixel RGB value
    edge_index, edge_weight = get_edges_and_weights(triangles, landmarks)


    dataframe['nodes'][i] = np.array(nodes, dtype=np.int8)
    dataframe['edge_index'][i] = np.array(edge_index, dtype=np.int8)
    dataframe['edge_weight'][i] = np.array(edge_weight, dtype=np.int8)
    dataframe['y'][i] = np.array(attr, dtype=np.int8)
    
  else:
    
    attr, _ = items
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray)

    for face in faces:
        landmarks = predictor(image, face)

        landmarks = np.array([(min(landmarks.part(n).x, 177), min(landmarks.part(n).y, 217) ) for n in range(0, 68)] )

    triangles = delaunay_triangulation(landmarks)
    nodes = np.mean(image[landmarks[:,1], landmarks[:,0]] , axis=1) #take mean of each pixel RGB value
    edge_index, edge_weight = get_edges_and_weights(triangles, landmarks)

    dataframe['nodes'][i] = np.array(nodes, dtype=np.int8)
    dataframe['edge_index'][i] = np.array(edge_index, dtype=np.int8)
    dataframe['edge_weight'][i] = np.array(edge_weight, dtype=np.int8)
    dataframe['y'][i] = np.array(attr, dtype=np.int8)

dataframe = pd.DataFrame.from_dict(dataframe)

if use_DLIB:
  d = "x68"
else:
  d = "x5"

filename = d + '_' + partition + '_landmarks.pkl'

dataframe.to_pickle(filename)
print("Saved ", filename)