import gzip
from entrypoint import mnist_helper

def loadLabelData(filepath):
  f = gzip.open(filepath, 'r')
  
  magic = int.from_bytes(f.read(4), 'big')
  length = int.from_bytes(f.read(4), 'big')

  if(magic != 2049): raise Exception('File {} corrupted. Must contain magic number 2049'.format(filepath))

  labels = []
  for index in range(length):
    label = int.from_bytes(f.read(1), 'big')
    labels.append(label)
    # print(index, label)

  f.close()

  return labels

def loadImageData(filepath):
  f = gzip.open(filepath, 'r')
  
  magic = int.from_bytes(f.read(4), 'big')
  length = int.from_bytes(f.read(4), 'big')
  rows = int.from_bytes(f.read(4), 'big')
  cols = int.from_bytes(f.read(4), 'big')

  if(magic != 2051): raise Exception('File {} corrupted. Must contain magic number 2051'.format(filepath))

  images = []
  # for index in range(length):
  #   label = int.from_bytes(f.read(1), 'big')
  #   images.append(label)

  magic = int.from_bytes(f.read(4), 'big')
  # read images
  for imageCounter in range(length):
    # read single image
    image = []
    for i in range(rows * cols): image.append(int.from_bytes(f.read(1), 'big'))
    images.append(image)

  f.close()

  return images
  
def loadMnistData():
  """
  This function looks in in the root dir for the following files.
  datasets/mnist/t10k-images-idx3-ubyte.gz
  datasets/mnist/t10k-labels-idx1-ubyte.gz
  datasets/mnist/train-images-idx3-ubyte.gz
  datasets/mnist/train-labels-idx1-ubyte.gz
  You can download them from https://deepai.org/dataset/mnist.
  The files are loaded in memory and returned as a dictionary.
  {
    train: { images: [ [...], [...], ], labels: [...]},
    test: { images: [ [...], [...], ], labels: [...]}
  }
  """
  testLabels = loadLabelData( 'datasets/mnist/t10k-labels-idx1-ubyte.gz' )
  testImages = loadImageData('datasets/mnist/t10k-images-idx3-ubyte.gz')
  trainLabels = loadLabelData( 'datasets/mnist/train-labels-idx1-ubyte.gz' )
  trainImages = loadImageData('datasets/mnist/train-images-idx3-ubyte.gz')
  
  return {
    "train": { "images": trainImages, "labels": trainLabels },
    "test": { "images": testImages, "labels": testLabels }
  }


