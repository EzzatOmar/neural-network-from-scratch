def plotImage(image):
  """ Takes a 28*28 int list and prints the image in the console """
  image = list(map(lambda x: "0" if x==0 else "*", image))
  for row in range(28):
    index = row*28
    print("".join(image[index:index+28]))