import urllib.request

# TODO
def loadMnist():
  response = urllib.request.urlopen('')
  data = response.read()
  print(data)
  