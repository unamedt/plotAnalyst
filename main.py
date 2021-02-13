#!/usr/bin/python3
import json
import plotly.graph_objs as go
from sys import argv
from math import exp, log



def plot(chart, series, settings):
  plotSettings = settings["plot"]
  #print(series)
  #print(plotSettings)
  if plotSettings["X_shift"] != 0.0:
    for i in range(len(series["X"])):
      series["X"][i] += plotSettings["X_shift"]
  if plotSettings["Y_shift"] != 0.0:
    for i in range(len(series["Y"])):
      series["Y"][i] += plotSettings["Y_shift"]
  chart.add_trace(go.Scatter(x=series["X"], y=series["Y"], name=series["name"], showlegend=plotSettings["showlegend"], mode=plotSettings["style"], line={"color":plotSettings["color"]}))

def file_import(filename, fileFormat, attrs):
  buff = {"X":[], "Y":[]}
  buff["name"] = attrs["name"]
  buff["tasks"] = attrs["tasks"]
  f = open(filename, "r")
  skipI = fileFormat["skip_strings"]
  while skipI > 0:
    f.readline()
    skipI -= 1
  line = f.readline()
  line = line.strip()
  while line != '':
    line_parsed = line_parse(line, fileFormat["string"])
    buff["X"].extend(line_parsed["X"])
    buff["Y"].extend(line_parsed["Y"])
    line = f.readline()
    line = line.strip()
  f.close()
  return buff

def line_parse(line, lineFormat):
  res = {"X":[], "Y":[]}
  line = line.split()
  lineFormat = lineFormat.split()
  '''
  if len(line) == len(lineFormat):
    for i in range(len(lineFormat)):
      if ((lineFormat[i][0] == "x") or (lineFormat[i][0] == "X")):
        res["X"].append(float(line[i]))
      elif ((lineFormat[i][0] == "y") or (lineFormat[i][0] == "Y")):
        res["Y"].append(float(line[i]))
  else:
    print("unknown line format")
    print("format:", lineFormat)
    print("line:  ", line)
  '''
  for item in zip(lineFormat, line):
    if (item[0] == "x") or (item[0] == "X"):
      res["X"].append(float(item[1]))
    if (item[0] == "y") or (item[0] == "Y"):
      res["Y"].append(float(item[1]))

  return res

def normalise(series, axis, minR, maxR):
  maxV = series[axis][0]
  minV = maxV
  for value in series[axis]:
    if maxV < value:
      maxV = value
    if minV > value:
      minV = value
  for i in range(len(series[axis])):
    series[axis][i] = ((series[axis][i] - minV) / (maxV - minV))*(maxR - minR) + minR
  return series

def cut(series, axis, left, right):
  series = sort(series, axis, "up")
  #print("data: [", left, ":", right, "]")
  #print("cut debug start============================")
  #for pair in zip(series["X"], series["Y"]):
  #  print(pair) 
  leftI = -1
  rightI = -1
  for i in range(len(series[axis])):
    #i += 1
    value = series[axis][i]
    if value >= left:
      if leftI == -1:
        leftI = i
    if value >= right:
      if rightI == -1:
        rightI = i + 1
  #print("indices: [", leftI,":", rightI, "]")
  if (rightI == -1) and (leftI != 0):
    rightI = len(series[axis])
  if (right != -1) and (left != -1):
    series["X"] = series["X"][leftI:rightI]
    series["Y"] = series["Y"][leftI:rightI]
  else:
    series["X"] = []
    series["Y"] = []
  #print("series after cut:")
  #for pair in zip(series["X"], series["Y"]):
  #  print(pair) 
  #print("cut debug stop=============================")
  return series

def sort(series, axis, order): #simple bubble sort
  axes = series["axes"]
  seriesBackup = series
  for eachAxis in axes:
    seriesBackup[eachAxis] = series[eachAxis].copy() #for immutability of incoming data
  sortComplete = False
  if order == "up":
    while not sortComplete:
      sortComplete = True
      for i in range(len(series[axis]) - 1):
        if series[axis][i] > series[axis][i + 1]:
          sortComplete = False
          for switchingAxis in axes:
            tmp = series[switchingAxis][i]
            series[switchingAxis][i] = series[switchingAxis][i + 1]
            series[switchingAxis][i + 1] = tmp
  elif order == "down":
    while not sortComplete:
      sortComplete = True
      for i in range(len(series[axis]) - 1):
        if series[axis][i] < series[axis][i + 1]:
          sortComplete = False
          for switchingAxis in axes:
            tmp = series[switchingAxis][i]
            series[switchingAxis][i] = series[switchingAxis][i + 1]
            series[switchingAxis][i + 1] = tmp
  seriesBuff = series
  series = seriesBackup

  return seriesBuff
    


def FindPeaksAdvanced(series, axis1, axis2, peaksTasks=[], der1Tasks=[], der2Tasks=[], int1Tasks=[], int0Tasks=[],  minDer_2=0.0, minWidth=0):
  #philosophy: peak-searching functions should generate a "series" dictionary with it`s own task set. Tasks for new "series" should be given in taskFile. New "series" should be appended to a "dataset"
  peaks = {"axes":series["axes"], "tasks":peaksTasks, "name":(series["name"]+"peaks")}

  der1 = derivative(peaks, axis1, axis2).update({"axes":series["axes"], "tasks":der1Tasks, "name":(series["name"] + "_der1")})
  der2 = derivative(der1,  axis1, axis2).update({"axes":series["axes"], "tasks":der2Tasks, "name":(series["name"] + "_der2")})
  
  #look for der2 zero crossings (double: up/down -- down/up) then integrate it twice between two zero crossings
  
  zeroCrosses = {"i":[], "type":[] } #type means: -1 -- up/down, 1 -- down/up
  for i in range(len(der2[axis2]) - 1):
    prev = der2[axis2][i]
    curr = der2[axis2][i + 1]
    if prev <= -0.0 and curr >= 0.0:
      zeroCrosses["i"] = i + 1 # + 1 is nessesary due to: result should include only under-zero part of der2. Otherwise the result will include one previous point of der2
      zeroCrosses["type"] = 1
    if prev >= 0.0 and curr <= -0.0:
      zeroCrosses["i"] = i
      zeroCrosses["type"] = -1

  #integrate
  #result is an array of peaks
  peaks["peaks"] = []
  noMorePeaks = False
  while not noMorePeaks:
    i = 0
    while zeroCrosses["type"][i] != 1:
      i += 1
      if i >= len(zeroCrosses["type"]):
        noMorePeaks = True
        break
    if i == 0:
      zeroCrosses["type"].pop(0)
      zeroCrosses["i"].pop(0)
      continue
    leftI = zeroCrosses["i"][i-1]
    rightI = zeroCrosses["i"][i]
    
    zeroCrosses["i"].pop(i-1)
    zeroCrosses["type"].pop(i-1)
    zeroCrosses["i"].pop(i-1)
    zeroCrosses["type"].pop(i-1)
    
    peaks["peaks"].append({axis1:series[axis1][leftI:rightI], axis2:series[axis2][leftI:rightI], der2:der2[axis2][leftI:rightI], der1:der1[axis2][leftI:rightI], "peak"+axis1:0.0, "peak"+axis2:0.0})

  for peakI in range(len(peaks["peaks"])):
    peak = peaks["peaks"][peakI]
    #integrate here 

    peaks["peaks"][peakI] = peak
        

 




  

  





def main():
  try:
    taskFile = open(argv[1], 'r')
  except BaseException:
    print("Error. Please provide the actual name of tasks file.")
    return -1
  tasks = json.load(taskFile)
  taskFile.close()
  #print(tasks)
  dataset = []
  for series in tasks:
    dataset.append(file_import(series["filename"], series["format"], series))
  #print(dataset)
  chart = go.Figure()
  for series in dataset:
    series["axes"] = ["X", "Y"] #TODO: add support of more than 2 axes
    sIndex = dataset.index(series)
    for task in series["tasks"]:
      print(task)
      if "normalise" in task:
        dataset[sIndex] = normalise(series, task["normalise"]["axis"],task["normalise"]["min"], task["normalise"]["max"])
      elif "sort" in task:
        dataset[sIndex] = sort(series, task["sort"]["axis"], task["sort"]["direction"])
      elif "cut" in task:
        dataset[sIndex] = cut(series, task["cut"]["axis"], task["cut"]["left"], task["cut"]["right"])
      elif "find peaks advanced":
        dataset[sIndex] = FindPeaksAdvanced(series, min_der_2=task["find peaks advanced"]["minimal der 2"], min_width=task["find peaks advanced"]["minimal peak width"], fake_hight=False)
      elif "plot" in task:
        plot(chart, series, task)
  chart.show()


main()

#TODO: add checking of incoming json config. All missed fields should be filled by default values.
#TODO: add immutability of incoming data to every function
