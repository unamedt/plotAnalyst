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
    

def FindPeaksAdvanced(series, min_der_2=0.0, min_width=0, fake_hight=False):
  peaks = {"I":[], "Q":[], "der_2":[], "der_1":[]}
  der_1 = {"I":[], "Q":[]}
  der_2 = {"I":[], "Q":[]}
  
  #discover avg dQ:
  total_points = len(data["Q"])
  dQ = (data["Q"][-1] - data["Q"][0])/total_points
   
  for i in range(len(data["I"]) - 1):
    Q = (data["Q"][i] + data["Q"][i + 1])/2
    I = (data["I"][i + 1] - data["I"][i])/(data["Q"][i + 1] - data["Q"][i])
    der_1["Q"].append(Q)
    der_1["I"].append(I)
   
  der_1_copy = {"I":der_1["I"].copy(), "Q":der_1["Q"].copy()}
  
  for i in range(len(der_1["I"]) - 1):
    Q = (der_1["Q"][i] + der_1["Q"][i + 1])/2
    I = (der_1["I"][i + 1] - der_1["I"][i])/(der_1["Q"][i + 1] - der_1["Q"][i])
    der_2["I"].append(I*200)
    der_2["Q"].append(Q)
  
  der_2_zero = {"I":[], "Q":[], "der_2":[]}
  for i in range(len(der_2["I"]) - 1):
    I_1 = der_2["I"][i]
    I_2 = der_2["I"][i + 1]
    Q = (der_2["Q"][i] + der_2["Q"][i + 1])/2
    if ((I_1 <= -1 * min_der_2) and (I_2 >= min_der_2)):
      der_2_zero["I"].append(-100)
      #der_2_zero["I"].append(I_1 - I_2)
      der_2_zero["Q"].append(Q)
    elif ((I_1 >= min_der_2) and (I_2 <= -1 * min_der_2)):
      der_2_zero["I"].append(100)
      #der_2_zero["I"].append(I_1 - I_2)
      der_2_zero["Q"].append(Q)
  
  der_2_zero_copy = {"I":[], "Q":[]}
  der_2_zero_copy["I"] = der_2_zero["I"].copy()
  der_2_zero_copy["Q"] = der_2_zero["Q"].copy()
  while True:
    i = 0
    peak_found = False
    while ((i < (len(der_2_zero["I"]) - 1)) and (not peak_found)):
      if ((der_2_zero["I"][i] > 0) and (der_2_zero["I"][i + 1] < 0)):
        dQ = der_2_zero["Q"][i + 1] - der_2_zero["Q"][i]
        if dQ >= min_width:
          peaks["Q"].append((der_2_zero["Q"][i] + der_2_zero["Q"][i+1])/2)
          peaks["I"].append(110)
          i_left = nearest(der_2["Q"], der_2_zero["Q"][i])
          i_right = nearest(der_2["Q"], der_2_zero["Q"][i + 1])
          peaks["der_2"].append(der_2["I"][i_left: i_right])
        der_2_zero["Q"].pop(i)
        der_2_zero["Q"].pop(i)
        der_2_zero["I"].pop(i)
        der_2_zero["I"].pop(i)
        peak_found = True
      i += 1
    if not peak_found:
      break
  #print(peaks["I"])
  for peak_i in range(len(peaks["der_2"])):
    peaks["der_1"].append([0.0])
    for i in range(len(peaks["der_2"][peak_i])):
      ddI = peaks["der_2"][peak_i][i]
      peaks["der_1"][peak_i].append(ddI*dQ)
  for peak_i in range(len(peaks["der_1"])):
    peaks["I"][peak_i] = 0.0
    for i in range(len(peaks["der_2"][peak_i])):
      ddI = peaks["der_2"][peak_i][i]
      peaks["I"][peak_i] -= (ddI*dQ)
  peak_i = 0
  while peak_i <(len(peaks["I"])):
    if peaks["I"][peak_i] > 0.0:
      peaks["I"][peak_i] = (peaks["I"][peak_i])**1.00
      peak_i += 1
    else:
      peaks["I"].pop(peak_i)
      peaks["Q"].pop(peak_i)
  peaks = Normalize(peaks,nonzero=True)
  if fake_hight: #moves peak highest point to the nearest point of data
    for peak_i in range(len(peaks["Q"])):
      dQ = 100500
      dQ_prev = dQ
      Q = peaks["Q"][peak_i]
      peak_I_not_found = True
      for i in range(len(data["Q"])):
        dQ = abs(data["Q"][i] - Q)
        #print(peak_i, dQ, data["Q"][i], peaks["Q"][peak_i], i)
        if ((dQ >= dQ_prev) and (peak_I_not_found)):
          peaks["I"][peak_i] = data["I"][i - 1]
          peak_I_not_found = False
        dQ_prev = dQ


  return {"peaks": peaks , "der":[data, der_1_copy, der_2, der_2_zero_copy]}

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
