#!/usr/bin/python3
import json
import plotly.graph_objs as go
from sys import argv
from math import exp, log



def plot(chart, series, settings):
  plotSettings = settings["plot"]
  #print(series)
  #print(plotSettings)
  if "X_shift" in plotSettings:
    if plotSettings["X_shift"] != 0.0:
      for i in range(len(series["X"])):
        series["X"][i] += plotSettings["X_shift"]
  if "Y_shift" in plotSettings:
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
    


def derivative(series, axis1, axis2, noShift=False):
  '''returns "series" with a result of derivation of axis2 by axis1
  noShift==True means that in a result X is a center of an gap with df/dx == value
  else it means that in a result X is a left border of a gap'''
  #print("start derivative debug=====================")
  #print(series)
  x = series[axis1]
  f = series[axis2]
  result = {axis1:[], axis2:[]}
  for i in range(len(f) - 1):
    df = f[i + 1] - f[i]
    dx = x[i + 1] - x[i]
    if dx != 0.0:
      result[axis2].append(df/dx)
      if noShift:
        result[axis1].append(x[i])
      else:
        result[axis1].append(x[i] + (x[i + 1] - x[i])/2)
  if not noShift:
    result[axis1].append(x[-1])
  #print(result)
  #print("stop derivative debug======================")
  return result 

def integrate(series, axis1, axis2, noShift=False):
  '''returns "series" with a result of integration of axis2 by axis1'''
  #print("start integrate debug=====================")
  x = series[axis1]
  f = series[axis2]
  #print("x:",x)
  #print("f:",f)

  result = {axis1:[], axis2:[]}
  if not noShift:
    result[axis1] = x[0: -1]
    for i in range(len(x) - 1):
      F = f[i] * (x[i - 1] - x[i])
      result[axis2].append(F)
  else:  #approximate that dx at both ends of  integrating segment == avg of all other dx
    dx = avg(x)
    X = x[0] - dx/2
    df = f[0]
    F = dx*df
    result[axis1].append(X)
    result[axis2].append(F)
    for i in range(1, len(x) - 2):
      dx = (x[i + 1] - x[i - 1])/2
      df = f[i]
      F = df*dx
      X = x[i] - dx/2
      result[axis1].append(X)
      result[axis2].append(F)
    
    dx = avg(x)
    X = x[-1] - dx/2
    df = f[-1]
    F = dx*df
    result[axis1].append(X)
    result[axis2].append(F)
  #print("stop integrate debug======================")
  return result

def find_peaks_advanced(series, axis1, axis2, peaksTasks=[], der1Tasks=[], der2Tasks=[], int2Tasks=[], int1Tasks=[],  level=0.0, levelSign="+", levelType="peakY",  minWidth=0, axis2Type="Y", axis2Value='min'):
  #philosophy: peak-searching functions should generate a "series" dictionary with it`s own task set. Tasks for new "series" should be given in taskFile. New "series" should be appended to a "dataset"
  peaks = {"axes":series["axes"], "tasks":peaksTasks, "name":(series["name"]+"_advanced_peaks")}
  series[axis1] = series[axis1] #TODO: what does this line?
  series[axis2] = series[axis2] #TODO: what does this line?
  der1 = derivative(series, axis1, axis2)
  der1.update({"axes":series["axes"], "tasks":der1Tasks, "name":(series["name"] + "_der1")})
  der2 = derivative(der1,  axis1, axis2)
  der2.update({"axes":series["axes"], "tasks":der2Tasks, "name":(series["name"] + "_der2")})
  
  #look for der2 zero crossings (double: up/down -- down/up) then integrate it twice between two zero crossings
  
  zeroCrosses = {"i":[], "type":[] } #type means: -1 -- up/down, 1 -- down/up
  for i in range(len(der2[axis2]) - 1):
    prev = der2[axis2][i]
    curr = der2[axis2][i + 1]
    if prev <= -0.0 and curr >= 0.0:
      zeroCrosses["i"].append(i + 1) # + 1 is nessesary due to: result should include only under-zero part of der2. Otherwise the result will include one previous point of der2
      zeroCrosses["type"].append(1)
    if prev >= 0.0 and curr <= -0.0:
      zeroCrosses["i"].append(i)
      zeroCrosses["type"].append(-1)
  #print(zeroCrosses)
  #delete all unmatching zero crosses:
  while zeroCrosses["type"][0] != -1:
    zeroCrosses["type"].pop(0)
    zeroCrosses["i"].pop(0)
    if len(zeroCrosses["type"]) == 0:
      break #TODO: add: this line should return empty list of peaks with format of the whole function
  if len(zeroCrosses["type"]) != 0:
    while zeroCrosses["type"][-1] != 1:
      zeroCrosses["type"].pop(-1)
      zeroCrosses["i"].pop(-1)
      if len(zeroCrosses["type"]) == 0:
        break #TODO: add: this line should return empty list of peaks with format of the whole function


  #integrate
  #result is an array of peaks
  peaks["peaks"] = []
  noMorePeaks = False
  while not noMorePeaks:
    i = 0
    while zeroCrosses["type"][i] != 1:
      i += 1
      #print(zeroCrosses)
      #print("noMorePeaks:", noMorePeaks)
      #print("length:", len(zeroCrosses["i"]), "  i =", i)
      if i >= (len(zeroCrosses["type"]) - 1):
        noMorePeaks = True
        #print("break")
        break
    #This is not actual because unmatching zero crosses deleted some lines ago
    #if i == 0:
    #  zeroCrosses["type"].pop(0)
    #  zeroCrosses["i"].pop(0)
    #  continue
    leftI = zeroCrosses["i"][i-1]
    rightI = zeroCrosses["i"][i]
    
    zeroCrosses["i"].pop(i-1)
    zeroCrosses["type"].pop(i-1)
    zeroCrosses["i"].pop(i-1)
    zeroCrosses["type"].pop(i-1)
    
    peaks["peaks"].append({axis1:series[axis1][leftI:rightI], axis2:series[axis2][leftI:rightI], "der2":der2[axis2][leftI:rightI], "der1":der1[axis2][leftI:rightI], "peak"+axis1:0.0, "peak"+axis2:0.0, "width":(series[axis1][rightI] - series[axis1][leftI])})
  


  for peakI in range(len(peaks["peaks"])):
    peak = peaks["peaks"][peakI]
    
    #integrate here 
    int1 = integrate({axis1:peak[axis1], axis2:peak["der2"]}, axis1, axis2)
    int2 = integrate({axis1:peak[axis1], axis2:int1[axis2]}, axis1, axis2)

    #approximate peak location
    
    peakAxis1Found = False
    for i in range(len(int1[axis2])- 1):
      a = int1[axis2][i]
      b = int1[axis2][i + 1]
      if (a >= 0.0) and (b <= 0.0):
        peakAxis1Found = True
        peak["peak" + axis1] = peak[axis1][i] + (peak[axis1][i + 1] - peak[axis1][i])*(a/(a + b))
    if not peakAxis1Found:
      peak["peak" + axis1] = peak[axis1][0] + (peak[axis1][-1] - peak[axis1][0])/2.0
    
    #calculate peak height
    peak["peak" + axis2] = max(int2[axis2])

    peak["int1"] = int1[axis2]
    peak["int2"] = int2[axis2]

    peaks["peaks"][peakI] = peak

  #delete all peaks which have abs(der2) higher than limit
  i = 0
  while i < (len(peaks["peaks"])):
    delPeak = False

    if levelType in ("der2", "der1", "int1", "int2", "X", "Y"):
      if levelSign == "+":
        if (max(peaks["peaks"][i][levelType]) < (minDer_2)):
          delPeak = True
      else:
        if (min(peaks["peaks"][i][levelType]) > (minDer_2)):
          delPeak = True

    elif levelType in ("peakX", "peakY", "width"):
      if levelSign == "+":
        if (peaks["peaks"][i][levelType] < (level)):
          delPeak = True
      else:
        if (peaks["peaks"][i][levelType] > (level)):
          delPeak = True

    if delPeak:  
      peaks["peaks"].pop(i)
    else:
      i += 1
    #print(i, len(peaks["peaks"]))
  del i

  resultPeaks = {axis1:[], axis2:[], "name":series["name"]+ "_trivial_peaks", "tasks":peaksTasks}
  for peak in peaks["peaks"]:
    #print(peak)
    resultPeaks[axis1].append(peak["peakX"])
    if axis2Type in ("der2", "der1", "int1", "int2", "X", "Y"):
      if axis2Value == "min":
        resultPeaks[axis2].append(min(peak[axis2Type]))
      else:
        resultPeaks[axis2].append(max(peak[axis2Type]))
    if axis2Type in ("peakX", "peakY", "width"): 
      resultPeaks[axis2].append(peak[axis2Type])

  return (resultPeaks, der1, der2)

#def sliding_median() #optimisation hack: this function should use buffer in a default argument value: buff=[]

def median(array):
  tmp = array.copy()
  tmp.sort()
  length = len(tmp)
  #print(length//2, length%2)
  if length%2:
    result = tmp[length//2]
  else:
    result = tmp[(length - 1)//2] + (tmp[length//2] - tmp[(length - 1)//2])/2
  del tmp
  return result

def average(array):
  accumulator = 0.0
  for value in array:
    accumulator += value
  return accumulator/(len(array))

def nearest(array, item):
  """This function returns object in array nearest to item
  """
  if len(array) == 1:
    return array[0]
  tmp = array.copy()
  tmp.sort()
  itemPrev = tmp[0]
  for itemCurr in tmp[1:]:
    deltaPrev = abs(item - itemPrev)
    deltaCurr = abs(item - itemCurr)
    if deltaCurr > deltaPrev:
      resut = itemPrev
      del array
      return result
  
  result = tmp[-1]
  del array
  return result


def find_peaks_trivial(series, axis1, axis2, apexesTasks=[], peaksTasks=[], levelTasks=[], triggerValue=0.0, triggerType="absolute", triggerSign="+", triggerFunction="median", triggerFunctionWindow=100, triggerFunctionTune=0.0, minimalWidth=1.0, minimalGap=1.0, apexFunctionAxis1="center", apexFunctionAxis2="max"):
  '''How this function works:
  1)for every point it calculates some levelFunction
  2)if this point is higher than level -- it adds to a peaks array
  3)peaks array is filtered and splitted to a number of single peaks
  4)each peak is processed to calculate it`s apex
  '''
  #init
  print("find_peaks_trivial debug start ================================")

  X = series[axis1]
  F = series[axis2]
  L = [] #array of level points
  peaksPossible = {"X":[], "F":[]} #array of all points possibly included peaks
  peaks = [] #array of peaks {"X":[], "F":[], "apexX":0.0, "apexY":0.0} 

  #verifications
  if triggerFunctionWindow + 1 >= len(X):
    triggerFunctionWindow = len(X) - 1

  #choise of desired levelFunction

  for I in range(len(X)): #I is general iterator here
    #choose windows
    #approximation: if part of the window protrudes from a possible data series -- protruding part is filled by a mirrored part of data series from this side
    halfWindowWidth = int(triggerFunctionWindow/2)
    xWindow = []
    fWindow = []
    leftI = I - halfWindowWidth
    rightI = I + halfWindowWidth
    if leftI < 0:
      xWindow = X[abs(leftI) - 1:: -1]
      fWindow = F[abs(leftI) - 1:: -1]
      xWindow.extend(X[:I])
      fWindow.extend(F[:I])
    else:
      xWindow.extend(X[leftI:I])
      fWindow.extend(F[leftI:I])

    if rightI > len(X):
      xWindow.extend(X[I:])
      fWindow.extend(F[I:])
      xWindow.extend(X[: len(X) -1 - (rightI - len(X)) :-1])
      fWindow.extend(F[: len(F) -1 - (rightI - len(F)) :-1])
    else:
      xWindow.extend(X[I:rightI])
      fWindow.extend(F[I:rightI])
    x = X[I]
    f = F[I]
     
    if len(fWindow) != triggerFunctionWindow:
      print("len(fWindow):",len(fWindow))
      print(leftI, rightI)
  
    if triggerFunction == "median":
      level = median(fWindow)
    elif triggerFunction == "average":
      level = average(fWindow)
    

    L.append(level)
    if triggerType == "absolute":
      if triggerSign == "+":
        if f >= level + triggerValue:
          peaksPossible["X"].append(x)
          peaksPossible["F"].append(f)
      else:
        if f <= level - triggerValue:
          peaksPossible["X"].append(x)
          peaksPossible["F"].append(f)
    
    if triggerType == "relative":
      if triggerSign == "+":
        if f >= level*triggerValue:
          peaksPossible["X"].append(x)
          peaksPossible["F"].append(f)
      else:
        if f <= level*triggerValue:
          peaksPossible["X"].append(x)
          peaksPossible["F"].append(f)

  #filter and split peaks array to single peaks
  #It is a state machine.
  #peaksPossible = {"X":[], "F":[]} #array of all points possibly included peaks
  #peaks = [] #array of peaks {"X":[], "F":[], "apexX":0.0, "apexY":0.0} 
  
  #peak sides:
  leftI = 0
  rightI = 0
  
  peakStarted = False
  peakStopped = False

  for I in range(1, len(peaksPossible["F"])): #I is main iterator here
    x = peaksPossible["X"][I]
    dx = peaksPossible["X"][I] - peaksPossible["X"][I - 1]
    if not peakStarted:
      leftI = I
      peakStarted = True
    if dx > minimalGap:
      rightI = I
      peakStopped = True
    if x - peaksPossible["X"][leftI] < minimalWidth:
      peakStopped = False

    if peakStopped:
      peaks.append({"X":peaksPossible["X"][leftI:rightI], "F":peaksPossible["F"][leftI:rightI]})
      peakStopped = False
      peakStarted = False

  #find the apex location
  for I in range(len(peaks)):
    peak = peaks[I]
    tmpX = peak["X"]
    tmpF = peak["F"]
    
    if apexFunctionAxis2 == "median":
      tmp2F = F.copy()
      tmp2F.sort()
      apexF = tmp2F[int(len(tmp2F)/2)]
    elif apexFunctionAxis2 == "min":
      apexF = min(tmpF)
    elif apexFunctionAxis2 == "max":
      apexF = max(tmpF)
    elif apexFunctionAxis2 == "width":
      apexF = tmpX[-1] - tmpX[0]
    else: # apexFunctionAxis2 == "center":
      apexF = tmpF[0] + (tmpF[-1] - tmpF[0])/2


    if apexFunctionAxis1 == "apex location":
      apexX = tmpX[tmpF.index(nearest(tmpF, apexF))]
    elif apexFunctionAxis1 == "median":
      tmp2X = tmpX.copy()
      tmp2X.sort()
      apexX = tmp2X[int(len(tmp2X)/2)]
    elif apexFunctionAxis1 == "left":
      apexX = tmpX[0]
    elif apexFunctionAxis1 == "right":
      apexX = tmpX[-1]
    else:# apexFunctionAxis1 == "center":
      apexX = tmpX[0] + (tmpX[-1] - tmpX[0])/2


    peak["apexX"] = apexX
    peak["apexF"] = apexF
    peaks[I] = peak
  
  apexes = { axis1:[], axis2:[],"axes":series["axes"], "tasks":apexesTasks, "name":(series["name"]+"_apexes")}
  levels = {axis1:X, axis2:L, "axes":series["axes"], "tasks": levelTasks, "name":(series["name"]+"_level")}
  peaksResult = {axis1:peaksPossible["X"], axis2:peaksPossible["F"], "axes":series["axes"], "tasks": peaksTasks, "name":(series["name"]+"_peaks")}
  
  for peak in peaks:
    apexes[axis1].append(peak["apexX"])
    apexes[axis2].append(peak["apexF"])
  
  print("find_peaks_trivial debug stop  ================================")
  return (levels, peaksResult, apexes)




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
      #print(series)
      if "normalise" in task:
        dataset[sIndex] = normalise(series, task["normalise"]["axis"],task["normalise"]["min"], task["normalise"]["max"])
      elif "sort" in task:
        dataset[sIndex] = sort(series, task["sort"]["axis"], task["sort"]["direction"])
      elif "cut" in task:
        dataset[sIndex] = cut(series, task["cut"]["axis"], task["cut"]["left"], task["cut"]["right"])
      elif "1find peaks advanced" in task: #this line is broken. Delete first '1' in a string
        tmpTask = task["find peaks advanced"]
  #def FindPeaksAdvanced(series, axis1, axis2, peaksTasks=[], der1Tasks=[], der2Tasks=[], int2Tasks=[], int1Tasks=[],  level=0.0, levelSign="+", levelType="peakY",  minWidth=0):
        dataset.extend(find_peaks_advanced(
                                          series, 
                                          axis1=tmpTask["axis1"], 
                                          axis2=tmpTask["axis2"], 
                                          peaksTasks=tmpTask["peaks tasks"], 
                                          der1Tasks=[], 
                                          der2Tasks=[], 
                                          int2Tasks=[], 
                                          int1Tasks=[],  
                                          level=tmpTask["minimal level"],
                                          levelSign=tmpTask["minimal level sign"],
                                          levelType=tmpTask["minimal level trigger"],
                                          minWidth=tmpTask["minimal peak width"],
                                          axis2Type=tmpTask["result peak type"],
                                          axis2Value=tmpTask["result peak function"]
                                        ))
      elif "find peaks trivial" in task:
        tmpTask = task['find peaks trivial']
        dataset.extend(find_peaks_trivial(
                                              series, 
                                              axis1=tmpTask["axis1"], 
                                              axis2=tmpTask["axis2"],
                                              apexesTasks=tmpTask["apexes tasks"], 
                                              peaksTasks=tmpTask["peaks tasks"], 
                                              levelTasks=tmpTask["level tasks"],
                                              triggerValue=tmpTask["trigger value"], 
                                              triggerType=tmpTask["trigger type"], 
                                              triggerSign=tmpTask["trigger sign"],
                                              triggerFunction=tmpTask["trigger function"], 
                                              triggerFunctionWindow=tmpTask["trigger function window"], 
                                              triggerFunctionTune=tmpTask["trigger function tune"], 
                                              minimalWidth=tmpTask["minimal peak width"],
                                              minimalGap=tmpTask["minimal gap between peaks"],
                                              apexFunctionAxis1=tmpTask["apex function axis1"],
                                              apexFunctionAxis2=tmpTask["apex function axis2"]
                                            ))
      elif "plot" in task:
        plot(chart, series, task)
  chart.show()



main()

#TODO: add checking of incoming json config. All missed fields should be filled by default values. User should be warned of all invalid fields.
#TODO: add immutability of incoming data to every function


exit()
