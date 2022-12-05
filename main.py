#/usr/bin/python3
import json
import plotly.graph_objs as go
from sys import argv
from math import exp, log, sin, cos, pi
import random
import os



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
  if "marker" in plotSettings:
    markerSets = plotSettings["marker"]
    chart.add_trace(go.Scatter(x=series["X"],
                               y=series["Y"], 
                               name=series["name"], 
                               showlegend=plotSettings["showlegend"], 
                               mode=plotSettings["style"], 
                               marker_symbol=markerSets["symbol"], 
                               marker= {"color":markerSets["color"], 
                                        "size":markerSets["size"], 
                                        "line":{"color":markerSets["color2"], 
                                                "width":markerSets["line width"]
                                                }
                                        }))
  else:
    chart.add_trace(go.Scatter(x=series["X"], y=series["Y"], name=series["name"], showlegend=plotSettings["showlegend"], mode=plotSettings["style"], line={"color":plotSettings["color"]}))

def file_import(filename, fileFormat, attrs):
  buff = {"X":[], "Y":[]}
  buff["name"] = attrs["name"]
  buff["tasks"] = attrs["tasks"]
  f = open(filename, "r")
  skipI = fileFormat["skip_strings"]
  try:
    splitter = fileFormat["splitter"]
  except KeyError:
    splitter = ""
  while skipI > 0:
    f.readline()
    skipI -= 1
  line = f.readline()
  line = line.strip()
  while line != '':
    if splitter != "":
      line_parsed = line_parse(line, fileFormat["string"], splitter=splitter)
    else:
      line_parsed = line_parse(line, fileFormat["string"])
    buff["X"].extend(line_parsed["X"])
    buff["Y"].extend(line_parsed["Y"])
    #print(line_parsed)
    line = f.readline()
    line = line.strip()
  f.close()
  return buff

def line_parse(line, lineFormat, splitter=""):
  res = {"X":[], "Y":[]}
  if splitter != "":
    line = line.split(splitter)
  else:
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
    if (abs(maxV - minV) >= 1e-10):
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
    


def derivative_adv(series, axis1, axis2, noShift=False):
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
  der1 = derivative_adv(series, axis1, axis2)
  der1.update({"axes":series["axes"], "tasks":der1Tasks, "name":(series["name"] + "_der1")})
  der2 = derivative_adv(der1,  axis1, axis2)
  der2.update({"axes":series["axes"], "tasks":der2Tasks, "name":(series["name"] + "_der2")})
  
  #look for der2 zero crossings (double: up/down -- down/up) then integrate it twice between two zero crossings
  
  zeroCrosses = {"i":[], "type":[] } #type means: -1 -- up/down, 1 -- down/up
  for i in range(len(der2[axis2]) - 1):
    prev = der2[axis2][i]
    curr = der2[axis2][i + 1]
    threshold = 0.0
    if prev <= -1*threshold and curr >= threshold:
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
    if zeroCrosses["type"] == []:
      noMorePeaks = True
      break
    #try:
    #    tmp_var_1 = bool(zeroCrosses["type"][i] != 1)
    #except IndexError:
    #    print("AAAAAAAAA")
    #    print(i)
    #    print(zeroCrosses)
    #    raise IndexError
    #tmp_var_1 = bool(zeroCrosses["type"][i] != 1)
    while zeroCrosses["type"][i] != 1:
      i += 1
      #print(zeroCrosses, i)
      #print("noMorePeaks:", noMorePeaks)
      #print("length:", len(zeroCrosses["i"]), "  i =", i)
      #print("length type:", len(zeroCrosses["type"]), "  i =", i)
      if i >= (len(zeroCrosses["type"]) - 1):
        noMorePeaks = True
        #print("break")
        break
      #try:
      #  tmp_var_1 = bool(zeroCrosses["type"][i] != 1)
      #except IndexError:
      #  print("BBBBBBBBB")
      #  print(i)
      #  print(zeroCrosses["type"])
      #  raise IndexError
    #This is not actual because unmatching zero crosses deleted some lines ago
    #if i == 0:
    #  zeroCrosses["type"].pop(0)
    #  zeroCrosses["i"].pop(0)
    #  continue
    try: #handle nonmatching zeroCrosses (fuck them off)
      if i != 0:
        leftI = zeroCrosses["i"][i-1]
        rightI = zeroCrosses["i"][i]
        #print(i, [leftI, rightI])
    
        zeroCrosses["i"].pop(i-1)
        zeroCrosses["type"].pop(i-1)
        zeroCrosses["i"].pop(i-1)
        zeroCrosses["type"].pop(i-1)
        if (rightI - leftI > 3):
    
          peaks["peaks"].append({axis1:series[axis1][leftI:rightI], axis2:series[axis2][leftI:rightI], "der2":der2[axis2][leftI:rightI], "der1":der1[axis2][leftI:rightI], "peak"+axis1:0.0, "peak"+axis2:0.0, "width":(series[axis1][rightI] - series[axis1][leftI])})
      else:
        zeroCrosses["i"].pop(i-1)
        zeroCrosses["type"].pop(i-1)
    except IndexError:
      noMorePeaks = True
  


  for peakI in range(len(peaks["peaks"])):
    peak = peaks["peaks"][peakI]
    #print(peaks)
    
    #integrate here 
    int1 = integrate({axis1:peak[axis1], axis2:peak["der2"]}, axis1, axis2)
    int2 = integrate({axis1:peak[axis1], axis2:int1[axis2]}, axis1, axis2)
    int3 = integrate({axis1:peak[axis1], axis2:int2[axis2]}, axis1, axis2)

    #approximate peak location
    
    peakAxis1Found = False
    for i in range(len(int1[axis2])- 1):
      a = int1[axis2][i]
      b = int1[axis2][i + 1]
      if (a > 0.0) and (b < 0.0):
        peakAxis1Found = True
        print(a,b, i)
        peak["peak" + axis1] = peak[axis1][i] + (peak[axis1][i + 1] - peak[axis1][i])*(a/(a - b))
    if not peakAxis1Found:
      peak["peak" + axis1] = peak[axis1][0] + (peak[axis1][-1] - peak[axis1][0])/2.0
    
    #calculate peak height
    peak["peak" + axis2] = max(int2[axis2])

    peak["int1"] = int1[axis2]
    peak["int2"] = int2[axis2]
    peak["int3"] = int3[axis2]

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
    if axis2Type == "area":
      resultPeaks[axis2].append(peak["int3"][-1])

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
  #print("find_peaks_trivial debug start ================================")

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
      print('"find peaks trivial" alert')
      print("\tlen(fWindow):",len(fWindow))
      print("\t from", leftI, "to", rightI)
  
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
    if I == (len(peaksPossible["F"]) - 1): 
      #print("aaa", peakStarted)
      rightI = I
      peakStopped = True
    #if x - peaksPossible["X"][leftI] < minimalWidth:
    #  print("minimal width")
    #  peakStopped = False
    if peakStopped:
      #print(peaksPossible["X"][leftI], peaksPossible["X"][rightI-1])
      peaks.append({"X":peaksPossible["X"][leftI:rightI], "F":peaksPossible["F"][leftI:rightI]})
      #print(peaks[-1])
      peakStopped = False
      peakStarted = True
      leftI = I


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
  if not "axes" in series:
    series["axes"] = (axis1, axis2)
  apexes = { axis1:[], axis2:[],"axes":series["axes"], "tasks":apexesTasks, "name":(series["name"]+"_apexes")}
  levels = {axis1:X, axis2:L, "axes":series["axes"], "tasks": levelTasks, "name":(series["name"]+"_level")}
  peaksResult = {axis1:peaksPossible["X"], axis2:peaksPossible["F"], "axes":series["axes"], "tasks": peaksTasks, "name":(series["name"]+"_peaks")}
  
  for peak in peaks:
    apexes[axis1].append(peak["apexX"])
    apexes[axis2].append(peak["apexF"])
  #print("find_peaks_trivial debug stop  ================================")
  return (levels, peaksResult, apexes)

def denoise(series, axis1="X", axis2="Y", function="median", windowPoints=50, windowArgument=0, functionTune1=0.0, functionTune2=0.0, name="_denoised"):
  #result = {axis1:[x for x in series[axis1]], axis2:[], "tasks":tasks, "name":name} #forks the data
  result = [] #in-place job
  X = series[axis1]
  F = series[axis2]
  for I in range(len(X)): #I is main iterator here
    xWindow = []
    fWindow = []
    if windowArgument == 0:
      halfWindow = windowPoints // 2

    #choose windows
    #approximation: if part of the window protrudes from a possible data series -- protruding part is filled by a mirrored part of data series from this side
      leftI = I - halfWindow
      rightI = I + halfWindow
    else:
      leftI=I
      rightI=I
      currArg = X[I]
      while abs(currArg - X[leftI]) <= windowArgument:
        leftI -= 1
        if leftI < 0:
          break
      while abs(X[rightI] - currArg) <= windowArgument:
        rightI += 1
        if rightI + 1 >= len(X):
          break
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

    if len(fWindow) != windowPoints:
      print('"find peaks trivial" alert')
      print("\tlen(fWindow):",len(fWindow))
      print("\t from", leftI, "to", rightI)

    if function == "median":
      level = median(fWindow)
    elif function == "average":
      level = average(fWindow)
    elif function == "weighted":
      level = weight(xWindow, fWindow, functionTune1)
    result.append(level)
  for i in range(len(result)):
    series[axis2][i] = result[i]

  return series
  
def weight(X, Y, distancePower):
  """returns triangle-weighted average"""
  result = 0
  #search for the center of X
  tmp = X.copy()
  tmp.sort()
  center = tmp[len(tmp)//2]
  diameter = abs(tmp[0] - tmp[-1])/2
  if diameter == 0:
    for y in Y:
      result += y
    result /= len(Y)
  else:
    for (x,y) in zip(X,Y):
      #result += y*1/((2*3.14159265)**0.5)*2.71828183**(-0.5*abs(x-center)**2)
      result += y*(2 - 2*abs(x - center)/diameter)/len(Y)
  return result

def derivative(series, axis1="X", axis2="Y", tasks=[]):
  """forks series and caculate it`s first derivative"""
  dY = []
  dX = []
  X = series[axis1]
  Y = series[axis2]
  for I in range(1, len(X)):
    if (X[I] - X[I-1])!=0:
      #dX.append((X[I] + X[I-1])/2)
      dX.append(X[I])
      dY.append((Y[I] - Y[I-1])/(X[I] - X[I-1]))
  result = {axis1:dX, axis2:dY, "tasks":tasks, "name":series["name"]+"_derivative", }
  return [result]


def chart_style(chart, style):
  if style["chart autosize"]:
    chart.update_layout(
      autosize=True
    )
  else:
    chart.update_layout(
      autosize=False,
      title_x=style["title x"],
      title_y=style["title y"],
      width=style["chart width"],
      height=style["chart height"],
      margin=style["margins"]
    )
  chart.update_layout(
    title_font_size=style["title font size"],
    title=style["title"],
    title_xanchor=style["title xanchor"],
    title_yanchor=style["title yanchor"],
    font_size=style["font size"],
    showlegend=style["showlegend"],
    xaxis_title=style["xaxis title"],
    xaxis_title_font_size=style["xaxis title font size"],
    yaxis_title=style["yaxis title"],
    yaxis_title_font_size=style["yaxis title font size"],
    plot_bgcolor=style["plot background"],
    xaxis={
      "showline":style["xaxis showline"],
      "linewidth":style["xaxis line width"],
      "linecolor":style["xaxis line color"],
      "ticks":style["xaxis ticks location"],
      "color":style["xaxis color"]
    },
    yaxis={
      "showline":style["xaxis showline"],
      "linewidth":style["xaxis line width"],
      "linecolor":style["xaxis line color"],
      "ticks":style["xaxis ticks location"],
      "color":style["xaxis color"]
    },
    legend={
      "yanchor":style["legend yanchor"],
      "y":style["legend y"],
      "xanchor":style["legend xanchor"],
      "x":style["legend x"]
    }
  )

def draw_verticals(marks, series, axis1="X", axis2="Y", color="black", text="X", fontSize=10, textAngle=0, aBottom=True, Bottom=0, aTop=True, Top=1):
  minY = min(series[axis2])
  maxY = max(series[axis2])
  for x in series[axis1]:
    annotation = {}
    annotation["font"] = {"size":fontSize, "color":color}
    annotation["arrowcolor"] = color
    annotation["text"] = x
    annotation["textangle"] = textAngle
    annotation["x"] = x
    annotation["y"] = minY
    annotation["ax"] = x
    annotation["ay"] = maxY
    annotation["xanchor"] = "left"
    print(annotation)
    marks.append[annotation]
  return marks 


def annotate(toAnnotate, series, axis1, axis2, text="X", color="black" , angle=0, fontSize=10, aX=0, aY=10):
  for x, y, text in zip(series[axis1], series[axis2], series[text]):
    toAnnotate["X"].append(x)
    toAnnotate["Y"].append(y)
    toAnnotate["color"].append(color)
    toAnnotate["angle"].append(angle)
    toAnnotate["font size"].append(fontSize)
    toAnnotate["text"].append(str(round(text, 1)))
    #toAnnotate["text"].append("adfgqws")
    toAnnotate["aX"].append(aX)
    toAnnotate["aY"].append(aY)
  return toAnnotate


def annotations_apply(chart, signs):
  #print("annotations_apply debug start ==================================")
  annotations = []

  for x, y, text, color, fontSize, ax, ay, angle in zip(signs["X"], signs["Y"], signs["text"], signs["color"], signs["font size"], signs["aX"], signs["aY"], signs["angle"]):
    annotation = {}
    annotation["font"] = {"size":fontSize, "color":color}
    annotation["arrowcolor"] = color
    annotation["text"] = text
    annotation["textangle"] = angle
    annotation["x"] = x
    annotation["y"] = y
    annotation["ax"] = ax
    annotation["ay"] = -1*ay
    annotation["xanchor"] = "left"
    annotations.append(annotation)

  annotations.sort(key=lambda a: a["x"], reverse=True)

  #a magic constant to bind plot "points" to pixels
  A = 5.0
  #another magic constant to bind font size points to pixels
  B = 5

  firstLabel = True
  for I in range(1, len(annotations)):
    ann = annotations[I]
    prevAnn = annotations[I - 1]
    dX = prevAnn["x"] - ann["x"]
    prevY = prevAnn["y"]
    prevaY = prevAnn["ay"]
    Y = ann["y"]
    #calculate the height of the text fields + gap between them
    textH = 2 + (ann["font"]["size"]* ann["text"].count("<br>") + prevAnn["font"]["size"]* prevAnn["text"].count("<br>"))/2
    
    #calculate a value to move upwards current label
    shiftY = -1*(prevY*A + -1*prevaY + B*textH/cos(ann["textangle"]*pi/180) - B*dX*sin(ann["textangle"]*pi/180) - Y*A)
    #print(dX*A, (1/2) * ann["font"]["size"] * cos(ann["textangle"]*pi/180) * max([len(substr) for substr in ann["text"].split("<br>")]))
    #first condition: check if it is nessesary to apply any shifts
    #last condition: check if a distance between current and previous label is big enough to contain current label without any vertical shifts
    if ((shiftY < ann["ay"]) and (dX*A < ((1/2) * ann["font"]["size"] * cos(ann["textangle"]*pi/180) * max([len(substr) for substr in ann["text"].split("<br>")])))):
      annotations[I]["ay"] = shiftY
  for ann in annotations:
    chart.add_annotation(ann)
    #print(ann)
  #print("annotations_apply debug stop  ==================================")
  return chart

def compare_apexes(series, pointsInc, seriesAxis1="X", seriesAxis2="Y", pointsAxis1="X", pointsAxis2="Y", axis1B=-1, axis2B=1.0, resultTasks=[], mainPoints=False, multiPointing=False, stickyPoints=False):
  """compares(overlaps) two incoming data series"""
  print("compare_apexes debug start=================================")
  print("pointsInc")
  print(pointsInc)
  result = {seriesAxis1:[], seriesAxis2:[]}
 
  roundFactor = 3
  apexesX = [round(value, roundFactor) for value in series[seriesAxis1]]
  apexesY = [round(value, roundFactor) for value in series[seriesAxis2]]
  pointsX = [round(value, roundFactor) for value in pointsInc[pointsAxis1]]
  pointsY = [round(value, roundFactor) for value in pointsInc[pointsAxis2]]
  
  points = [(x,y) for x,y in zip(pointsX, pointsY)]
  apexes = [(x,y) for x,y in zip(apexesX, apexesY)]

  apexMapping = {} #map apex to points
  
  for apex in apexes:
    for point in points:
      if ((axis1B > abs(apex[0] - point[0])) and (axis2B > abs(apex[1] - point[1]))):
        print(apex, point, axis1B, abs(round(apex[0] - point[0], 3)), axis2B, abs(round(apex[1] - point[1], 3)), "------------------")
        if apex in apexMapping:
          apexMapping[apex].append(point)
        else:
          apexMapping[apex] = [point]
      else:
        print(apex, point, axis1B, abs(round(apex[0] - point[0], 3)), axis2B, abs(round(apex[1] - point[1], 3)))
        pass
  
  
  
  if not multiPointing:
    for apex, points in apexMapping.items():
      points.sort(key=lambda x: (abs(x[0] - apex[0]))) #sorts points by it`s distance to apex
      nearestPoint = points[0]
      apexMapping[apex] = nearestPoint

  if stickyPoints:
    for apex in apexMapping:
      apexMapping[apex] = [apex]


  if not ("marked" in series):
    series["marked"] = [0 for item in apexes]
  marked = series["marked"]

  if mainPoints:
    for i in range(len(marked)):
      for apex in apexMapping:
        if (apexesX[i], apexesY[i]) == apex:
          series["marked"][i] -= 1
  else:
    for i in range(len(marked)):
      if marked[i] < 0:
        for apex in apexMapping:
          if ((apexesX[i], apexesY[i]) == apex):
            apexMapping.pop(apex)
            break

  for apex, point in apexMapping.items():
    #print(apex, ":", point)
    for p in point:
      result[seriesAxis1].append(p[0])
      result[seriesAxis2].append(p[1])

  result["axes"] = [seriesAxis1, seriesAxis2]
  result["tasks"] = resultTasks
  result["name"] = series["name"] + "&" + pointsInc["name"] 

  print("result:")
  print(result)
  print(series["marked"])
  
  print("compare_apexes debug stop =================================")
  return series, pointsInc, result

#TODO: rewrite line formatting to be more powerful
def dump_to_file(series, fileName, lineFormat="x y"):
  print("dumping", series["name"], "to", fileName, ":")
  try:
    os.system("rm "+ fileName)
  except Exception:
    pass
  with open(fileName, 'a') as f:
    print("\tfile opened")
    f.write(series["name"])
    f.write("\n")
    lineFormat = lineFormat.upper()
    lineFormat = lineFormat.split()
    dataRows = []
    for axis in lineFormat:
      dataRows.append(series[axis])
    for i in range(min([len(row) for row in dataRows])):
      line = ''
      for row in dataRows:
        line += str(row[i] + round((random.random() - random.random()), 10)) #outputs noizy data for some fun
        #line += str(row[i])
        line += ' '
      line += '\n'
      f.write(line)
  print("\tdone")

def copy(series, name='', tasks=[]):
  series = series.copy()
  series["name"] = name
  series["tasks"] = tasks
  return series

def settings_check(settings):
  """this function checks tasks for syntax missing and restore missings to default values"""
  for seriesI in range(len(settings)):
    series = settings[seriesI]
    if "chart settings" in series:
      series = {"chart settings":series["chart settings"]} #cleanup dictionary (flushes all but "chart settings")
      series = plot_settings(series)
      settings[seriesI] = series
    elif "filename" in series:
      series = series_settings_check(series)
      settings[seriesI] = series
  return settings

def series_settings_check(series, isFile=True):
  if isFile:
    try:
      if not (type(series["filename"]) is str):
        print('Config file error\n Series "filename" should be a string')
    except KeyError:
      print('Config file error\n Nessesary field "filename" not found')
      exit()
    filename = series["filename"]
    if "format" in series:
      if not (type(series["format"]) is dict):
        print('Config file error\n Series file "format" should be a dictionary')
      else:
        try:
          if not (type(series["format"]["skip_strings"]) is int):
            print("Config file error:", name, '\n "format/skip_strings" should be an integer')
          elif series["format"]["skip_strings"] < 0:
            print("Config file error:", name, '\n "format/skip_strings" should be non-negative')
        except KeyError:
          print('Config file error\n Nessesary field "format/skip_strings" not found')
          exit()
        try:
          if not (type(series["format"]["string"]) is str):
            print("Config file error:", name, '\n "format/string" should be a string')
        except KeyError:
          print('Config file error\n Nessesary field "format/string" not found')
          exit()
    else:
      print('Config file error\n Nessesary field "format" not found')
      exit()
  try:
    if not (type(series["name"]) is str):
      print('Config file error\n Series "name" should be a string')
  except KeyError:
    print('Config file error\n Nessesary field "name" not found')
    exit() 
  name = series["name"]
  try:
    if not (type(series["tasks"]) is list):
      print('Config file error\n Series "tasks" should be a list')
    else:
      series["tasks"] = tasks_check(series["tasks"])
  except KeyError:
    print('Config file error\n Nessesary field "tasks" not found')
    exit() 
  return series

def tasks_check(tasks, depth=0):
  '''This function check tasks fields. Missing fields are added and filled by default values.
  Returns fixed tasks or raises exception
  '''
  #TODO: add user alerts
  '''taskTemplate = {
  }
  '''
  taskTemplate = {
    "normalise":{
      "min":0.0,
      "max":1.0,
      "axis":["Y", "X"]
    },
    "sort":{
      "axis":["X", "Y"],
      "direction":["up", "down"]
    },
    "cut":{
      "axis":["X", "Y"],
      "left":0.0,
      "right":1.0
    },
    "find peaks advanced":{
      "axis1":["X", "Y"],
      "axis2":["Y", "X"],
      "peaks tasks":"tasks",
      "der1Tasks":"tasks",
      "der2Tasks":"tasks",
      "int1Tasks":"tasks",
      "int2Tasks":"tasks",
      "minimal level":0.0,
      "minimal level sign":["+", "-"],
      "minimal level trigger":["peakY", "peakX" "der2", "der1", "int1", "int2", "X", "Y", "width"],
      "minimal peak width":0.0,
      "result peak type":["Y", "X", "der1", "der2", "int1", "int2", "peakX", "peakY", "width"],
      "result peak function":["min", "max"]
    },
    "find peaks trivial":{
      "axis1":["X", "Y"],
      "axis2":["Y", "X"],
      "apexes tasks":"tasks",
      "peaks tasks":"tasks",
      "level tasks":"tasks",
      "trigger value":0.0,
      "trigger type":["absolute", "relative"],
      "trigger sign":"+",
      "trigger function":["median", "average"],
      "trigger function window":100,
      "trigger function tune":0.0,
      "minimal peak width":1.0,
      "minimal gap between peaks":1.0,
      "apex function axis1":["center", "apex location", "median", "left", "right"],
      "apex function axis2":["max", "min", "median", "width", "center"]
    },
    "annotate":{
      "axis1":["X", "Y"],
      "axis2":["Y", "X"],
      "text":"",
      "color":"black",
      "label angle": -45,
      "font size": 9,
      "aX":0,
      "aY":10
    },
    "compare points":{
      "series axis1":["X", "Y"],
      "series axis2":["Y", "X"],
      "points axis1":["X", "Y"],
      "points axis2":["Y", "X"],
      "axis1 scope": 0.5,
      "axis2 scope": 0.5,
      "result tasks":"tasks",
      "main points":[True, False],
      "multiple points":[False, True],
      "sticky points":[True, False]
    },
    "dump":{
      "file name":"dump.txt",
      "format":"X Y"
    },
    "copy":{
      "new name":"Name me, please!",
      "tasks":"tasks"
    },
    "delete":{
      "name":"Name me, please!"
    },
    "rename":{
      "new name":"Name me, please!"
    }
  }
  
  
  for taskI in range(len(tasks)):
    task = tasks[taskI]
    #print(task, '\n')
    for taskName, taskFields in taskTemplate.items():
      if taskName in task:
        for field in taskFields:
          #print(json.dumps(task, indent=2))
          if not field in task[taskName]: #if field is missing
            if taskFields[field] == 'tasks':
              task[taskName][field] = []
              print('added missing field "'+ field +'" at', "'" + str([key for key in task.keys()][0]) +"'", '. Default value is', task[taskName][field])
            elif type(taskFields[field]) in (str, int, float):
              task[taskName][field] = taskFields[field]
              print('added missing field "'+ field +'" at', "'" + str([key for key in task.keys()][0]) +"'", '. Default value is', task[taskName][field])
            elif type(taskFields[field]) in (list, tuple):
              task[taskName][field] = taskFields[field][0]            
              print('added missing field "'+ field +'" at', "'" + str([key for key in task.keys()][0]) +"'", '. Default value is', task[taskName][field])
          else: #if field existing
            if taskFields[field] == 'tasks':
              print("recursive (depth = " + str(depth + 1)+ ") checking", '"'+ taskName +'"[' + field + ']')
              task[taskName][field] = tasks_check(task[taskName][field], depth=(depth+1))
            elif ((type(taskFields[field]) is not type(task[taskName][field])) and (not (type(taskFields[field]) in (list, tuple)))):
              print('fixed wrong parameter type at "' + taskName + '"[' + field + ']', 'from', '"' + str(task[taskName][field]) + '"', 'to', '"' + str(taskFields[field]) + '"')
              task[taskName][field] = taskFields[field]
            elif type(taskFields[field]) in (list, tuple):
              if task[taskName][field] not in taskFields[field]:
                print('parameter' , '"' + str(task[taskName][field]) + '"', 'is not allowed at', '"' + taskName+ '"[' + field + '].', 'Changed from', '"' + str(task[taskName][field]) + '"', 'to', '"' + str(taskFields[field][0]) + '"')
                task[taskName][field] = taskFields[field][0]
    tasks[taskI] = task
  return tasks

def plot_settings(settings):
  types = { #TODO: fill it by field names
    "int":["title font size", "font size", "xaxis title font size", "yaxis title font size", "chart width", "chart height", "xaxis line width", "yaxis line width", ],
    "bool":["showlegend", "chart autosize", "xaxis showline", "yaxis showline"],
    "float":["title x", "title y", "legend y", "legend x"],
    "str":["title", "title xanchor", "title yanchor", "xaxis title", "yaxis title", "plot background", "xaxis line color", "xaxis ticks location", "xaxis color", "yaxis line color", "yaxis ticks location", "yaxis color", "legend yanchor", "legend xanchor"]
  }
  settings = settings["chart settings"]
  abort = False
  for t, fields in types.items():
    for field in fields:
      if field in settings:
        if t == "int":
          if not (type(settings[field]) is int):
            path = "'chart settings/" + field + "'"
            print("Config file error\n Please check type of section:", path, "it should be int(..., -1, 0, 1, ...)")
            abort = True
        elif t == "bool":
          if not (type(settings[field]) is bool):
            path = "'chart settings/" + field + "'"
            print("Config file error\n Please check type of section:", path, "it should be bool (true, false)")
            abort = True
        elif t == "float":
          if not (type(settings[field]) is float):
            path = "'chart settings/" + field + "'"
            print("Config file error\n Please check type of section:", path, "it should be float (..., -1, 0, 1, ..., -0.1, 0.0, 10.3, ...)")
            abort = True
        elif t == "str":
          if not (type(settings[field]) is str):
            path = "'chart settings/" + field + "'"
            print("Config file error\n Please check type of section:", path, 'it should be any string ("...")')
            abort = True
  if abort:
    print("Aborting.")
    exit()
  return {"chart settings":settings}

  

def main():
  try:
    taskFile = open(argv[1], 'r')
  except BaseException:
    print("Error. Please provide the actual name of tasks file.")
    return -1
  tasks = json.load(taskFile)
  taskFile.close()
  if False: #switch for debug purposes
    tasksBackup1 = open("back1_"+ argv[1], 'w')
    json.dump(tasks, tasksBackup1, indent=2, separators=(',', ':'))
    tasksBackup1.close()
  
    tasks = settings_check(tasks)
    print(json.dumps(tasks, indent=2))

    tasksBackup2 = open("back2_"+ argv[1], 'w')
    json.dump(tasks, tasksBackup2, indent=2, separators=(',', ':'))
    tasksBackup2.close()
  #print(tasks)
  dataset = []
  for series in tasks:
    if "chart settings" in series:
      chartSettings = series
    else:
      dataset.append(file_import(series["filename"], series["format"], series))
  #print(dataset)
  chart = go.Figure()
  labels = {"X":[], "Y":[], "color":[], "angle":[], "font size":[], "text":[], "aX":[], "aY":[]}
  for series in dataset:
    series["axes"] = ["X", "Y"] #TODO: add support of more than 2 axes
    sIndex = dataset.index(series)
    for task in series["tasks"]:
      #print("\n++++++++++++++++++++++++++++++++++++++++++++++++\n")
      #print(task)
      #print(series)
      if "normalise" in task:
        dataset[sIndex] = normalise(series, task["normalise"]["axis"],task["normalise"]["min"], task["normalise"]["max"])
      elif "sort" in task:
        dataset[sIndex] = sort(series, task["sort"]["axis"], task["sort"]["direction"])
      elif "cut" in task:
        dataset[sIndex] = cut(series, task["cut"]["axis"], task["cut"]["left"], task["cut"]["right"])
      elif "derivative" in task:
        dataset.extend(derivative(
                                  series, 
                                  axis1=task["derivative"]["axis1"],
                                  axis2=task["derivative"]["axis2"],
                                  tasks=task["derivative"]["tasks"],
                                  ))
      elif "find peaks advanced" in task: #this line is broken. Delete first '1' in a string
        tmpTask = task["find peaks advanced"]
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
      elif "annotate" in task:
        tmpTask = task['annotate']
        labels = annotate(
                        labels, 
                        series,
                        axis1=tmpTask["axis1"], 
                        axis2=tmpTask["axis2"],
                        text=tmpTask["text"],
                        color=tmpTask["color"],
                        angle=tmpTask["label angle"],
                        fontSize=tmpTask["font size"],
                        aX=tmpTask["aX"],
                        aY=tmpTask["aY"]
                     )
      elif "compare points" in task:
        tmpTask = task["compare points"]
        pointSeries = {"X":[], "Y":[], "name":'', "tasks":[]}
        for ser in dataset:
          if ser["name"] == tmpTask["points name"]:
            pointSeries = ser
        seriesNew, pointsNew, pointsOverlapped = compare_apexes(
                      series, 
                      pointsInc=pointSeries, 
                      seriesAxis1=tmpTask["series axis1"], 
                      seriesAxis2=tmpTask["series axis2"], 
                      pointsAxis1=tmpTask["points axis1"], 
                      pointsAxis2=tmpTask["points axis2"], 
                      axis1B=float(tmpTask["axis1 scope"]), 
                      axis2B=float(tmpTask["axis2 scope"]), 
                      resultTasks=tmpTask["result tasks"],
                      mainPoints=tmpTask["main points"],
                      multiPointing=tmpTask["multiple points"],
                      stickyPoints=tmpTask["sticky points"]
                      )
        dataset[sIndex] = seriesNew
        #print(pointsOverlapped)
        dataset.append(pointsOverlapped)
      elif "dump" in task:
        tmpTask = task["dump"]
        dump_to_file(series, tmpTask["file name"], lineFormat=tmpTask["format"])
      elif "copy" in task:
        if not (task["copy"]["new name"] in [s["name"] for s in dataset]):
          dataset.append = copy(series, name=task["copy"]["new name"], tasks=task["copy"]["tasks"])
        else:
          print 
      elif "delete" in task:
        for s in dataset:
          if s["name"] == task["delete"]["name"]:
            dataset.remove(s)
            break
        else:
          print('Tried to delete "' + name + '", but it is not found. Continuing ...')
      elif "rename" in task:
        dataset[sIndex]["name"] = task["rename"]["new name"]
      elif "denoise" in task:
        dataset[sIndex] = denoise(series, 
                                        axis1=task["denoise"]["axis1"],
                                        axis2=task["denoise"]["axis2"],
                                        windowPoints=task["denoise"]["window"],
                                        functionTune1=task["denoise"]["tune1"],
                                        functionTune2=task["denoise"]["tune2"],
                                        function=task["denoise"]["function"],
                                        windowArgument=task["denoise"]["window argument"]
                                        )
      elif "plot" in task:
        plot(chart, series, task)
  chart = annotations_apply(chart, labels)
  chart_style(chart, chartSettings["chart settings"])
  chart.show()


if __name__ == "__main__":
  main()

#DONE: add checking of incoming json config. All missed fields should be filled by default values. User should be warned of all invalid fields.
#TODO: add immutability of incoming data to every function
#TODO: add mark stacking function. For example: if there are many marks at some site, move up some of them
#DONE: add symbol styling in "plot" function
#TODO: add "delete" function. It should delete the given points from series.
#TODO: add queue customisation in config (or smart queue manager). Now program may try to access unfinished/nonexisting "series"
#   all pairs of tasks and series are in main queue.
#   tasks to process are taken from the top of queue
#   if some task tries to use non-existing series it is moved to the bottom of queue with it`s following(recognised by names) tasks
# implementation details:
#   "rename" tasks are moved to the end of queue by default at start
#TODO: add check for same namings of different series

#мысль: можно задавать произвольное степенное уравнение, как пары степени и коэффициента. Например: 
#уравнение y = Kx + B будет {"Y":{"X": {0:B, 1:K, 2:0}}, "X": {}}
