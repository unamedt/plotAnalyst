#!/usr/bin/python3

#import plotly.express as px
import plotly.graph_objs as go
from sys import argv
from math import exp, log

# .ASC example:
'''
/*Cu2SrSnS4*/
5.00235831 63.9262536756889 7.99538952620127
5.00835831 94.7050147027756 9.73165015312283
5.01435831 102.201528756382 10.1094771752243
5.02035831 110.172566162542 10.4963120267331
5.02635831 93.3435237297208 9.66144521951663
'''
# .txt example
'''
PDF_Number: [26-1116]      Wavelength:  1.54060
Chemical formula:
 Cu2 S
Chemical name:
  Copper Sulfide

I/Icor:   0.00   QM:  C   Sys: H

 #   D_space   2Theta  Int.   H   K   L
_______________________________________

 1    3.3610   26.498    5    0   0   2
 2    3.0550   29.209    9    1   0   1
 3    2.4010   37.426   88    1   0   2
'''

def import_TXT(filename):
  f = open(filename, "r")
  data = {"compound":"", "code": "", "Q": [], "I": []}
  line = f.readline().strip()
  if line[0:10] == "PDF_Number":
    card_number = line[13:19]
    if f.readline().strip() == "Chemical formula:":
      card_compound = f.readline().strip()
      card_compound_buff = ""
      for i in range(len(card_compound)):
        if card_compound[i] != " ":
          card_compound_buff += card_compound[i]
      card_compound = card_compound_buff
      data["compound"] = card_compound
      data["code"] = card_number
    else:
      data["compound"] = "?????"
      data["code"] = card_number
    line = ' '
    while (line[0] != "_"):
      line = f.readline().strip()
#      print(line)
      if len(line) < 1:
        line = ' '
    line = f.readline()
    line = f.readline()
    while line != '':
      data_pair = LineParse(line, {"I": 3, "Q": 2})
      data["I"].append(data_pair["I"])
      data["Q"].append(data_pair["Q"])
      line = f.readline()
#      print(line)
  elif line[5] == '_':
    SkipLines(f,1)
    line = f.readline()
    if line[13:23] == 'PDF-2 Sets':
      card_number = line[1:12]
      SkipLines(f, 3)
      card_compound = f.readline().strip()
      card_compound_buff = ""
      for i in range(len(card_compound)):
        if card_compound[i] != ' ':
          card_compound_buff += card_compound[i]
      card_compound = card_compound_buff
      data["compound"] = card_compound
      data["code"] = card_number
      empty_lines = 5
      """
      SkipLines(f, 28)
      line = f.readline() 
      prev = ' '
      colons = 0
      while (colons < 3) or (colons >  13):
        colons = 0
        for curr in line:
          if (curr != ' ') and (prev == ' '):
            colons += 1
          prev = curr
        line = f.readline().strip()
      """
      __LINES__ = 10
      __LINES__ -= 2
      while __LINES__:
        line = f.readline().strip()
      #  print(line)
        if len(line) > 1:
          if line[0] == '_':
            __LINES__ -= 1
      while empty_lines != 0:
        line = f.readline()
        if line != '' and line[0] != '_':
          empty_lines = 5
#          print(line)
          data_pair = LineParseSmart(line, ["Q", "I"]) 
          for item in data_pair["I"]:
            data["I"].append(item)
          for item in data_pair["Q"]:
            data["Q"].append(item)
        else:
          empty_lines -= 1
      
  while (data["code"][-1] == "]" or data["code"][-1] == " "):
    data["code"] = data["code"][:-1]
  if data["compound"] == "Cu2SrSnS4":
    print(data)  
  return data

def LineParseSmart(line, addr):
  buff = []
  res = {"I": [], "Q": []}
  prev = ' '
  line = line.strip()
  for curr in line:
    if (curr != ' ') and (prev == ' '):
      buff.append(curr)
    elif (curr != ' ') and (prev != ' '):
      buff[-1] += curr
    prev = curr
  buff_len = len(buff)
  if buff_len == 12:
    for i in [1, 7]:
      res["I"].append(buff[i])
    for i in [2, 8]:
      res["Q"].append(buff[i])
  elif buff_len > 2 and buff_len < 6:
    for i in [1]:
      res["I"].append(buff[i])
    for i in [2]:
      res["Q"].append(buff[i])
  elif buff_len == 6:
    for i in [2, 5]:
      res["I"].append(buff[i])
    for i in [1, 4]:
      res["Q"].append(buff[i])
  elif buff_len == 8:
    if (len(buff[3]) == 1) and (len(buff[4] == 1)) and (len(buff[5]) == 1):
      for i in [1, 7]:
        res["I"].append(buff[i])
      for i in [2, 8]:
        res["Q"].append(buff[i])
    else:
      for i in [1, 4]:
        res["I"].append(buff[i])
      for i in [2, 5]:
        res["Q"].append(buff[i])
  for i in range(len(res["I"])):
    res["I"][i] = float(res["I"][i])
    res["Q"][i] = float(res["Q"][i])
  print(line, "buff_len", buff_len,  res)
  return res



def import_DESC(file_name):
  f = open(file_name, "r")
  line = f.readline().strip()
  data = {"Q":[], "text":[]}
  while (line != ''):
    for i in range(len(line)):
      if line[i] == '\t':
        separator = i
    data["Q"].append(float(line[0:separator]))
    data["text"].append(line[separator + 1:])
    line = f.readline().strip()
  #print(data)
  return data

def import_CSV(file_name, dataset):
  table= []
  ifile = open(file_name, "rt")
  string = ifile.readline()
  while string != '':
    table_string = []
    number = ''
    for char_i in range(len(string)):
      char = string[char_i]
      if ((char == '\t') or (char == '\n')):
        if number == '':
          table_string.append('')
        else:
          for i in range(len(number)):
            if number[i] == ',':
              number = number[:i] + '.' + number[i+1:]
          table_string.append(float(number))
          number = ''
      else:
        number += char
    string = ifile.readline()
    if table_string != ['']:
      table.append(table_string)
  ifile.close()

  for data_i in range(len(dataset)):
    Q_loc = dataset[data_i]["Q_loc"]
    I_loc = dataset[data_i]["I_loc"]
    for line in table:
      print(Q_loc, I_loc, line)
      Q = line[Q_loc]
      if I_loc != -1:
        I = line[I_loc]
      else:
        I = 100
      print(dataset[data_i]["name"], line, I_loc,":", I, "  ", Q_loc,":", Q)
      if not ((Q == '') or (I == '')):
        dataset[data_i]["Q"].append(Q)
        dataset[data_i]["I"].append(I)
  return(dataset)

def LineParseMultiColon(line, addr):
#  print(addr)
  buff = []
  prev = ' '
  res = {"I": [], "Q": []}
  line = line.strip()
  if len(line) > 2:
    for curr in line:
      if (curr != ' ') and (prev == ' '):
        buff.append(curr)
      elif (curr != ' ') and (prev != ' '):
        buff[-1] += curr
      prev = curr
    for col in addr["I"]:
      try:
        res["I"].append(float(buff[col]))
      except IndexError:
        None

    for col in addr["Q"]:
      try:
        res["Q"].append(float(buff[col]))
      except IndexError:
        None
  return res

def SkipLines(file, lines):
  a = ''
  for i in range(lines):
    a = file.readline()
  return file

def LineParse(line, dictionary):
  buff = []
  prev = ' '
  line = line.strip()
#  print(line)
  for curr in line:
    if (curr != ' ') and (prev == ' '):
      buff.append(curr)
    elif (curr != ' ') and (prev != ' '):
      buff[-1] += curr
    prev = curr
  return {"I": float(buff[dictionary["I"]]), "Q": float(buff[dictionary["Q"]])}

def dpt_LineParse(string):
  spacer = -1
  for i in range(len(string)):
    if (string[i] == '\t'):
      spacer = i

  Q = float(string[:spacer])
  I = float(string[spacer:])
  return [Q, I]

def import_dpt(file):
  print("read from " + file)
  #Q -- for angle (greek Teta)
  #I -- for intensity
  data = {"compound": "", "code": "??-????", "Q": [], "I": []}
  ASC = open(file, "r")
  #read compound name
  string = ASC.readline().strip()
  print(string[:2]+"#"+ string[2:-2]+ "#"+ string[-2:  ])
  if ((string[:2] == "/*" )and( string[-2:] == "*/")):
#    print("name: "+ string[2:-2])
    data["compound"] = string[2:-2]
    line = ASC.readline()
  i = 0
  while(line != ''):
#    print(i)
    i += 1
    data_pair = dpt_LineParse(line.strip())
    line = ASC.readline()
    data["Q"].append(data_pair[0])
    data["I"].append(data_pair[1])

  return data

def ASC_LineParse(string):
  space_1 = -1
  for i in range(len(string)):
    if (string[i] == ' '):
      if (space_1 == -1):
        space_1 = i
      space_2 = i

  Q = float(string[:space_1])
  I = float(string[space_1+1:space_2])
  return [Q, I]

def import_ASC(file):
  print("read from " + file)
  #Q -- for angle (greek Teta)
  #I -- for intensity
  data = {"compound": "", "code": "", "Q": [], "I": []}
  ASC = open(file, "r")
  #read compound name
  string = ASC.readline().strip()
  print(string[:2]+"#"+ string[2:-2]+ "#"+ string[-2:  ])
  if ((string[:2] == "/*" )and( string[-2:] == "*/")):
#    print("name: "+ string[2:-2])
    data["compound"] = string[2:-2]
    line = ASC.readline()
  i = 0
  while(line != ''):
#    print(i)
    i += 1
    data_pair = ASC_LineParse(line.strip())
    line = ASC.readline()
    data["Q"].append(data_pair[0])
    data["I"].append(data_pair[1])
  return data

def Max(array):
  max_val = 0.0;
  for value in array:
    if max_val < value:
      max_val = value
  return max_val

def Normalize(data, nonzero=False):
  max_I = 0.0
  min_I = 100500
  for I in data["I"]:
    if max_I < I:
      max_I = I
    if min_I > I:
      min_I = I
  if nonzero:
    min_I = 0
  for i in range(len(data["I"])):
    data["I"][i] -= min_I
    data["I"][i] *= 100.0/(max_I - min_I)
  return data

def ConstantCutter(data):
  buff = []
  for value in data["I"]:
    buff.append(value)
  window_W = 300
  median_buff = BubbleSort(buff[:window_W])
  for i in range(len(buff) - window_W):
    median_data = SlidingMedian(median_buff, buff[i], buff[i + window_W])
    data["I"][i + 100] -= median_data[1] 
    median_buff = median_data[0]
  return data 

def avg(inp):
  summ = 0.0
  i = 0
  for value in inp:
    summ += value
    i += 1
  return summ / i

def BubbleSort(buff):
  sorted = False
  while not sorted :
    sorted = True
    for i in range(len(buff) -1):
      if buff[i] > buff[i + 1]:
        sorted = False
        tmp = buff[i]
        buff[i] = buff[i + 1]
        buff[i + 1] = tmp
  return(buff)

def SortedInsert(buff, index, value):
  left = 0
  right = len(buff)
  while True:
    mid = int((left + right)//2)
    print(mid)
    if (buff[mid -1] <= value) and (value <= buff[mid+1]):
      buff.insert(mid+1, value)
      return buff
    if buff[mid] >= value:
      right = mid
    else:
      left = mid + 1
def SillySearch(array, left, right, key):
  for i in range(len(array)):
    if array[i] >= key:
      return(i)
  return(len(array))


def BinarySearch(array, left, right, key):
  mid = (left + right) //2
  if ((array[mid - 1] <= key) and (array[mid] < key)):
    return mid + 2
  if array[mid] < key:
    return BinarySearch(array, mid, right, key)
  else:
    return BinarySearch(array, left, mid + 1, key)


def SlidingMedian(buff, old, new):
#  print(old)
  buff.remove(old)
  buff.insert(SillySearch(buff, 0, len(buff), new)+2, new)
  length = len(buff)
  if length%2:
    med = buff[length//2]
  else:
    med = avg(buff[length//2 : length//2 + 1])
  return [buff, med]

def Noize(array_a):
  negative_counter = 0
  negative = 0
  print("++++++++++++++++++++++++++++++")
  #buff = BubbleSort(array_a)
  buff = array_a.copy()
  buff.sort()
  print("++++++++++++++++++++++++++++++")
  length = len(buff)
  if length%2:
    med = buff[length//2]
  else:
    med = avg(buff[length//2 : length//2 + 1])
  negative_counter = 0
  for value in array_a:
    if value < med:
      negative_counter += 1
      negative += med - value
  result = negative/negative_counter
  print("avg_noise",result)
  return result 

def FindPeaks(data, window_W=300, error=1.6, error_abs=0, return_median=False, noisy=False):
  peaks = {"Q": [], "I": []}
  if return_median:
    peaks["median"] = {"Q": [], "I": []}
  try:
    peaks["compound"]= data["compound"]
    peaks["code"]= data["code"]
  except KeyError:
    None
  inten = data["I"].copy()
#  i = int(window_W/2)
  i = 0
  if noisy:
    avg_noise = Noize(data["I"])
  else:
    avg_noise = 0
  inten += inten[-1:-1 -1*int(window_W):-1]
  avg_value = avg(inten[:window_W])
  median_buff = BubbleSort(inten[0: window_W])
  l = len(median_buff)
  if l%2:
    median = median_buff[l//2]
  else:
    median = avg(median_buff[l//2 : l//2 + 1])
  while(i < int(window_W/2)):
    if return_median: #for debug purposes
      peaks["median"]["I"].append(median)
      peaks["median"]["Q"].append(data["Q"][i])
    if (error_abs == 0) and (inten[i] > (median * error + avg_noise)):
      peaks["Q"].append(data["Q"][i])
      peaks["I"].append(data["I"][i])
    elif(error_abs != 0) and (inten[i] >= (median + error_abs + avg_noise)):
      peaks["Q"].append(data["Q"][i])
      peaks["I"].append(data["I"][i])
    i += 1
  while(i +1  < len(data["Q"])):
    #print(i)
    window = inten[int(i - window_W/2) :int(i + window_W/2)]
    median_data = SlidingMedian(median_buff, inten[i - int(window_W/2)], inten[i + int(window_W/2)])
    median_buff = median_data[0]
    median = median_data[1]
    if return_median: #for debug purposes
      peaks["median"]["I"].append(median)
      peaks["median"]["Q"].append(data["Q"][i])
    if (error_abs == 0) and (inten[i] > median*error + avg_noise):
      peaks["Q"].append(data["Q"][i])
      peaks["I"].append(data["I"][i])
    elif(error_abs != 0) and (inten[i] >= (median + error_abs + avg_noise)):
      peaks["Q"].append(data["Q"][i])
      peaks["I"].append(data["I"][i])
    i += 1
  return peaks

def SplitPeaks(data, step=0.006, max_gap=5, min_gap=30, min_width=0, highest_is_peak=False):
  #  print(data)
  #step -- average distance between nearest points (angle)
  peaks = {"Q": [], "I": []}
  try:
    peaks["compound"]= data["compound"]
    peaks["code"]= data["code"]
  except KeyError:
    None
  prev_angle = 0
  left = data["Q"][0]
  right = left
  prev_right = -100500.0
  prev_left = -100500.0
  for angle in data["Q"]:
    if ((right + max_gap*step < angle) or (data["Q"].index(angle) == (len(data["Q"]) - 1))):
      if ((right - left) >= min_width*step):
        if (left - prev_right) >= step*min_gap:
          left_i = int(data["Q"].index(left))
          right_i = int(data["Q"].index(right))
          is_rising = False
          is_falling = False
          #check if it a wall (a side of a peak)
          for i in range(left_i, right_i):
            if data["I"][i] > data["I"][i + 1]:
              is_falling = True
            else:
              is_rising = True
          print(data["Q"][left_i], data["Q"][right_i], is_rising, is_falling)
          if (not (is_rising ^ is_falling) or (right_i - left_i < 3)) :
          #if True:
            prev_right = right
            if left_i == right_i:
              I_max = data["I"][left_i]
              I_max_i = left_i
            else:
              I_max = Max(data["I"][left_i: right_i])
              I_max_i = data["I"].index(I_max)
            peaks["I"].append(I_max)
            if highest_is_peak:
              peaks["Q"].append(data["Q"][I_max_i])
            else:
              peaks["Q"].append((left + right)/2)
        else:
          left_i = int(data["Q"].index(prev_left))
          right_i = int(data["Q"].index(right))
          prev_right = right
          if left_i == right_i:
            peaks["I"][-1] = max(data["I"][left_i], peaks["I"][-1])
          else:
            peaks["I"][-1] = Max(data["I"][left_i: right_i])
          #peaks["Q"][-1] = (prev_left + right)/2
      prev_left = left
      left = angle
    right = angle
  return peaks

def nearest(array, item):
  delta = 100500
  delta_prev = 100500;
  #print(array, item)
  for i in range(len(array)):
    delta_prev = delta
    delta = abs(array[i] - item)
    if delta_prev < delta:
      #print(i)
      return i
  return(len(array) - 1)


def FindPeaksAdvanced(data, min_der_2=0.0, min_width=0, fake_hight=False):
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
      #der_2_zero["I"].append(I_1 - I_2)
      der_2_zero["I"].append(-100)
      der_2_zero["Q"].append(Q)
    elif ((I_1 >= min_der_2) and (I_2 <= -1 * min_der_2)):
      #der_2_zero["I"].append(I_1 - I_2)
      der_2_zero["I"].append(100)
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
  if fake_hight:
    #moves peak highest point to the nearest point of data
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

def GaussSumm(data, c=1.0, k=0.0, y0=0.0, start=25.0, stop=500.0, step=0.5):
  result = {"I":[], "Q":[]}
  for i in range(int((stop-start)/step)):
    result["I"].append(0.0)
    result["Q"].append(start + step*i)

  for peak_i in range(len(data["I"])):
    a = data["I"][peak_i]
    b = data["Q"][peak_i]
    #c = c
    for point_i in range(len(result["I"])):
      x = result["Q"][point_i]
      g = a*exp(-((x - b)**2)/(2*c**2))
      result["I"][point_i] += ( g + k*x + y0)
  return result
  


def PeaksOverlap(data1, data2, use_text=False, min_range=0.2):
  peaks = {"Q":[], "I":[], "text":[], "compound": "", "code": "", "symbol":"", "size": "", "color":"", "color2":""}
  #print(data1)
  #print(data2)
  for peak1 in data1["Q"]:
    for peak2 in data2["Q"]:
      if abs(peak1 - peak2) <= min_range:
        peak1_i = data1["Q"].index(peak1)
        peak2_i = data2["Q"].index(peak2)
        peaks["Q"].append(peak1)
        peaks["I"].append(data1["I"][peak1_i]*1.0)
  #      print(peak2_i)
        if use_text:
          peaks["text"].append(data2["text"][peak2_i])
#        peaks["name"] = data1["name"]+" & "+data2["name"]
  try:      
    peaks["compound"] = data2["compound"]
  except KeyError:
    None
  try:      
    peaks["code"] = data2["code"]
  except KeyError:
    None
  try:      
    peaks["symbol"] = data2["symbol"]
  except KeyError:
    None
  try:      
    peaks["size"] = data2["size"]
  except KeyError:
    None
  try:      
    peaks["color"] = data2["color"]
  except KeyError:
    None
  try:      
    peaks["color2"] = data2["color2"]
  except KeyError:
    None
  #print(peaks)
  return peaks

def Cut(data, left, right):
  error_prev = left
  left_i = -1
  right_i = -1
  for i in range(len(data["Q"])):
    error_curr = abs(left - data["Q"][i])
    if (error_curr > error_prev) and (left_i == -1):
      left_i = i
    error_prev = error_curr
  error_prev = right
  for i in range(len(data["Q"])):
    error_curr = abs(right- data["Q"][i])
    if (error_curr > error_prev) and (right_i == -1):
      right_i = i
    error_prev = error_curr
  print("left_i", left_i,"right_i", right_i)
  data["Q"] = data["Q"][left_i:right_i]
  data["I"] = data["I"][left_i:right_i]
#  print(data)
  return data

def CutRight(data, border):
  i = 0
  while i < len(data["I"]):
    if data["Q"][i] > 150:
      data["I"].pop(i)
      data["Q"].pop(i)
    else:
      i+= 1
  return data

def PeaksSort(data, number, lowest_I=''):
  sorted = False
  while not sorted :
    sorted = True
    for i in range(len(data["I"]) -1):
      if data["I"][i] < data["I"][i + 1]:
        sorted = False
        tmp_I = data["I"][i]
        tmp_Q = data["Q"][i]
        data["I"][i] = data["I"][i + 1]
        data["Q"][i] = data["Q"][i + 1]
        data["I"][i + 1] = tmp_I
        data["Q"][i + 1] = tmp_Q
  if lowest_I != '':
    number = nearest(data["I"], lowest_I)
  data["I"] = data["I"][:number]
  data["Q"] = data["Q"][:number]
  return(data)

def MergeCards(cards_stack):
  print("merging cards")
  merged = True
  while not merged:
    merged = True
    for card_a in cards_stack:
      for card_b in cards_stack:
        if (card_a["compound"] == card_b["compound"]) and (card_a != card_b):
          merged = False
          card_b["code"] += " " + card_a["code"]
          for i in range(len(card_a["Q"])):
            card_b["I"].append(card_a["I"][i])
            card_b["Q"].append(card_a["Q"][i])
          cards_stack.remove(card_a)
  return cards_stack

def PeaksDelete(data1, data2):
  peaks = data2["Q"]
  for Q in peaks:
    if data1["Q"].count(Q):
      i = data1["Q"].index(Q)
      data1["Q"].pop(i)
      data1["I"].pop(i)
  return data1 

def PeaksOverlapScore(data1, data2, offset, min_dist=1.0, power=2):
  score = 0.0
  overlaps = 0
  for i in range(len(data1["Q"])):
    for j in range(len(data2["Q"])):
      if abs(data1["Q"][i]+offset - data2["Q"][j]) <= min_dist:
        overlaps += 1
        I1 = data1["I"][i]/100.0
        I2 = data2["I"][j]/100.0
        dI = abs(I1 - I2)
        dQ = abs(data1["Q"][i] + offset - data2["Q"][j])/min_dist
        curr_score= ((1 - dQ)*(1 - dI))**(power)
#        curr_score = 1 - (dI**2 + dQ**2)**0.5
#        print(dI, dQ, curr_score)
        score += curr_score
  score = score/overlaps
  return score

def OffsetPeaksRecognise(data, error=0.05):
  result = {"Q":[], "I":[]}
  prev_I = data["I"][0]
  curr_I = data["I"][1]
  for i in range(len(data["I"]) - 2):
    i += 2
    next_I = data["I"][i]
    if (curr_I > prev_I) and (curr_I > next_I):
      if ((curr_I - next_I + curr_I - prev_I) > error):
        result["I"].append(curr_I)
        result["Q"].append(data["Q"][i - 1])
    prev_I = curr_I
    curr_I = next_I
  return(result)

def PeaksRecognise(args, SHOW_ALL=True):
  DIAGRAM_SHIFT = 120 
  SIGNS_DIST = 5 
  main_compound = args[0]
  args = args[1:]
  RFA_stack = []
  
  while args[0] == "-s":
    args = args[1:]
    RFA = import_ASC(args[0])
    args = args[1:]
    #RFA = ConstantCutter(RFA)
    #print("noize:", Noize(RFACutted["I"]))
    RFACutted = Cut(RFA, 15.0, 80.0)
    RFACutted = Normalize(RFACutted)
    #def FindPeaks(data, window_W=300, error=1.6, error_abs=0, return_median=False):
    #def SplitPeaks(data, step=0.006, max_gap=5, min_gap=30, min_width=0, highest_is_peak=False):
    RFACaps =  FindPeaks(RFACutted, error_abs=4.5, noisy=True)
    RFAPeaks =  SplitPeaks(RFACaps, max_gap=10)
    RFAPeaks = PeaksSort(RFAPeaks, 50)
    RFA_stack.append({"RFA":RFA, "RFAPeaks":RFAPeaks, "overlaps": "", "main_overlaps":"", "RFACaps":RFACaps})
  #moving diagram up
  for i in range(len(RFA_stack)):
    for j in range(len(RFA_stack[i]["RFA"]["I"])):
      RFA_stack[i]["RFA"]["I"][j] += i * DIAGRAM_SHIFT 
    for j in range(len(RFA_stack[i]["RFAPeaks"]["I"])):
      RFA_stack[i]["RFAPeaks"]["I"][j] += i * DIAGRAM_SHIFT
    for j in range(len(RFA_stack[i]["RFACaps"]["I"])):
      RFA_stack[i]["RFACaps"]["I"][j] += i * DIAGRAM_SHIFT

  #import and merge cards
  results = []
  cards = []
  colors = ["#000", "#0FF", "#F0F", "#F00", "#0F0", "#00F"]
  symbols = ["circle", "diamond", "square","cross", "x", "triangle-up", "triangle-down", "star", "hexagram", "star-square", "star-diamond", "diamond-tall", "diamond-wide", "hourglass", "bowtie", "circle-cross", "circle-x", "square-cross", "square-x", "asterisk", "hash", "y-up", "y-down"]
  i = 0
  args.sort(reverse=False)
  sign_stack = {"Q":[], "N":[]} # array, which stores information about positions of signs
  for card_name in args:
    if card_name[-4:] == ".ASC":
      card = import_ASC(card_name)
    elif card_name[-4:] == ".txt":
      card = import_TXT(card_name)
    elif card_name == "dummy.ASC":
      None
    print("file:", card_name, " card:", card["compound"], card["code"], " dots:", len(card["Q"]))
  #merge different cards with same compounds
    merged = False
    for other_card in cards:
      if other_card["compound"] == card["compound"]:
        merged = True
        other_card["code"] += ' ' + card["code"]
        for j in range(len(card["Q"])):
          other_card["I"].append(card["I"][i])
          other_card["Q"].append(card["Q"][i])
    if not merged:
      cards.append(card)
  main_card = {"Q": [], "I": [], "compound": "", "code": ""}
  for card_i in range(len(cards)):
    cards[card_i]["symbol"] = symbols[card_i%len(symbols)]
    cards[card_i]["size"] = 8
    cards[card_i]["color"] = colors[len(colors) - 1 - ((card_i+6)%len(colors))]
    cards[card_i]["color2"]= colors[(card_i+ 2)%len(colors)]
  for card in cards:
    if card["compound"] == main_compound:
      main_card = card
      cards.remove(main_card)
  #print(cards)
  #print(main_card)
  for RFA_i in range(len(RFA_stack)):
    RFAPeaks = RFA_stack[RFA_i]["RFAPeaks"]
    #print(RFA_i, len(RFAPeaks["I"]))
    main_overlaps = PeaksOverlap(RFAPeaks, main_card)
    RFAPeaks = PeaksDelete(RFAPeaks, main_overlaps)
    results = [main_overlaps]
    for card in cards:
    #  print(card)
      card = Normalize(card)
      overlaps = PeaksOverlap(RFAPeaks, card)
      results.append(overlaps)
    '''  for j in range(len(results)):
  
      results[j]["symbol"] = 
      results[j]["size"] = 
      results[j]["color"] = colors[len(colors) - 1 - ((i+6)%len(colors))]
      results[j]["color2"] = colors[(i+ 2)%len(colors)]
      i += 1
    '''
    for overlap in results:
      prev = 0
      for i in range(len(overlap["Q"])):
        curr = overlap["Q"][i]
        if abs(curr - prev) < 0.01:
          overlap["I"][i] = max(overlap["I"][i], overlap["I"][i - 1])
          overlap["Q"][i - 1] = -1
          overlap["I"][i - 1] = -1
          overlap["Q"][i] = (prev + curr)/2
        prev = curr
      i = 0
      while i < len(overlap["I"]):
        if overlap["I"][i] == -1:
          overlap["I"].remove(-1)
          overlap["Q"].remove(-1)
        else:
          i += 1
      for i in range(len(overlap["Q"])):
        peak = overlap["Q"][i]
  #      print("peak:", peak)
        if sign_stack["Q"].count(peak) == 0:
          sign_stack["Q"].append(peak)
          sign_stack["N"].append(1)
        else:
          sign_stack["N"][sign_stack["Q"].index(peak)] += 1
        overlap["I"][i] += sign_stack["N"][sign_stack["Q"].index(peak)] * SIGNS_DIST
    if SHOW_ALL:
      unknown  = {"Q": [], "I": [], "compound": "неизвестно","code": ""}
      unknown["symbol"] = "triangle-down"
      unknown["size"] = 8
      unknown["color"] = "black"
      unknown["color2"] = "black" 
      for peak in RFAPeaks["Q"]:
        if sign_stack["Q"].count(peak) == 0:
          peak_i = RFAPeaks["Q"].index(peak)
          unknown["Q"].append(peak)
          unknown["I"].append(RFAPeaks["I"][peak_i] + SIGNS_DIST)
      results.append(unknown)
    else:
      results[0]["symbol"] = "triangle-down"
      results[0]["size"] = 8
      results[0]["color"] = "black"
      results[0]["color2"] = "black" 
    RFA_stack[RFA_i]["results"] = results

  '''
    for i in range(len(overlap["Q"])):
      overlap["Q"][i] = overlap["Q"][i]//0.2 * 0.2
  '''
  #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  #print(RFA_stack)
  #exit()




  chart = go.Figure()
  for RFA_i in range(len(RFA_stack)):
    RFA = RFA_stack[RFA_i]["RFA"]
    RFACaps = RFA_stack[RFA_i]["RFACaps"]
    RFAPeaks = RFA_stack[RFA_i]["RFAPeaks"]
    #chart.add_trace(go.Scatter(x= RFA["Q"], y= RFA["I"], name= RFA["compound"], showlegend=True))
    chart.add_trace(go.Scatter(x= RFA["Q"], y= RFA["I"], name= RFA["compound"], line={"color":"black"}, showlegend=True))
    #chart.add_trace(go.Scatter(x= RFACaps["Q"], y= RFACaps["I"], name= RFA["compound"] + " peaks", mode= "markers"))
    #chart.add_trace(go.Scatter(x= RFAPeaks["Q"], y= RFAPeaks["I"], name= RFA["compound"] + " caps",  mode= "markers"))
    '''
    for card in cards:
      chart.add_trace(go.Scatter(x= card["Q"], y= card["I"], name= card["name"], mode="markers", showlegend=True))
    '''
  used_comps = [] 
  for RFA_i in range(len(RFA_stack)):
    results = RFA_stack[RFA_i]["results"]
    for overlaps in results:
  #    print(overlaps)
      card_showlegend = True
      for comp in used_comps:
        if (comp["compound"] == overlaps["compound"]):
          card_showlegend = False
          overlaps["color"] = comp["marker"]["color"]
          overlaps["color2"] = comp["marker"]["color2"]
          overlaps["size"] = comp["marker"]["size"]
          overlaps["symbol"] = comp["marker"]["symbol"]
      if card_showlegend:
        used_comps.append({
          "compound":overlaps["compound"], 
          "marker":{
            "symbol":overlaps["symbol"], 
            "color":overlaps["color"], 
            "size": overlaps["size"], 
            "color2": overlaps["color2"]
            }
        })
      
      chart.add_trace(go.Scatter(x= overlaps["Q"], y= overlaps["I"], name= overlaps["compound"] + " " + overlaps["code"], mode= "markers", showlegend= card_showlegend,  marker_symbol=overlaps["symbol"], marker= {"color": overlaps["color"],"size": overlaps["size"], "line":{"color":overlaps["color2"], "width":1}}))

  chart.update_layout(
    title_font_size=30,
    title="",
    title_xanchor="center",
    title_yanchor="top",
    title_x=0.5,
    title_y=0.95,
    font_size=18,
    showlegend=True,
    xaxis_title="$$2\Theta, град.$$",
    xaxis_title_font_size=20,
    yaxis_title="I, произв.ед.",
    yaxis_title_font_size=20,
    autosize=False,
    width=1000,
    height=750,
    plot_bgcolor = "#FFF",
    margin={"l":50, "r":50, "b": 10, "t" : 10, "pad": 4},
    xaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
    },
    yaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
    },
    legend={
      "yanchor":"top",
      "y":0.75,
      "xanchor":"right",
      "x":1.25
    }
  )
  chart.show()

def PeaksCompare(args):
  RFA1 =import_ASC(args[0])
  RFA1 = ConstantCutter(RFA1)
  RFA1Cutted = Cut(RFA1, 15.0, 80.0)
  RFA1Cutted = Normalize(RFA1Cutted)
  RFA1Caps =  FindPeaks(RFA1Cutted)
  RFA1Peaks =  SplitPeaks(RFA1Caps)
  RFA1Peaks = PeaksSort(RFA1Peaks, 50)

  RFA2 = import_ASC(args[1])
  RFA2 = ConstantCutter(RFA2)
  RFA2Cutted = Cut(RFA2, 15.0, 80.0)
  RFA2Cutted = Normalize(RFA2Cutted)
  RFA2Caps =  FindPeaks(RFA2Cutted)
  RFA2Peaks =  SplitPeaks(RFA2Caps)
  RFA2Peaks = PeaksSort(RFA2Peaks, 50)
  


  start = -10
  stop = 10
  step = 0.01
  offset = start
  scores = {"I": [], "Q": []}
#  PeaksOverlapScore(RFA1Peaks, RFA2Peaks, 0, min_dist=0.1)
#  exit() 
  while offset <= stop:
    scores["I"].append(PeaksOverlapScore(RFA1Peaks, RFA2Peaks, offset, power=2, min_dist=1))
    scores["Q"].append(offset)
    offset += step
#  print("scores:", scores)
  
#  scores_peaks= OffsetPeaksRecognise(scores, error=0.025)
#  scores = ConstantCutter(scores)
  scores = Normalize(scores)
  scores_peaks=FindPeaks(scores, window_W = 300, error=1.05)
  #print(scores_peaks)
  scores_caps = scores_peaks
# SplitPeaks(data, step=0.006, max_gap=5, min_gap=30, min_width=0):
  scores_peaks = SplitPeaks(scores_peaks, step=step, max_gap = 20, min_gap=21, min_width=0)
  #print(scores_peaks)
  scoring = go.Figure()
  scoring.add_trace(go.Scatter(x = scores["Q"], y= scores["I"], name="scores"))
  scoring.add_trace(go.Scatter(x = scores_peaks["Q"], y= scores_peaks["I"], mode="markers", name="scores peaks"))
#  scoring.add_trace(go.Scatter(x = scores_caps["Q"], y= scores_caps["I"], mode="markers", name="scores caps"))
  scoring.update_layout(
    title_font_size=30,
    title_xanchor="center",
    title_yanchor="top",
    title="Степень совпадения",
    xaxis_title="относительный сдвиг в градусах",
    plot_bgcolor = "#FFF",
    yaxis_title="относительное расстояние между вершинами графиков",
    showlegend=False
   ) 
  scoring.show()
#  exit()





  max_score= max(scores["I"])
#  max_score_offset = scores["Q"][scores["I"].index(max_score)]
  max_score_offset = 5.1
  chart = go.Figure()
  chart.add_trace(go.Scatter(x= RFA2["Q"], y= RFA2["I"], name= RFA2["compound"], line={"color":"black"}))
  for offset in scores_peaks["Q"]:
    RFA_offseted = {"I":[], "Q":[]} 
    for I in RFA1["I"]:
      RFA_offseted["I"].append( I )
    for Q in RFA1["Q"]:
      RFA_offseted["Q"].append(Q + offset)
    chart.add_trace(go.Scatter(x= RFA_offseted["Q"], y= RFA_offseted["I"], name= RFA1["compound"] + "; сдвиг = "+ str(offset), visible="legendonly"))
  #chart.add_trace(go.Scatter(x= RFACaps["Q"], y= RFACaps["I"], name= RFA["compound"] + " peaks", mode= "markers"))
  #chart.add_trace(go.Scatter(x= RFAPeaks["Q"], y= RFAPeaks["I"], name= RFA["compound"] + " caps",  mode= "markers"))
  '''
  for card in cards:
    chart.add_trace(go.Scatter(x= card["Q"], y= card["I"], name= card["name"], mode="markers", showlegend=True))

  '''

  chart.update_layout(
    title_font_size=30,
    title="",
    title_xanchor="center",
    title_yanchor="top",
    title_x=0.5,
    title_y=0.95,
    font_size=14,
    showlegend=True,
    xaxis_title="$$2\Theta, град.$$",
    xaxis_title_font_size=20,
    yaxis_title="I, произв.ед.",
    yaxis_title_font_size=20,
#    autosize=False,
    autosize=True,
#    width=1000,
#    height=750,
    plot_bgcolor = "#FFF",
#    margin={"l":50, "r":50, "b": 10, "t" : 10, "pad": 4},
    xaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
    },
    yaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
#    },
#    legend={
#      "yanchor":"top",
#      "y":0.75,
#      "xanchor":"right",
#      "x":1.25
    }
  )
  chart.show()
  chart.write_html("Наложения.html")

def PeaksWrite(source, dest, filetype="txt"):
  RFA =import_ASC(source)
  RFA = ConstantCutter(RFA)
  RFACutted = Cut(RFA, 15.0, 80.0)
  RFACutted = Normalize(RFACutted)
  RFACaps =  FindPeaks(RFACutted)
  RFAPeaks =  SplitPeaks(RFACaps)
  RFAPeaks = PeaksSort(RFAPeaks, 50)
  print(RFAPeaks)
  out_file = open(dest, "w")
  if filetype == "txt":
    out_file.write("compound: "+ RFA["compound"]+ "\n")
    out_file.write("2Q I\n")
    for i in range(len(RFAPeaks["I"])):
      string = str(RFAPeaks["Q"][i]) + " " + str(RFAPeaks["I"][i]) + "\n"
      out_file.write(string)
  elif filetype == "par":
    out_file.write(
  '''Background
               0 1           0 1           0 1           0 1           0 1  
               0 1           0 1           0 1           0 1           0 1   2   0
  Groups
   4  0  1.00000  0.00000  2.00000  0.50000  0.010000  0.000000  0.000000    0 0 1 1 0 0
  Peaks
  ''')
    for i in range(len(RFAPeaks["I"])):
      string = "  "+ str(round(RFAPeaks["Q"][i], 4)) + "      " + str(round(RFAPeaks["I"][i], 2)) + "   " + "0.2000  1     0  0  0""\n"
      out_file.write(string)

  out_file.close()

def DiagramsOverlap(files, PROCESS=True, DIAGRAM_SHIFT=150, LABEL=True, DRAW_LINES=False, gap=0):
  if DRAW_LINES:
    lines = []
    while files[0] == '--lines':
      tmp = ( import_dpt(files[1])) #draw vertical lines
      lines.append(tmp)
      files = files[2:]

  SIGNS_DIST = 5 
  dataset = []
  DESC = {"Q":[],"text":[]}
  for j in range(len(files)):
    print("#" + str(files[j][-4:]) + "#")
    if files[j][-4:] == ".dpt":
      RFA = import_dpt(files[j])
      RFA = Cut(RFA, 0, 800)
    elif args[j][-4:] == ".ASC":
      RFA =import_ASC(files[j])
    else:
      RFA = import_DESC(files[j])
    try:
      is_descr = RFA["text"]
      is_descr = True
    except KeyError:
      is_descr = False
    if is_descr:
      DESC = RFA
    else:
      RFACutted = RFA
      RFACutted = Normalize(RFACutted)
      if PROCESS:
        RFACaps =  FindPeaks(RFACutted, window_W=50, error_abs=0.5, noisy=False, return_median=True)
        RFAPeaks =  SplitPeaks(RFACaps, step=0.5, max_gap=1, min_gap=5, min_width=5, highest_is_peak=True)
        #RFAPeaks = FindPeaksAdvanced(RFACutted, min_der_2=1.0, min_width=0, fake_hight=True)["peaks"]
        RFAMedian = RFACaps["median"]
        #RFAPeaks = FindPeaksAdvanced(RFACutted)
        #RFAMedian = {"I":[], "Q":[]}
        #print(RFACaps)
        #RFAPeaks = PeaksSort(RFAPeaks, 8)
        RFAPeaks["text"] = []

        '''
    for i in range(len(RFAPeaks["I"])):
      RFAPeaks["text"].append(str(RFAPeaks["Q"][i]))
    
    for i in range(len(RFA["I"])):
      RFA["I"][i] += j * gap
    for i in range(len(RFAPeaks["I"])):
      RFAPeaks["I"][i] += j * gap
    for i in range(len(RFACaps["I"])):
      RFACaps["I"][i] += j * gap
    for i in range(len(RFAMedian["I"])):
      RFAMedian["I"][i] += j * gap
    '''
        dataset.append({"data":RFA, "peaks":RFAPeaks, "caps":RFACaps, "median":RFAMedian})
      else:
        dataset.append({"data":RFA})
      #dataset.append({"data":RFA, "peaks":RFAPeaks, "caps":RFACaps})
  if PROCESS:
    labels = []
    for Set in dataset:
      labels.append(PeaksOverlap(Set["peaks"], DESC, use_text=True, min_range=1))
      peaks = Set["peaks"]
      print(Set["data"]["compound"])
      for i in range(len(peaks["I"])):
        print(peaks["Q"][i],round(peaks["I"][i], 1))

  chart = go.Figure()
  if DRAW_LINES:
    colors = ["blue", "red"]
    color_i = 0
    for line in lines:
      for i in range(len(dataset)):
        first_in_line = True
        for point_i in range(len(line["I"])):
          line_draw = {"I":[i*150, line["I"][point_i]+ i*150], "Q":[line["Q"][point_i], line["Q"][point_i]], "name":line["compound"]}
          if ((i == 0) and first_in_line):
            first_in_line = False
            chart.add_trace(go.Scatter(x=line_draw["Q"], y=line_draw["I"], name=line_draw["name"], line={"color":colors[color_i]}, mode="lines", showlegend=True))
          else:
            chart.add_trace(go.Scatter(x=line_draw["Q"], y=line_draw["I"], name=line_draw["name"], line={"color":colors[color_i]}, mode="lines", showlegend=False))
      color_i += 1


  DESC["I"] = []
  for i in range(len(DESC["Q"])):
    #DESC["I"].append( float(DESC["text"][i]))
    DESC["I"].append(0)
  #chart.add_trace(go.Scatter(x= DESC["Q"], y=DESC["I"] , name="DESC", showlegend=False, mode="markers", marker={"size":5, "symbol":"circle", "line":{"color":"blue"}}))
  Set_i = 0
  for Set in dataset:
    RFA = Set["data"]
    for i in range(len(RFA["I"])):
      RFA["I"][i] += Set_i * DIAGRAM_SHIFT
    if PROCESS:
      peaks = Set["peaks"]
      for i in range(len(peaks["I"])):
        peaks["I"][i] += Set_i * DIAGRAM_SHIFT
      caps = Set["caps"]
      for i in range(len(caps["I"])):
        caps["I"][i] += Set_i * DIAGRAM_SHIFT
      median = Set["median"]
      for i in range(len(median["I"])):
        median["I"][i] += Set_i * DIAGRAM_SHIFT
    Set_i += 1
    chart.add_trace(go.Scatter(x= RFA["Q"], y= RFA["I"], name= RFA["compound"], line={"color":"black"}, showlegend=True))
    #chart.add_trace(go.Scatter(x= RFA["Q"], y= RFA["I"], name= RFA["compound"], showlegend=True))
    #chart.add_trace(go.Scatter(x= RFA["Q"], y= RFA["I"], name= RFA["compound"], showlegend=True, mode="markers"))
    if PROCESS:
      #chart.add_trace(go.Scatter(x= caps["Q"], y= caps["I"], name= caps["compound"] + " caps", mode= "markers", marker_color="green" ))
      #chart.add_trace(go.Scatter(x= peaks["Q"], y= peaks["I"], text= peaks["text"], textposition="top center", name= RFA["compound"] + " peaks",  mode= "markers", marker_color="red"))
      #chart.add_trace(go.Scatter(x=median["Q"], y= median["I"], name= "median", mode= "lines", line={"color":"grey"} ))
      None
      #labels labelling the peaks
  if LABEL:
    labels = []
    Set_i = 0
    for Set in dataset:
      tmp_labels = {"I":Set["peaks"]["I"].copy(), "Q":Set["peaks"]["Q"].copy()}
      #print(tmp_labels)
      tmp_labels["text"] = []
      for i in range(len(tmp_labels["I"])):
        tmp_labels["text"].append(str(round(tmp_labels["Q"][i], 1)) )
        #tmp_labels["text"].append(str(round(tmp_labels["Q"][i], 1)) + "; " + str(round(tmp_labels["I"][i] - Set_i*DIAGRAM_SHIFT, 1)))
      labels.append(tmp_labels)
      Set_i += 1

    # label the peaks
  
    label_i = 0
    print(labels)
    for label in labels:
      #for i in range(len(label["I"])):
      #  label["I"][i] += label_i * DIAGRAM_SHIFT
      label_i += 1
      #chart.add_trace(go.Scatter(x= label["Q"], y= label["I"], text= label["text"], textposition="top center", mode= "text", showlegend=False, textfont={"family": "sans serif", "size": 14, "color": "black"}))
  

      label["shift"] = []
      for i in range(len(label["I"])):
        label["shift"].append(20)
      for i in range(len(label["I"]) - 1):
        i = len(label["I"]) - 2 -i
        Q = label["Q"][i + 1]
        I = label["I"][i + 1]
        if (Q < label["Q"][i] + 40):
          if (label["I"][i] > label["I"][i + 1] + label["shift"][i+1]*1):
            #print(label["Q"][i], "True")
            None
          else:
            label["shift"][i] = label["shift"][i + 1] + 25

      print(label["shift"])
      for i in range(len(label["I"])):
        chart.add_annotation(
          x=label["Q"][i],
          y=label["I"][i],
          text=label["text"][i],
          ay = -1* label["shift"][i]

        )
      chart.update_annotations(dict(
        xref="x",
        yref="y",
        textangle=-60,
        xanchor="left",
        font={
        "size": 24
      },
      showarrow=True,
      arrowhead=7,
      ax=0,
      #ay=-20
    ))
    '''
    chart.add_annotation(
        x="800",
        y="185",
        text="2",
        font_size=40,
        textangle=0,
        ay=0
    )
    chart.add_annotation(
        x="800",
        y="35",
        text="1",
        font_size=40,
        textangle=0,
        ay=0
    )
    '''
  '''
  for card in cards:
    chart.add_trace(go.Scatter(x= card["Q"], y= card["I"], name= card["name"], mode="markers", showlegend=True))

  '''

  chart.update_layout(
    title_font_size=30,
    #title=RFA["compound"],
    title_xanchor="center",
    title_yanchor="top",
    title_x=0.5,
    title_y=0.95,
    font_size=18,
    showlegend=False,
    xaxis_title=r"$Рамановский\ сдвиг,\ (см^{-1})$",
    xaxis_title_font_size=25,
    yaxis_title="Интенсивность",
    yaxis_title_font_size=25,
    autosize=False,
    width=1000,
    height=750,
    plot_bgcolor = "#FFF",
    margin={"l":50, "r":50, "b": 10, "t" : 10, "pad": 4},
    xaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
    },
    yaxis={
      "showline":True,
      "linewidth":1,
      "linecolor":"black",
      "ticks":"outside",
      "color":"#000"
    },
    legend={
      "yanchor":"top",
      "y":1.00,
      "xanchor":"right",
      "x":1.25
    }
  )
  chart.show()

def Develop(args):
  data = import_dpt(args[0])
  data = Cut(data, 0, 450)
  #data = CutRight(data, 150)
  data = Normalize(data)
  data_set = FindPeaksAdvanced(data)
  #peaks =  PeaksSort(data_set["peaks"], 0, lowest_I=0.0005)
  peaks = data_set["peaks"]
  #return {"peaks":{"I":[], "Q":[]}, "der":[data, der_1, der_2]}
  f = open(args[0][:-4]+"_ram.pks", "w")
  for i in range(len(peaks["I"])):
    f.write(str(peaks["Q"][i]) + "\t" + str(peaks["I"][i])+ "\n")
  f.close()
  ''' 
  for i in range(1):
    peaks["Q"].append(1500+ i)
    peaks["I"].append(11209)
  '''
  #reverse = Normalize(GaussSumm(peaks, c=2.0, k=0.1, y0=1590.0, start=data["Q"][0], stop=data["Q"][-1]))
  reverse = Normalize(GaussSumm(peaks, c=5.0, start=data["Q"][0], stop=data["Q"][-1]))
  peaks = Normalize(peaks)



  chart = go.Figure()
  der_I = 0
  for data_piece in data_set["der"]:
    #if der_I != 0:
      #chart.add_trace(go.Scatter(x= data_piece["Q"], y= data_piece["I"], name=der_I,  showlegend=True, mode="markers"))
    der_I += 1
  chart.add_trace(go.Scatter(x= data["Q"], y= data["I"], name="data",  showlegend=True, mode="markers"))
  chart.add_trace(go.Scatter(x= reverse["Q"], y= reverse["I"], name="data_reversed",  showlegend=True, mode="markers"))
  chart.add_trace(go.Scatter(x= data_set["peaks"]["Q"], y= data_set["peaks"]["I"], name="peaks",  showlegend=True, mode="markers"))
  chart.update_layout(
    yaxis={
      "type":"linear"
    }
  )
  chart.show()

def Plot(args):
  dataset = [{"name":"Cu2SrSnS4", "I":[], "Q":[], "Q_loc":0, "I_loc":1}, {"name":"Cu1.9SrSnS4", "I":[], "Q":[], "Q_loc": 2, "I_loc":3}, {"name":"Cu2SrSnS4 calc", "I":[], "Q":[], "Q_loc":4, "I_loc":5}, {"name":"Cu2SrSnS4 other","I":[], "Q":[], "Q_loc":6, "I_loc":-1}]  
  dataset = import_CSV(args[0], dataset)
  
  for data_i in range(len(dataset)):
    #normalise I
    I_min = 100500
    I_max = -100500
    for I in dataset[data_i]["I"]:
      if I > I_max:
        I_max = I
      if I < I_min:
        I_min = I
    if I_min != I_max:
      for i in range(len(dataset[data_i]["I"])):
        dataset[data_i]["I"][i] = (dataset[data_i]["I"][i] - I_min)*100/(I_max - I_min)

  
  print(dataset)
  chart = go.Figure()
  for data in dataset:
    chart.add_trace(go.Scatter(x= data["Q"], y= data["I"], name=data["name"],  showlegend=True, mode="markers"))
  
  chart.update_layout(
    yaxis={
      "type":"linear"
    }
  )
  chart.show()

args = argv
task = args[1]  

args = argv[2:]
if task == "-r":  #recognise
  PeaksRecognise(args, SHOW_ALL=False)
elif task == "-c":  #compare
  PeaksCompare(args)
elif task == "-w":  #write a table to a file
  PeaksWrite(args[0], args[1], filetype="par")
elif task == "-o":  #process&overlap 2 diagramms
  DiagramsOverlap(args, PROCESS=True, LABEL=True, DRAW_LINES=True, gap=10)
elif task == "-d":
  Develop(args)
elif task == "-plot":
  Plot(args) #plots data from a CSV format (/t separated)
