import numpy as np

DEBUG = False

def locate_max(matrix):
  maxVal = None 
  row = None
  col = None

  for (rowIdx, colIdx), num in np.ndenumerate(matrix):
    if maxVal is None or num > maxVal:
      maxVal = num
      row = rowIdx
      col = colIdx

  return (maxVal, row, col)

def basic(similarityMatrix):
  mtrx = similarityMatrix
  maxVals = []
  while True:
    w, h = mtrx.shape
    if w == 0 or h == 0:
      intersection = sum(maxVals)
      width, height = similarityMatrix.shape
      union = width + height - intersection
      if DEBUG:
        print(maxVals, intersection)
        print(width, height)
        print(union)
      return intersection / union
    else:
      maxVal, row, col = locate_max(mtrx)
      if maxVal is not None:
        mtrx = np.delete(np.delete(mtrx, row, 0), col, 1)
        maxVals.append(maxVal)
      else:
        return 0.0

def smallerunion(similarityMatrix):
  mtrx = similarityMatrix
  maxVals = []
  while True:
    w, h = mtrx.shape
    if w == 0 or h == 0:
      intersection = sum(maxVals)
      width, height = similarityMatrix.shape
      union = 2 * min([width, height]) - intersection
      if DEBUG:
        print(maxVals, intersection)
        print(width, height)
        print(union)
      return intersection / union
    else:
      maxVal, row, col = locate_max(mtrx)
      if maxVal is not None:
        mtrx = np.delete(np.delete(mtrx, row, 0), col, 1)
        maxVals.append(maxVal)
      else:
        return 0.0

def average(similarityMatrix):
  mtrx = similarityMatrix
  maxVals = []
  while True:
    w, h = mtrx.shape
    if w == 0 or h == 0:
      if DEBUG:
        print(maxVals)
      return sum(maxVals) / len(maxVals)
    else:
      maxVal, row, col = locate_max(mtrx)
      if maxVal is not None:
        mtrx = np.delete(np.delete(mtrx, row, 0), col, 1)
        maxVals.append(maxVal)
      else:
        return 0.0


testData = [
  ("single 1.0-s",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]
   ])),

  ("multiple 1.0-s",
   np.matrix([
     [0.0, 1.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]
   ])),

  ("one 0.9",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [0.9, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]
   ])),

  ("one 0.8",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [0.8, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]
   ])),

  ("single 0.8-s",
   np.matrix([
     [0.0, 0.8, 0.0, 0.0],
     [0.8, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.8, 0.0],
     [0.0, 0.0, 0.0, 0.8]
   ])),

  ("0.8-s and 0.1-s",
   np.matrix([
     [0.1, 0.8, 0.1, 0.1],
     [0.8, 0.1, 0.1, 0.1],
     [0.0, 0.1, 0.8, 0.1],
     [0.0, 0.1, 0.0, 0.8]
   ])),

  ("0.7-s and 0.1-s",
   np.matrix([
     [0.1, 0.7, 0.1, 0.1],
     [0.7, 0.1, 0.1, 0.1],
     [0.0, 0.1, 0.7, 0.1],
     [0.0, 0.1, 0.0, 0.7]
   ])),

  ("06-s and 0.1-s",
   np.matrix([
     [0.1, 0.6, 0.1, 0.1],
     [0.6, 0.1, 0.1, 0.1],
     [0.0, 0.1, 0.6, 0.1],
     [0.0, 0.1, 0.0, 0.6]
   ])),

  ("one missing",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]
   ])),

  ("one missing and one 0.9",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [0.9, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]
   ])),

  ("one missing and one 0.8",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [0.8, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]
   ])),

  ("one missing and one 0.7",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0],
     [0.7, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]
   ])),
  
  ("different sizes",
   np.matrix([
     [0.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0]
   ])),

  ("'Майским утром корова щипала траву' - 'Лань на восходе ела зелень'",
   np.matrix([
     [0.3259534, 0.5844235, 0.28864202, 0.45578888],
     [0.28870443, 0.54684883, 0.333184, 0.25857088],
     [0.48130777, 0.23757665, 0.39316016, 0.31410146],
     [0.35827807, 0.22410284, 0.13277341, 0.33228177],
     [0.3674479, 0.39475724, 0.25897616, 0.62726533]
   ])),

  ("'Майским утром корова щипала траву' - 'Смотрит, как баран на новые ворота'",
   np.matrix([
     [0.23033454, 0.2774898, 0.29194513, 0.24739341],
     [0.32567796, 0.24833752, 0.2830554, 0.37499633],
     [0.2853681, 0.700481, 0.1431155, 0.30429754],
     [0.523359, 0.47411788, 0.228054, 0.26519686],
     [0.27651066, 0.38031867, 0.23215908, 0.30979127]
   ])),

  ("'Майским утром корова щипала траву' - 'Лань на восходе ела зелень' | pairs",
   np.matrix([
     [0.6243664622306824, 0.6212965846061707, 0.4828345775604248],
     [0.5986961126327515, 0.5766487121582031, 0.5055530071258545],
     [0.454327255487442, 0.34171950817108154, 0.41358691453933716],
     [0.47741633653640747, 0.3556085228919983, 0.48481741547584534]
   ])),

  ("'Майским утром корова щипала траву' - 'Смотрит, как баран на новые ворота' | pairs",
   np.matrix([
     [0.3817530870437622, 0.4193417727947235, 0.4202512800693512],
     [0.5928255319595337, 0.5641160607337952, 0.4178675413131714],
     [0.6832882761955261, 0.5748881101608276, 0.32232794165611267],
     [0.5796042084693909, 0.4972270131111145, 0.3607397675514221]
   ])),

  ("'Майским утром корова щипала траву' - 'Лань на восходе ела зелень' | allpairs",
   np.matrix([
     [0.6243664622306824, 0.45618537068367004, 0.4741503596305847, 0.6212965846061707, 0.6310870051383972, 0.4828345775604248],
     [0.635769248008728, 0.5994628071784973, 0.6139809489250183, 0.5815399289131165, 0.5939548015594482, 0.5724034905433655],
     [0.5865787267684937, 0.44822242856025696, 0.5771737694740295, 0.47896236181259155, 0.5998790264129639, 0.48023808002471924],
     [0.6017748713493347, 0.46064403653144836, 0.637637197971344, 0.544387936592102, 0.7094408273696899, 0.5928328037261963],
     [0.5986961126327515, 0.5945775508880615, 0.5159169435501099, 0.5766487121582031, 0.49977630376815796, 0.5055530071258545],
     [0.5424373149871826, 0.4392435848712921, 0.47242075204849243, 0.4689334034919739, 0.49812638759613037, 0.4085160493850708],
     [0.6248511075973511, 0.5036501884460449, 0.6016159057617188, 0.5944352746009827, 0.6833457350730896, 0.5840601325035095],
     [0.454327255487442, 0.4918595254421234, 0.5175893902778625, 0.34171950817108154, 0.36991220712661743, 0.41358691453933716],
     [0.5288230776786804, 0.5528645515441895, 0.6376530528068542, 0.45449915528297424, 0.5372576117515564, 0.5749111771583557],
     [0.47741633653640747, 0.40934258699417114, 0.5969688296318054, 0.3556085228919983, 0.5358695983886719, 0.48481741547584534]
   ])),

  ("'Майским утром корова щипала траву' - 'Смотрит, как баран на новые ворота' | allpairs",
   np.matrix([
     [0.3817530870437622, 0.4067472815513611, 0.42681506276130676, 0.4193417727947235, 0.39430153369903564, 0.4202512800693512],
     [0.5751371383666992, 0.37310028076171875, 0.4218585789203644, 0.5873494744300842, 0.5731866359710693, 0.37790030241012573],
     [0.5836699604988098, 0.5033220648765564, 0.503966212272644, 0.53226238489151, 0.47702786326408386, 0.3982219099998474],
     [0.41347604990005493, 0.37301671504974365, 0.3877340257167816, 0.4529685378074646, 0.41977083683013916, 0.38181138038635254],
     [0.5928255319595337, 0.4017418920993805, 0.5033507347106934, 0.5641160607337952, 0.602156400680542, 0.4178675413131714],
     [0.5932155847549438, 0.5232669711112976, 0.5770171284675598, 0.502679705619812, 0.5005748867988586, 0.43225839734077454],
     [0.47498035430908203, 0.43946027755737305, 0.5097642540931702, 0.47651687264442444, 0.49325186014175415, 0.46058306097984314],
     [0.6832882761955261, 0.41427740454673767, 0.48736628890037537, 0.5748881101608276, 0.5847422480583191, 0.32232794165611267],
     [0.5787466764450073, 0.3364992141723633, 0.42526328563690186, 0.5537943243980408, 0.5811148285865784, 0.34669336676597595],
     [0.5796042084693909, 0.4499374330043793, 0.494424432516098, 0.4972270131111145, 0.48737338185310364, 0.3607397675514221]
   ]))
]

# for idx, (title, m) in enumerate(testData):
#   print("----------------\n{0}.\t{1}".format(idx, title))
#   print('\tbasic:', basic(m))
#   print('\tsmallerunion:', smallerunion(m))
#   print('\taverage:', average(m))
