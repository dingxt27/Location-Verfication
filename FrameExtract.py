
import cv2
import os
dirname = '/home/dingxt/PycharmProjects/BigdataProject/Location-Verfication/frames/'+format(2,'04d')
#os.mkdir(dirname)

vidcap = cv2.VideoCapture('/home/dingxt/PycharmProjects/BigdataProject/Location-Verfication/video/new/6/VID_20190331_133713.mp4')
success,image = vidcap.read()


def getFrame(sec,count):
  vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
  hasFrames, image = vidcap.read()
  if hasFrames:
    cv2.imwrite(os.path.join(dirname,'%d.jpg'%count),image)  # save frame as JPG file
    #print('yes')
  return hasFrames


sec = 0
frameRate = 0.5
count = 39
success = getFrame(sec,count)
while success:
  sec = sec + frameRate
  sec = round(sec, 2)
  success = getFrame(sec,count)
  count +=1

#count = 0
#delayTime = 100
#while success:
#  cv2.imwrite(os.path.join(dirname,'%d.jpg'%count),image)
#  #cv2.imwrite("/home/parallels/BigdataProject/frames/BSB/frame%d.jpg" % count, image)     # save frame as JPEG file
#  success,image = vidcap.read()
#  cv2.waitKey(delayTime)
#  print('Read a new frame: ', success)
#  count += 1