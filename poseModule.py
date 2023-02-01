import cv2
import mediapipe as mp
import time



from main import mpPose, img, pose, mpDraw, cap, results


class poseDetector( ):
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        upBody = upBody
        smooth = smooth
        detectionCon = detectionCon
        trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,
                                     self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if not results.pose_landmarks:
            return
        if draw:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            return img

    def getPosition(self, img, draw=True):
        lmList =[]

        if self.results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
             h , w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id,cx,cy])
            if draw:
                cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture('PoseVideo.mp4')
    previousTime = 0
    detector = poseDetector()


while True:
    success, img = cap.read()

    img=detector.findPose(img)
    lmList =detector.getPosition(img,draw=False)
    print(lmList[14])
    currentTime = time.time( )

    cv2.circle(img,(lmList[14][1],  lmList[14][2]),15,(0,0,225),cv2.FILLED)
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if __name__ == "__main__":
        main( )
