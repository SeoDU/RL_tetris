import pygame
from PIL import Image
from agent_tet import deepAgent
from agent_tet import blockPhotos
import os
import random
from time import sleep
import sys
import numpy as np
from matplotlib import pylab as plt

backphoto = Image.open('images/background.jpg')
pad_width, pad_height = backphoto.size

WHITE = (255,255,255)
RED = (255,0,0)

class Env:

    def __init__(self):
        self.resizeSize = (0,0)
        self.timeStep = 0
        self.velocity = 5
        self.score = 0
        self.blocksize = (pad_width/10, pad_height/20)
        self.myGrid = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0]]

        self.backGrid = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0]]

        # [1: Left, 2: Right, 3:Rotate]
        self.action_space = [1, 2, 3]
        self.action_size = 3

    def resizeImage(self,img,i):
        resize_image = img.resize(self.resizeSize)
        resize_image.save('images/Resized'+ '/' + str(i)+ '.png')

    def drawObject(self,obj,x,y) :
        global gamepad
        position = (x,y)
        gamepad.blit(obj,position)

    def drawMyGrid(self):
        for posY,rowvalue in enumerate(self.myGrid):
            for posX,colvalue in enumerate(rowvalue):
                if colvalue > 0:
                    self.drawObject(blockPhotos.block,posX*40,posY*40)

    def drawBackGrid(self):
        for posY,rowvalue in enumerate(self.backGrid):
            for posX,colvalue in enumerate(rowvalue):
                if colvalue > 0:
                    self.drawObject(blockPhotos.block_finished,posX*40,posY*40)

    def drawScore(self,count):
        global gamepad

        font = pygame.font.SysFont(None,25)
        text = font.render('Score : '+str(count),True,RED)
        gamepad.blit(text,(0,2))

    def CheckgameOver(self,myObj):
        for pos in myObj:
            if self.backGrid[pos[1]][pos[0]] == 1:
                pygame.mixer.Sound.play(explosion_sound)
                self.dispMessage('Score : ' +str(self.score))
                return True
        return False

    def textObj(self,text, font):
        textSurface = font.render(text, True, RED)
        return textSurface, textSurface.get_rect()

    def dispMessage(self,text):
        global gamepad

        largeText = pygame.font.Font('freesansbold.ttf', 70)
        TextSurf, TextRect = self.textObj(text, largeText)
        TextRect.center = ((pad_width / 2), (pad_height / 2))
        gamepad.blit(TextSurf, TextRect)
        pygame.display.update()
        sleep(2)

    def showMyObj(self,myObj):
        for pos in myObj:
            self.myGrid[pos[1]][pos[0]] = 1

    def deleteMyObj(self,myObj):
        for pos in myObj:
            self.myGrid[pos[1]][pos[0]] = 0

    # 범위밖을 벗어나는지 체크, rotation의 경우 가상의 위치를 체크
    def checkOutofRange(self,myObj,direction):

        if direction == 1:
            for pos in myObj:
                if pos[0] == 0:
                    return True
                # Test 필요
                if self.backGrid[pos[1]][pos[0] - 1] == 1:
                    return True

        elif direction == 2:
            for pos in myObj:
                if pos[0] == 9:
                    return True
                # Test 필요
                if self.backGrid[pos[1]][pos[0] + 1] == 1:
                    return True

        elif direction == 3:
            for pos in myObj:
                if pos[0] < 0 or pos[0] > 9 or pos[1] < 0:
                    return True
                # Test 필요
                if self.backGrid[pos[1]][pos[0]] == 1:
                    return True

        return False

    # direction [1: Left, 2: Right, 3:Rotate(UP), 4:Down]
    def changeMyObj(self,myObj,direction):
        if direction == 1:
            if not self.checkOutofRange(myObj,direction):
                for pos in myObj:
                    pos[0] -= 1

        if direction == 2:
            if not self.checkOutofRange(myObj,direction):
                for pos in myObj:
                    pos[0] += 1

        if direction == 3:
            # temp는 myObj의 중심
            temp = myObj[1]
            # tempObj는 회전이동 한 뒤의 위치
            tempObj = []

            # tempObj를 (0,0) 중심으로 이동
            for obj in myObj:
                tempObj.append([obj[0]-temp[0],obj[1]-temp[1]])

            # 회전이동 후 원위치로 이동
            for i in tempObj:
                y = []
                y.append(-i[1] + temp[0])
                y.append(i[0] + temp[1])
                i[0] = y[0]
                i[1] = y[1]
                y.clear()

            # 정상적이라면 이동
            if not self.checkOutofRange(tempObj,direction):
                for c,_ in enumerate(myObj):
                 myObj[c] = tempObj[c]

        if direction == 4:
            if not self.checkOutofRange(myObj,direction):
                for pos in myObj:
                    pos[1] += 1

    def clearFullRow(self):
        clearCount = 0
        clearRow = []
        for c,row in enumerate(self.backGrid):
            # 만약 col중에 하나라도 안채워져 있으면 다음 row로 넘어간다
            for i,col in enumerate(row):
                if col == 0:
                    break

                # i=9면서 break하지 않았다는건 전부 1이라는 뜻이므로
                if i == 9:
                    clearCount += 1
                    clearRow.append(c)
                    # 해당 backgrid row 를 전부 청소해준다.
                    self.backGrid[c] = [0,0,0,0,0,0,0,0,0,0]

        # clear되는 row의 윗줄은 전부다 1줄을 내린다. Ex. 17줄, 19줄
        for cRow in clearRow:
            for row in range(cRow,0,-1):
                for c,_ in enumerate(self.backGrid[row]):
                    self.backGrid[row][c] = self.backGrid[row-1][c]
            self.backGrid[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if clearCount < 3:
            self.score += clearCount * clearCount
        elif clearCount >= 3:
            self.score += clearCount * 3

    def runGame(self):
        global gamepad, background, clock

        done = False
        isReached = False
        gamepad.fill(WHITE)
        while not done:

            drawMyObj = True
            if isReached:
                agent1.nowObject = agent1.CreateObj()
                if self.CheckgameOver(agent1.nowObject):
                    done = True
                    break

                isReached = False
                self.timeStep = 0

            # Left, Right action
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                if self.timeStep % 5 == 0:
                    self.deleteMyObj(agent1.nowObject)
                    self.changeMyObj(agent1.nowObject, 1)

            if keys[pygame.K_RIGHT]:
                if self.timeStep % 5 == 0:
                    self.deleteMyObj(agent1.nowObject)
                    self.changeMyObj(agent1.nowObject, 2)

            '''
            if keys[pygame.K_UP]:
                if self.timeStep % 8 == 0:
            '''

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.deleteMyObj(agent1.nowObject)
                        self.changeMyObj(agent1.nowObject, 3)

            # down키를 누르면 속도 조절
            if keys[pygame.K_DOWN]:
                self.velocity = 30
            else:
                self.velocity = 1

            # 자동으로 밑으로 내려가게 함
            if self.timeStep / 100 > 1:     # 난이도 증가용 : self.timeStep * int(1 + self.score/10)
                shouldStop = False
                for pos in agent1.nowObject:
                    # object가 밑바닥에 닿으면 stop
                    if pos[1] == 19:
                        shouldStop = True
                        break

                    # object앞에 backGrid가 쌓여 있으면 stop
                    if self.backGrid[pos[1]+1][pos[0]] == 1:
                        shouldStop = True
                        break

                if shouldStop:
                    # myObject가 멈춘 경우 backGrid로 전환
                    for pos in agent1.nowObject:
                        self.backGrid[pos[1]][pos[0]] = 1
                    self.deleteMyObj(agent1.nowObject)
                    drawMyObj = False
                    isReached = True
                    self.clearFullRow()

                if not shouldStop:
                    self.deleteMyObj(agent1.nowObject)
                    self.changeMyObj(agent1.nowObject, 4)
                    self.timeStep = 0

            # Draw Background
            self.drawObject(background, 0, 0)

            # Draw MyObject
            if drawMyObj:
                self.showMyObj(agent1.nowObject)
            self.drawMyGrid()
            self.drawBackGrid()

            self.drawScore(self.score)

            # 내려가는 속도
            self.timeStep += self.velocity
            pygame.display.update()
            clock.tick(60)  # FPS 설정

        pygame.quit()
        quit()

    def initGame(self):
        global gamepad, background, clock
        global explosion_sound

        pygame.init()
        gamepad = pygame.display.set_mode((pad_width, pad_height))
        pygame.display.set_caption('Tetris_Test')

        ### BGM ###
        pygame.mixer.music.load('sounds/bgm.mp3')
        pygame.mixer.music.set_volume(0.1)
        pygame.mixer.music.play(-1)

        explosion_sound = pygame.mixer.Sound('sounds/explosion.wav')
        background = pygame.image.load('images/background.jpg')
        clock = pygame.time.Clock()

        self.runGame()

if __name__ == '__main__' :
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (700, 150)

    env = Env()
    agent1 = deepAgent()
    env.initGame()