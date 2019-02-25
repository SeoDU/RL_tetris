import pygame
from PIL import Image
import os
import random
from time import sleep
from copy import deepcopy
import sys
import numpy as np
from matplotlib import pylab as plt
from pygame.locals import *

backphoto = Image.open('images/background.jpg')
pad_width, pad_height = backphoto.size
newGrid = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                         ,[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]\
                             ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0,0,0,0,0,0,0,0,0,0]]
WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)

class Env:

    def __init__(self,render,play_mode):
        self.render = render
        self.play_mode = play_mode
        self.done = False

    def reset(self):
        # Game start for new env
        self.tetris = TetrisEnv()
        self.tetris.initGame(self.render,self.play_mode)
        self.done = self.tetris.done
        ret_image = self.tetris.giveImage()
        return ret_image

    def step(self,agentAction):
        self.tetris.action = agentAction

        # 우선 한번 runGame을 실행 후
        self.tetris.runGame()

        # frameGet이 True가 아니라면
        while not self.tetris.frameGet:
            # 계속 runGame을 해본다
            if self.tetris.runGame(): #만약 게임이 끝나버리면 탈출
                break

        # frameGet이 True라면 이미지 추출
        ret_image = self.tetris.giveImage()

        # 보상은 ClearRow에서의 점수로 규정
        reward = self.tetris.reward

        # 게임이 끝난 것인지 한번더 체크
        if self.tetris.done == True:
            self.done = True
            # 게임이 끝난것이라면 보상은 - 5
            reward += -5

        return ret_image,reward,self.done

class TetrisEnv:
    def __init__(self):
        self.resizeSize = (0,0)
        self.timeStep = 0
        self.velocity = 1
        self.score = 0
        self.action = 5
        self.isReached = False
        self.done = False
        self.reward = 0
        self.frameGet = False # action을 받을 때 or 시간으로 인해 한칸 내려가기 직전 Frame을 따와 진행
        self.blocksize = (pad_width/10, pad_height/20)
        self.myGrid = deepcopy(newGrid)
        self.backGrid = deepcopy(newGrid)

    def resizeImage(self,img,i):
        resize_image = img.resize(self.resizeSize)
        resize_image.save('images/Resized'+ '/' + str(i)+ '.png')

    def drawObject(self,obj,x,y) :

        position = (x,y)
        self.gamepad.blit(obj,position)

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

        font = pygame.font.SysFont(None,25)
        text = font.render('Score : '+str(count),True,RED)
        self.gamepad.blit(text,(0,2))

    def CheckgameOver(self,myObj):
        for pos in myObj:
            if self.backGrid[pos[1]][pos[0]] == 1:
                # pygame.mixer.Sound.play(explosion_sound)
                self.done = True
                self.frameGet = True
                if self.playMode:
                    self.dispMessage('Score : ' +str(self.score))
                return True
        return False

    def textObj(self,text, font):
        textSurface = font.render(text, True, RED)
        return textSurface, textSurface.get_rect()

    def dispMessage(self,text):

        largeText = pygame.font.Font('freesansbold.ttf', 70)
        TextSurf, TextRect = self.textObj(text, largeText)
        TextRect.center = ((pad_width / 2), (pad_height / 2))
        self.gamepad.blit(TextSurf, TextRect)
        pygame.display.flip()
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
                if pos[0] < 0 or pos[0] > 9 or pos[1] < 0 or pos[1] > 19:
                    return True
                # Test 필요
                #print(self.backGrid[pos[1]][pos[0]])
                try:
                    if self.backGrid[pos[1]][pos[0]] == 1:
                        return True
                except:
                    print(pos[0], pos[1], self.backGrid)
                    return True

        return False
    # direction [1: Left, 2: Right, 3:Rotate(UP), 4:Down]
    def changeMyObj(self,myObj,direction):
        self.frameGet = True

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
            self.reward = clearCount * clearCount
        elif clearCount >= 3:
            self.score += clearCount * 2
            self.reward = clearCount * 2

    def ActionOnce(self,direction):
        self.deleteMyObj(agent1.nowObject)
        self.changeMyObj(agent1.nowObject, direction)
        self.frameGet = True

    def update(self):
        global background, clock

        self.frameGet = False
        drawMyObj = True

        if self.isReached:
            agent1.nowObject = agent1.CreateObj()
            self.isReached = False
            self.timeStep = 0

        # isReached가 True라면 nowObject가 반드시 새로 생긴 후에 체크해야 함
        if self.CheckgameOver(agent1.nowObject):
            return True

        # Play Mode
        if self.playMode:
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

            # Up action
            for event in pygame.event.get():
                # if event.type == pygame.QUIT:
                #    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.deleteMyObj(agent1.nowObject)
                        self.changeMyObj(agent1.nowObject, 3)

            # down action (속도 증가)
            if keys[pygame.K_DOWN]:
                self.velocity = 30
            else:
                self.velocity = 1

        # Learning Mode
        else:
            self.ActionOnce(self.action)  # 한번 내려가기 전에 action 3번 가능
            self.velocity = 34

        # 자동으로 밑으로 내려가게 함
        if self.timeStep / 100 > 1:  # 난이도 증가용 : self.timeStep * int(1 + self.score/10)
            shouldStop = False
            for pos in agent1.nowObject:
                # object가 밑바닥에 닿으면 stop
                if pos[1] == 19:
                    shouldStop = True
                    break

                # object앞에 backGrid가 쌓여 있으면 stop
                if self.backGrid[pos[1] + 1][pos[0]] == 1:
                    shouldStop = True
                    break

            if shouldStop:
                # myObject가 멈춘 경우 backGrid로 전환
                for pos in agent1.nowObject:
                    self.backGrid[pos[1]][pos[0]] = 1
                self.deleteMyObj(agent1.nowObject)
                drawMyObj = False
                self.isReached = True
                self.clearFullRow()

            if not shouldStop:
                self.deleteMyObj(agent1.nowObject)
                self.changeMyObj(agent1.nowObject, 4)
                self.timeStep = 0


        # Draw Background
        self.gamepad.fill(BLACK)
        self.drawObject(background, 0, 0)

        # Draw MyObject
        if drawMyObj:
            self.showMyObj(agent1.nowObject)
        self.drawMyGrid()
        self.drawBackGrid()
        self.drawScore(self.score)

        # 내려가는 속도
        self.timeStep += self.velocity

        if self.render:
            pygame.display.flip()

        if self.playMode:
            clock.tick(60)  # FPS 설정

        return False

    def runGame(self):
        global background, clock
        global agent1

        done = False
        if self.playMode:
            while not done:
                if self.update():
                    pygame.quit()
                    quit()

        else:
            if self.update():
                done = True

        return done

    # game의 이미지 array를 반환
    def giveImage(self):
        #global gamepad
        if self.frameGet:
            return pygame.surfarray.array3d(self.gamepad)
        else:
            return []

    def initGame(self,render,play_mode):
        global background, clock
        global agent1
        # global explosion_sound

        self.playMode = play_mode
        self.render = render
        pygame.init()

        gpd = initPad(render)
        self.gamepad = gpd

        ### BGM ###
        #pygame.mixer.music.load('sounds/bgm.mp3')
        #pygame.mixer.music.set_volume(0.1)
        #pygame.mixer.music.play(-1)

        # Sounds
        # explosion_sound = pygame.mixer.Sound('sounds/explosion.wav')
        self.gamepad.fill(WHITE)
        background = pygame.image.load('images/background.jpg')
        clock = pygame.time.Clock()

        agent1 = playAgent()
        self.runGame()

class playAgent:
    def __init__(self):
        self.nowObject = self.CreateObj()

    # blcok을 새로 만듦, [image]
    def CreateObj(self):
        idx = random.choice(blockPhotos.order)
        Obj = []
        # 각 block은 Grid의 인덱스를 저장, 중심은 Obj[1]
        if idx == 1:
            Obj = [[0, 0], [1, 0], [2, 0], [3, 0]]  # I
        elif idx == 2:
            Obj = [[0, 0], [1, 0], [2, 0], [2, 1]]  # J
        elif idx == 3:
            Obj = [[0, 0], [1, 0], [2, 0], [0, 1]]  # L
        elif idx == 4:
            Obj = [[0, 0], [1, 0], [0, 1], [1, 1]]  # O
        elif idx == 5:
            Obj = [[2, 0], [1, 1], [1, 0], [0, 1]]  # S
        elif idx == 6:
            Obj = [[1, 0], [1, 1], [0, 1], [2, 1]]  # T
        elif idx == 7:
            Obj = [[0, 0], [1, 1], [1, 0], [2, 1]]  # Z

        for pos in Obj:
            pos[0] += 3

        return Obj

class blockPhotos:

    block = pygame.image.load('images/block.png')
    block_finished = pygame.image.load('images/block_finished.png')

    # 1:I, 2:J, 3:L, 4:O, 5:S, 6:T, 7:Z
    order = [1,2,3,4,5,6,7]

def initPad(render):
    if render:
        gamepad = pygame.display.set_mode((pad_width, pad_height))
        pygame.display.set_caption('Tetris_Test')
    else:
        gamepad = pygame.Surface((pad_width, pad_height)).copy()
    return gamepad

if __name__ == '__main__' :
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (700, 150)
    env = TetrisEnv()
    play_mode = True
    render = False
    if play_mode:
        render = True
    env.initGame(render,play_mode)
