import pygame
from PIL import Image
import random

class blockPhotos:

    block = pygame.image.load('images/block.png')
    block_finished = pygame.image.load('images/block_finished.png')

    # 1:I, 2:J, 3:L, 4:O, 5:S, 6:T, 7:Z
    order = [1,2,3,4,5,6,7]

class deepAgent:
    def __init__(self):
        self.nowObject = self.CreateObj()

    # blcok을 새로 만듦, [image]
    def CreateObj(self):
        idx = random.choice(blockPhotos.order)
        Obj = []
        # 각 block은 BackGrid의 인덱스를 저장, 중심은 Obj[1]
        if idx == 1:
            Obj = [[0,0],[1,0],[2,0],[3,0]] # I
        elif idx == 2:
            Obj = [[0,0],[1,0],[2,0],[2,1]] # J
        elif idx == 3:
            Obj = [[0,0],[1,0],[2,0],[0,1]] # L
        elif idx == 4:
            Obj = [[0,0],[1,0],[0,1],[1,1]] # O
        elif idx == 5:
            Obj = [[2,0],[1,1],[1,0],[0,1]] # S
        elif idx == 6:
            Obj = [[1,0],[1,1],[0,1],[2,1]] # T
        elif idx == 7:
            Obj = [[0,0],[1,1],[1,0],[2,1]] # Z

        for pos in Obj:
            pos[0] += 3

        return Obj