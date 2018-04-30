import pygame
from pygame.locals import *
import sys

size = width, height = (600, 600)

step = width/8
screen = None


candy = pygame.image.load("candy.png")

rock = []
for i in range(1, 6):
    rock.append(pygame.image.load("nanachi_"+str(i)+".png"))

snakeHead = pygame.image.load("miku_head.png")

snakeBody = []
for i in range(1, 9):
    snakeBody.append(pygame.image.load("miku_body"+str(i)+".png"))

def init():
    global screen
    pygame.init() #pygame init
    screen = pygame.display.set_mode(size) #pygame size setting
    #screen.fill( (255, 255, 255) )

def draw( mapArray, score, isOver):
    global screen
    screen.fill( (255,255,255) )
    rock_index = 0
    for y in range(len(mapArray)):
        for x in range(len(mapArray[y])):
            if mapArray[y][x] == 1:
                index = (score%8)
                screen.blit(snakeBody[index], (x*step, y*step))
            elif mapArray[y][x] == 2:
                screen.blit(candy, (x*step, y*step))
            elif mapArray[y][x] == -1:
                screen.blit(rock[rock_index], (x*step, y*step))
                rock_index += 1
            elif mapArray[y][x] == 3:
                screen.blit(snakeHead, (x*step, y*step))

    drawScore(score)
    if isOver:
        gameOver(screen)
    pygame.display.update()
    pygame.time.delay(300)

def drawScore(score):
    global width, screen
    scoreFont = pygame.font.SysFont("comicsansms", 60)
    scoreSurf = scoreFont.render('Score: %s' % (score), True, (0,0,0))
    scoreRect = scoreSurf.get_rect()
    scoreRect.topleft = (width/2, 10)
    screen.blit(scoreSurf, scoreRect)

def gameOver(screen):
    global width, height
    gameOverFont = pygame.font.SysFont("comicsansms", 100)
    gameOverSurf = gameOverFont.render('Game Over!', True, (255,0,0))
    gameOverRect = gameOverSurf.get_rect()
    gameOverRect.center = ((width/2),(height/3))
    screen.blit(gameOverSurf, gameOverRect)
