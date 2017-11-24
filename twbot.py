from operator import itemgetter, attrgetter
import pygame
from pygame.locals import *
import random
import math
import pickle
import time
import heapq
import numpy as np
import random

import sys
import pylab
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

maxEnergy = 150
numGames = 300
winReward = 150
livingReward = -1
energyRate = 13
mlModuloConst = 13
AIstopTime = 13
FPS=0
GREEN = (47,171,51)
RED = (200,0,0)
GRAY = (55,55,55)
WHITE = (255,255,255)
WARMYELLOW = (255,255,85)

class ml2k17(object):
    def __init__(self, numCells):

        self.temp = 0
        self.load_model = False
        self.shouldTrain = True
        self.numCells = numCells
        self.stateSize = numCells*(numCells+1)
        self.actionSize = (2*numCells*(numCells-1))+1
        self.discountFactor = 0.999
        self.learningRate = 0.01
        self.epsilon = 0.75
        self.epsilonDecay = 0.005
        self.epsilonMin = 0.01
        self.batchSize = 10
        self.trainStart = 10
        self.winmemory = []
        self.losememory = []
        self.tempmemory = deque(maxlen=1000)
        self.winGameNum = 20
        self.loseGameNum = 20
        self.model = self.buildModel()
        if self.load_model:
            self.model.load_weights("model.h5")

    def buildModel(self):
        model=Sequential()
        model.add(Dense(int(self.stateSize*1.3), input_dim=self.stateSize,
            kernel_initializer='zeros', bias_initializer='random_uniform', activation='relu'))
        model.add(Dense(int(self.stateSize*1.6), kernel_initializer='zeros',
            bias_initializer='random_uniform',  activation='relu'))
        model.add(Dense(int(self.stateSize*1.3), kernel_initializer='zeros',
            bias_initializer='random_uniform',  activation='relu'))
        model.add(Dense(self.actionSize, activation='linear'))
        #model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        return model

    # get action from model using epsilon-greedy policy
    def getAction(self, state):
        # if np.random.rand() <= self.epsilon:
        #     rand = random.randrange(self.actionSize)
        #     return rand
            # n = self.numCells
            # row = int((rand-1)/(n-1))
            # if(state[0][row+n*n] == 1): return rand
            # else: return self.getAction(state)
        # else:
        qValue = self.model.predict(state)
        # print(qValue[0])
        return np.argmax(qValue[0])

    # save sample <s,a,r,s'> to the replay memory
    def appendSample(self, state, action, reward, nextState, done, score):
        self.tempmemory.append((state, action, reward, nextState, done))
        if done:
            temp = deque(maxlen=1000)
            for i in self.tempmemory: temp.append(i)
            try:
                if (reward == winReward): heapq.heappush(self.winmemory, (-score, temp))
                else: heapq.heappush(self.losememory, (-score, temp))
            except Exception as e:
                pass
            if self.epsilon > self.epsilonMin:
                self.epsilon -= self.epsilonDecay
            self.tempmemory.clear()

    # pick samples randomly from replay memory (with batch_size)
    def trainModel(self):
        # winBatch = min(self.winGameNum, len(self.winmemory))
        # loseBatch = min(self.loseGameNum, len(self.losememory))
        # winmini = random.sample(self.winmemory, winBatch)
        # losemini = random.sample(self.losememory, loseBatch)
        winmini = heapq.nsmallest(self.winGameNum, self.winmemory, key=lambda x:x[0])
        losemini = heapq.nsmallest(self.loseGameNum, self.losememory, key=lambda x:x[0])
        for score, i in winmini:
            # print(score)
            miniBatch = list(reversed(i))
            for state, action, reward, next_state, done in miniBatch:
                target = reward
                if not done:
                    target = reward + self.discountFactor*np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=5, verbose=0)
        for score, i in losemini:
            # print  (score)
            miniBatch = list(reversed(i))
            for state, action, reward, next_state, done in miniBatch:
                target = reward
                if not done:
                    target = reward + self.discountFactor*np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=5, verbose=0)

class CellWar(object):
    def __init__(self):
        self.numGames = numGames
        self.actionForNow=0
        self.score=0
        self.cellStart=-1
        self.cellEnd=-1
        self.numCells=3
        self.maxLinks=2
        self.adjMat = np.zeros((self.numCells,self.numCells))
        self.mlCurrState = np.zeros(self.numCells*(self.numCells+1))
        self.mlPrevState = np.zeros(self.numCells*(self.numCells+1))
        self.mlNumActions = 0
        self.mlModuloConst = mlModuloConst
        self.fps=FPS

    def reset(self):
        n=self.numCells
        self.score=0;
        self.adjMat=np.zeros((n,n));
        self.mlNumActions=n*n-n+1;
        self.mlCurrState=np.zeros(n*(n+1));
        for i in range(n):
            if(self.cellList[i].color==RED):
                self.mlCurrState[i+n*n] = -self.cellList[i].value
            else: self.mlCurrState[i+n*n] = self.cellList[i].value
        np.copyto(self.mlPrevState, self.mlCurrState)

    def mousePressed(self):
        self.recordPos = None
        x = self.cellStart.x
        y = self.cellStart.y

        self.lineDrawn = [(x,y),(x,y),False]
        for cell in self.cellList:
            if cell.color == GREEN:
                if dist(x,y,cell.x,cell.y,cell.radius):
                    self.lineDrawn = [(cell.x,cell.y),(cell.x,cell.y),True]
                    self.dealCell = cell
                    break

        self.redrawAll()


    def isGameOver(self):
        for cell in self.cellList:
            if cell.name == "ATT" and cell.color == GREEN:
                return False
        return True # Game is over

    def isWin(self):
        for cell in self.cellList:
            if cell.name == "ATT" and cell.color != GREEN and cell.color != GRAY:
                # one enemy survives
                return False
        return True # Win

    def doTimeAdjust(self):
        if self.animateCount % energyRate == 0:
            self.shinex = self.shiney = None
            self.increaseValue(GREEN)
            self.increaseValue(RED)
        if self.animateCount % AIstopTime == 0:
            self.AIControl()

    def doTimeThing(self):
        self.animateCount += 1
        # print(self.animateCount)
        self.redrawAll()
        self.doTimeAdjust()            
        # if self.AIEMB != None:
        #     self.tryMoveAIEMB()
        self.fps = FPS
        self.clock.tick(self.fps)
        # self.testCollide()

    def mlAct(self, cutChain):
        if (cutChain and (self.adjMat[self.cellStart.index][self.cellEnd.index]==0) and 
            (int(np.sum(self.adjMat[self.cellStart.index])) < self.maxLinks)):
            # Creating tentacles if not present and current number of tentacles are less than max
            self.mousePressed()
            return
        if not cutChain and (self.adjMat[self.cellStart.index][self.cellEnd.index]==1):
            # Destroying current tentacle
            self.tryConsiderCut()

    def mlChooseEvent(self,rand):
        n=self.numCells
        cutChain = False
        if rand > n*(n-1):
            rand -= n*(n-1)
            cutChain = True
        row=int((rand-1)/(n-1));
        col=(rand-1)%(n-1);
        if(col>=row):
            col+=1

        return (self.cellList[row], self.cellList[col], cutChain)

    def mlGetState(self):
        n = self.numCells
        for i in range(n):
            if(self.cellList[i].color==RED):
                self.mlCurrState[i+n*n] = -self.cellList[i].value
            else: self.mlCurrState[i+n*n] = self.cellList[i].value

        for i in range(n):
            for j in range(n):
                self.mlCurrState[n*i+j] = self.adjMat[i][j]

    def mlGetReward(self):
        return 0

    def timerFired(self):
        if not self.isGameOver() and not self.isWin(): # Game running
            self.doTimeThing()
            if((self.animateCount)%(self.mlModuloConst)==0):

                self.mlGetState()
                reward = self.mlGetReward() + livingReward
                self.score += reward
                # print("REWARD=",reward,"    SCORE=",self.score)

                mlAgent.appendSample(np.reshape(self.mlPrevState,[1, mlAgent.stateSize]), self.actionForNow, 
                    reward, np.reshape(self.mlCurrState,[1, mlAgent.stateSize]), False, None)
                # mlAgent.trainModel()
                self.actionForNow=mlAgent.getAction(np.reshape(self.mlCurrState,[1,mlAgent.stateSize]))
                # print("ACTION=", self.actionForNow)

                if(self.actionForNow!=0):
                    (self.cellStart, self.cellEnd, cutChain) = self.mlChooseEvent(self.actionForNow)
                    if(self.cellStart.color==GREEN):
                        self.mlAct(cutChain)

                self.mlPrevState = self.mlCurrState
            #manually manage the event queue

            if len(self.lineDrawn) == 3:
                self.initial = self.lineDrawn[0]
                self.recordPos = self.lineDrawn[1]
                self.lineDrawn = []

            if self.recordPos != None:
                # first judges if EMB or ATT needs to move.
                do = self.tryMoveCell()
                if do == False: # means potentially a cut
                    #self.tryConsiderCut()
                    self.recordPos = None

        elif not self.isGameOver() and self.isWin(): # Win!
                self.score+=winReward
                self.numGames -= 1
                print("EPSILON={0:.5f}   GAME NUMBER={1:3d}".format(mlAgent.epsilon,self.numGames),end=' ');
                print("  WIN    score=",self.score, end="  \n")
                mlAgent.appendSample(np.reshape(self.mlPrevState,[1, mlAgent.stateSize]), self.actionForNow,
                    winReward, np.reshape(self.mlCurrState,[1, mlAgent.stateSize]), True, self.score);
                if mlAgent.shouldTrain: mlAgent.trainModel()
                mlAgent.model.save_weights("model.h5")
                self.init(4)
                self.reset()

        elif self.isGameOver() and not self.isWin(): # Lose!
                self.score-=winReward
                self.numGames -= 1
                print("EPSILON={0:.5f}   GAME NUMBER={1:3d}".format(mlAgent.epsilon,self.numGames),end=' ');
                print("  LOSS   score=",self.score, end="  \n")
                mlAgent.appendSample(np.reshape(self.mlPrevState,[1, mlAgent.stateSize]), self.actionForNow,
                    -winReward, np.reshape(self.mlCurrState,[1, mlAgent.stateSize]), True, self.score);
                if mlAgent.shouldTrain: mlAgent.trainModel()
                mlAgent.model.save_weights("model.h5")
                self.init(4)
                self.reset()

    ######################### END OF TIMERFIRED ##########################            
    ######################### END OF TIMERFIRED ##########################
    ######################### END OF TIMERFIRED ##########################

    def tryConsiderCut(self):
        for chain in self.chains:
            if chain.color == GREEN and not chain.shouldGrow and\
               not chain.shouldCollapse:
                # only cut friendly chain...

                intersect = 1
                if intersect:
                    # that is, if there exist any intersection
                    x0= 0.5*(self.cellStart.x + self.cellEnd.x);
                    y0= 0.5*(self.cellStart.y + self.cellEnd.y);
                    breakInd = self.findBreakPoint(chain,x0,y0)
                    if(breakInd == None): continue
                    chain.shouldBreak = True
                    self.adjMat[self.cellStart.index][self.cellEnd.index]=0;
                    try:
                        chain.chainList[breakInd] = (0,0)
                        chain.breakInd = breakInd
                    except:
                        pass
                    if self.mode == "Tutorial":
                        self.tutorialStep = 5

    def makeGray(self):
        for x in range(700):
            for y in range(700):
                (R,G,B,A) = self.background.get_at((x,y))
                gray = (R+G+B)/3
                self.background.set_at((x,y),(gray,gray,gray,255))
        self.grayify = True

    def findBreakPoint(self,chain,x0,y0):
        for i in range(len(chain.chainList)):
            (dotx,doty) = chain.chainList[i]
            if dist(x0,y0,dotx,doty,3.5):
                # there must be such a point
                return i # notice: an index is returned!
    
    #########################################################################
    #########################################################################
    # Artificial Intelligence Part
    #########################################################################
    #########################################################################

    #################################################
    # Acting as if we are controlling enemy cell
    #################################################

    def testCellInList(self,cell,L):
        # test if a cell is in a list (of chains) as a target
        for i in range(len(L)):
            chain = L[i]
            if dist(chain.endx,chain.endy,cell.x,cell.y,5):
                return True
        return False
    
    def AIControl(self):
        level = self.levelChosen
        for cell in self.cellList:
            do = random.randint(1,2) # not move in every round, but just some
            if cell.color != GRAY and cell.color != GREEN and do == 1:
                # filter out enemy cells, excluding neutral ones.
                modifiedCellList = []
                for other in self.cellList:
                    try:
                        if not self.testCellInList(other,self.dic[cell]) and\
                           other.x != cell.x:
                            # current target
                            modifiedCellList.append(other)
                    except: # meaning self.dic[cell] = -1! NO CHAIN AT ALL!
                        modifiedCellList.append(other)
                if cell.name == "ATT":
                    # feed with the latest cellList and info
                    cell.update(modifiedCellList,self.animateCount)
                    if cell.state == "Attack":
                        self.AICellAttack(cell)
                    elif cell.state == "Defense":
                        self.AICellCollapse(cell)
                        pass   ################# for now #################
                    elif cell.state == "Attack and Assist" and level > 3:
                        self.AIAssist(cell)
                if cell.name == "EMB":
                    judge = random.randint(1,10)
                    if 1 <= judge <= 3:
                        cell.update(modifiedCellList,self.animateCount)
                        self.AIEMBcontrol(cell)
                    else:
                        pass

    def AIEMBcontrol(self,cell):
        # for EMB only
        targetList = cell.allOtherList
        if cell.state == "Attack":
            target = targetList.pop(0)[2]
            self.AIEMB = (cell,target)
            self.tryMoveAIEMB()

    def tryMoveAIEMB(self):
        cell = self.AIEMB[0]
        target = self.AIEMB[1]
        subtractMoves = 14
        if not dist(cell.x,cell.y,target.x,target.y,0.5):
                    # not yet at the destination
            cell.moveJudge = True
            cell.move(target.x,target.y,self.fps)
            if self.animateCount % subtractMoves == 0:
                cell.value -= 1
        else:
            self.AIEMB = None

    def AIShrinkTent(self,cell):
        for i in range(2):
            chain = self.dic[cell][i]
            chainEnd = self.findTarget(chain.endx,chain.endy)
            if chainEnd.color == GREEN or chainEnd.color.color ==GRAY:
                self.adjMat[chain.startCell.index][chain.endCell.index]=0;
                chain.shouldCollapse = True

                break # only break one of the tentacles

    def AIAssist(self,cell):
        alliesList = cell.alliesList
        try:
            emergency = cell.findEmergencyCell()
            emergency.withHelp = False
            # by default. It tells us if there is any existing help
            for allyInfo in alliesList:
                ally = allyInfo[2]
                if self.dic[ally] != -1:
                    for i in range(len(self.dic[ally])):
                        chain = self.dic[ally][i]
                        if dist(chain.endx,chain.endy,emergency.x,emergency.y,5):
                            emergency.withHelp = True
            if not emergency.withHelp:
                if len(self.dic[cell]) == 2:
                    self.AIShrinkTent(cell)
                if self.dic[cell] == -1 or len(self.dic[cell]) < 2:
                    target = emergency
                    # establish a chain between cell and the emergency cell
                    self.noRepeatChain(target,cell)
                    if(self.adjMat[cell.index][target.index]==1):
                        return;
                    chain = Chain(cell.x,cell.y,target.x,target.y,cell,target,cell.color)
                    self.chains.append(chain)

                    self.adjMat[cell.index][target.index]=1;

                    if self.dic[cell] == -1:
                        # newly created key, in essence
                        self.dic[cell] = [chain]
                    else:
                        # "experienced" enemy cell
                        self.dic[cell].append(chain)
            else:
                self.AICellAttack(cell)
        except:
            pass
                                                
                        

    def AICellAttack(self,enemyCell):
        alliesList = enemyCell.alliesList
        targetList = enemyCell.allOtherList
        if self.dic[enemyCell] == -1 or len(self.dic[enemyCell]) < 2:
            # maxmimum two tentacles, by far
            if(len(targetList)==0):
                return;
            target = targetList.pop(0)[2]
            # recall that what get popped out is (cell.value,cell.color,cell)
            if(self.adjMat[enemyCell.index][target.index]==1):
                return;

            chain = Chain(enemyCell.x,enemyCell.y,target.x,target.y,enemyCell,target,enemyCell.color)
            self.chains.append(chain) ##### important #####
            self.adjMat[enemyCell.index][target.index]=1;
            if self.dic[enemyCell] == -1:
                # newly created key, in essence
                self.dic[enemyCell] = [chain]
            else:
                # "experienced" enemy cell
                self.dic[enemyCell].append(chain)

    def AICellCollapse(self,cell):
        minimumValue = 6
        if self.dic[cell] != -1:
            for chain in self.dic[cell]:
                if cell.value > minimumValue:
                    break
                if not chain.shouldGrow:
                    self.adjMat[chain.startCell.index][chain.endCell.index]=0;
                    chain.shouldCollapse = True
            
            
            
                               
   
    ########################################################################
    # tryMoveCell is for moving GREEN. AIControl is for moving enemies
    ########################################################################
    def tryMoveCell(self):
        try:
            if self.dealCell.name == "EMB":             
                self.tryMoveEMB()
                return True
        except:
            return False
        try:
            if self.dealCell.name == "ATT":
                self.tryMoveATT()
                return True
                # set to None after finding once
        except:
            return False

    def tryMoveATT(self):

        for cell in self.cellList:

            if cell.name == "ATT" and \
                (cell.x,cell.y) != (self.dealCell.x,self.dealCell.y):
                if dist(self.recordPos[0],self.recordPos[1],cell.x,\
                        cell.y,cell.radius):

                    if((self.adjMat[self.dealCell.index][cell.index])==1):
                        break; 

                    chain = Chain(self.dealCell.x,self.dealCell.y,cell.x,cell.y,self.dealCell,cell,self.dealCell.color)
                    self.testAddAssist(chain)
                    self.chains.append(chain)

                    self.adjMat[self.dealCell.index][cell.index]=1;

                    if self.dic[self.dealCell] == -1:
                        # newly created key
                        self.dic[self.dealCell] = [chain]
                    else:
                        self.dic[self.dealCell].append(chain)
                        # note the direction is from self.dealCell to cell
                    break
                        # found the intended target
        self.recordPos = None
        self.dealCell = None

    def tryMoveEMB(self):
        cycle = 14
        if not dist(self.dealCell.x,self.dealCell.y,\
                   self.recordPos[0],self.recordPos[1],0.5):
        # not yet at the destination
            self.dealCell.moveJudge = True
            self.dealCell.move(self.recordPos[0],\
                                self.recordPos[1],self.fps)
            if self.animateCount % cycle == 0:
                self.dealCell.value -= 1
        else:
            tutMoveEMBStep = 5
            if self.mode == "Tutorial":
                if self.tutorialStep == tutMoveEMBStep:
                    self.tutorialStep += 1
            self.recordPos = None
            self.dealCell = None

    def testAddAssist(self,chain): # test if self.totalAssist should add 1
        chainEnd = self.findTarget(chain.endx,chain.endy)
        self.noRepeatChain(chainEnd,self.dealCell)
        if chainEnd.value <= 10 and chainEnd.color == GREEN:
            if self.dic[chainEnd] != -1:
                for chain in self.dic[chainEnd]:
                    if chain.shouldGrow:
                        return

    def noRepeatChain(self,chainEnd,startCell):
        if self.dic[chainEnd] != -1 and chainEnd.color == startCell.color:
            # consider repetitive only if the same color
            for back in self.dic[chainEnd]:
                if self.findTarget(back.endx,back.endy) == startCell:
                    self.adjMat[chainEnd.index][startCell.index]=0;
                    back.shouldCollapse = True
                    break # make sure no transport in 2 directions
                
    def findTarget(self,x,y):
        # find the closest ATT cell or EMB cell
        self.target = None
        curdist = 10000
        for cell in self.cellList:
            dist = ((cell.x-x)**2+(cell.y-y)**2)**0.5
            if dist < curdist:
                curdist = dist
                self.target = cell
        return self.target

    def playShine(self,x,y,do=False):
        if do:
            image = pygame.image.load("resoures/BOOM.png").convert_alpha()
            self.screen.blit(image,(x-90,y-64)) # Adjust the center of BOOM
        
    def isCollide(self,s1,s2):
        return dist(s1.rect.x,s1.rect.y,s2.rect.x,s2.rect.y,30)
    
    def testCollide(self):
        # test if any EMB is colliding with anything
        for cell in self.cellList:
            if cell.name == "EMB":
                for cell2 in self.cellList:
                    if cell != cell2 and dist(cell.x,cell.y,cell2.x,cell2.y,43): #collide!
                        subtract,turnColor = cell.value,cell.color
                        self.shinex,self.shiney = cell.x,cell.y
                        self.playShine(cell.x,cell.y,True)
                        self.cellList.remove(cell)
                        target = cell2
                        self.adjustValue(target,subtract,turnColor)
                        cell.value = abs(cell.value)

    def adjustValue(self,cell,minus,color):
        # dropping or strengthening cell!
        delta = -1 if cell.color == color else 1
        while minus != 0 and cell.value < self.maximum:
            cell.value -= delta
            minus -= 1
            self.redrawAll()
            if cell.value < 0:
                if cell.color == GRAY:
                    cell.value = 10 # bonus value for neutral occupy!
                else:
                    if cell.name == "ATT":
                        self.forceMakeCollapse(cell)
                    cell.value = abs(cell.value)
                cell.color = color
                delta = -1

    

    def increaseValue(self,color):
        for cell in self.cellList:
            if cell.color == color and cell.value < self.maximum: #9 is a max
                if cell.name == "ATT":
                    cell.value += 1
                else:
                    cell.increaseCount += 1
                    if cell.increaseCount % 2 == 0:
                        cell.value += 1
                        cell.increaseCount = 0

    def findIntersection(self, xxx_todo_changeme1, xxx_todo_changeme2, xxx_todo_changeme3, xxx_todo_changeme4):
        # reflect the axis, and using mathematical method to find the slope
        (x1,y1) = xxx_todo_changeme1
        (x2,y2) = xxx_todo_changeme2
        (stx,sty) = xxx_todo_changeme3
        (endx,endy) = xxx_todo_changeme4
        if x2 != x1:
            l = curslope = float(y2-y1)/(x2-x1)
        elif min(stx,endx) <= x2 <= max(stx,endx) and min(y1,y2) <= \
             endy-float(endy-sty)/(endx-stx)*(x2-endx) <= max(y1,y2):
            k = float(endy-sty)/(endx-stx)
            return (x1,k*x1-k*endx+endy)
        else:
            return False
        print("geese")
        if endx != stx:
            k = tarslope = float(endy-sty)/(endx-stx)
        elif min(x1,x2) <= stx <= max(x1,x2) and \
             min(sty,endy) <= (y1+y2)/2. <= max(sty,endy):
            l=float(y2-y1)/(x2-x1)
            return (stx,l*stx-l*x1+y1)
        else:
            return False
        if k == l: return False # parallel
        x0 = (k*endx-l*x1+y1-endy)/(k-l)
        y0 = k*x0-k*endx+endy
        if min(stx,endx) <= x0 <= max(stx,endx) and \
           min(sty,endy) <= y0 <= max(sty,endy) and \
           min(x1,x2) <= x0 <= max(x1,x2) and \
           min(y1,y2) <= y0 <= max(y1,y2):
           # in correct range
            return (x0,y0)
        else:
            return False
        
    
    def drawLine(self):
        lineDrawn = self.lineDrawn
        if len(lineDrawn) == 3:
            pygame.draw.line(self.screen,(255,255,0),lineDrawn[0],lineDrawn[1],3)
    
    def traceLine(self):
        # draw the yellow line by tracing the position of the mouse
        if len(self.lineDrawn) == 3:
            self.lineDrawn.pop(1)
            self.lineDrawn.insert(1,(self.cellEnd.x,self.cellEnd.y))
            if self.lineDrawn[-1]:
                x=self.cellEnd.x;
                y=self.cellEnd.y;
                for cell in self.cellList:
                    if dist(x,y,cell.x,cell.y,cell.radius):
                        self.potential = cell
                        Lock(cell.x,cell.y).drawLock(self.screen)
                        

    def drawLock(self):
        lineDrawn = self.lineDrawn
        try:
            if self.lineDrawn[-1]: # True means indeed locked
                Lock(lineDrawn[0][0],lineDrawn[0][1]).drawLock(self.screen)
        except:
            pass

    def drawChain(self):
        for chain in self.chains:
            #pygame.draw.line(self.screen,GREEN,(chain.startx,chain.starty),(chain.endx,chain.endy),2);
            if chain.shouldGrow and not (chain.shouldCollapse or chain.shouldBreak):
                chain.growNum += 1
                if chain.growNum % 2 == 0:
                    chain.grow()
            chain.drawChain(self.screen)

    
    def chainUpdate(self,cell):
        # determine the frequency of transporting based on source cell value
        if cell.value == 1:
            for chain in self.dic[cell]:
                chain.shiningInd = []
                chain.freq = 40
        elif 1 < cell.value < 15:
            for chain in self.dic[cell]:
                if len(chain.shiningInd) != 1:
                    chain.shiningInd = [-1]
                chain.freq = 25
        elif 15 <= cell.value < 35:
            for chain in self.dic[cell]:
                if len(chain.shiningInd) == 1:
                    chain.determineInd()
                chain.freq = 15
        elif 35 <= cell.value < 80:
            for chain in self.dic[cell]:
                if len(chain.shiningInd) == chain.IndNum:
                    chain.shiningInd.append(min(chain.shiningInd)-13)
                chain.freq = 10
                
    def traceTransfer(self):
        for cell in self.cellList:
            if cell.name == "ATT" and type(self.dic[cell]) != int:
                # meaning it is an object (i.e. a chain!)
                self.chainUpdate(cell)
                for i in range(len(self.dic[cell])):
                    # recall that self.dic[cell] returns the chains that "cell"
                    # currently has.
                    try:
                        currentChain = self.dic[cell][i]
                        chainEnd = self.findTarget(currentChain.endx,currentChain.endy)
                        if currentChain.shouldGrow:
                            self.doChainGrow(currentChain,cell)
                        else:
                            if currentChain.shouldBreak:
                                do = self.inBreakProcess(cell,chainEnd,currentChain)
                                if do == "done":
                                    break
                            else:
                                self.doCompleteChain(currentChain,cell,chainEnd)
                        if currentChain.shouldCollapse and \
                            len(currentChain.chainList) > 0:
                            self.doingCollapse(currentChain,cell)
                        elif len(currentChain.chainList) == 0:
                            self.dic[cell].remove(currentChain)
                            self.chains.remove(currentChain)
                            break # avoid changing list inside a loop
                    except:
                        pass

    def doingCollapse(self,currentChain,cell):
        currentChain.chainList.pop() # pop from the last one
        currentChain.shiningInd = []
        if len(currentChain.chainList)%2 == 0:
            if currentChain.color == cell.color and cell.value < self.maximum:
                cell.value += 1
            # same rule as when subtracting
        

    def doChainGrow(self,currentChain,cell):
        if currentChain.subtractCellValue:
            cell.value -= 1
            currentChain.subtractCellValue = False
        if cell.value <= 0: # ehhh... not enough length!
            currentChain.shouldCollapse = True
            self.adjMat[currentChain.startCell.index][currentChain.endCell.index]=0;

            # so currentChain.grow() is no longer called

    def doCompleteChain(self,currentChain,cell,chainEnd):
        for i in range(len(currentChain.shiningInd)):
            currentChain.shiningInd[i] += 1                        
            if currentChain.shiningInd[i] >= currentChain.dotNum:
                # one signal ends
                delta = -1 if chainEnd.color == cell.color else 1
                if chainEnd.value == self.maximum and delta==1:
                    chainEnd.value -= delta;
                elif chainEnd.value < self.maximum:
                    chainEnd.value -= delta
                if chainEnd.value < 0:
                    if chainEnd.color == GRAY:
                        chainEnd.value = 10
                    else:
                        self.forceMakeCollapse(chainEnd)
                    chainEnd.color = cell.color
                    chainEnd.value = abs(chainEnd.value)
                currentChain.shiningInd[i] = 5-currentChain.freq
                #currentChain.shiningInd[i] = 5
                    # every signal destroys one target life value
        

    def inBreakProcess(self,cell,chainEnd,currentChain):
        grayBonus = 10
        validLow,validHigh = self.collapseBothWays(currentChain)
        if max(validLow,validHigh) % 2 == 1: # for every two round, value drops
            if validHigh != 0 and chainEnd.value < self.maximum:
                delta = -1 if chainEnd.color != cell.color else +1
                chainEnd.value += delta
                if chainEnd.value < 1:
                    if chainEnd.color == GRAY:
                        chainEnd.value = grayBonus # bonus for gray!
                    else:
                        self.forceMakeCollapse(chainEnd)
                        chainEnd.value = abs(chainEnd.value)
                    chainEnd.color = cell.color                    
            if validLow != 0 and cell.value < self.maximum:
                cell.value += 1
            return "continue"
        elif (validLow + validHigh) == 0:# remove reference from current cell
            self.dic[cell].remove(currentChain)
            self.chains.remove(currentChain)
            return "done" 
    
    def forceMakeCollapse(self,cell):
        if self.dic[cell] != -1:
            for chain in self.dic[cell]:
                self.adjMat[chain.startCell.index][chain.endCell.index]=0;
                chain.shouldCollapse = True

    def collapseBothWays(self,chain):
        # i is the breaking point index
        tempList = list(chain.chainList)
        if chain.chainList[0] != (0,0):
            for i in range(chain.breakInd+1):
                if chain.chainList[i] == (0,0):
                    low = i-1
                    break
            tempList[low] = (0,0) # turns an examined dot to (0,0)
        else:
            low = 0
        if chain.chainList[-1] != (0,0):
            for i in range(len(chain.chainList)-1,chain.breakInd-1,-1):
                if chain.chainList[i] == (0,0):
                    high = i+1 # high index shift by one
                    break
            tempList[high] = (0,0)
        else:
            high = len(chain.chainList)-1
        chain.chainList = tempList
        return low,len(chain.chainList)-high-1
    
    def collapseChain(self,currentChain):
        currentChain.chainList.pop()
        currentChain.shiningInd = []
        if len(currentChain.chainList)%2 == 0:
            cell.value += 1


    def redrawShapes(self):
        self.traceLine()
        self.drawLock()
        self.traceTransfer()
        self.drawChain()
        self.drawLine()

        
    def redrawAll(self):
        cellImg = self.imageList[0]
        self.screen.blit(self.bgimage,(0,0))
        self.redrawShapes()
        maxFont = 15
        font = pygame.font.SysFont("Calibri",maxFont,False)
        textObj = font.render("Maximum life value: %s"%self.maximum,True,GREEN)
        self.screen.blit(textObj,(10,10))
        for cell in self.cellList:
            if cell.name == "ATT":
                cycle,backCycle,rd,adjustx,adjusty = 35,31,5,48,53
                if self.animateCount % len(self.imageList) == 0:
                    self.order = not self.order
                cellImg = self.imageList[int((self.animateCount%cycle)/rd)] \
                    if self.order else \
                    self.imageList[int(-(self.animateCount%backCycle)/rd - 1)]
                self.screen.blit(cellImg,(cell.x-adjustx,cell.y-adjusty))
            cell.drawCell(self.screen)
        try:self.playShine(self.shinex,self.shiney,True)
        except: pass
        pygame.display.flip()

    def loadImageList(self):
        self.imageList = [pygame.image.load('resources/GreenCell4.png'),\
                     pygame.image.load('resources/GreenCell9.png'),\
                     pygame.image.load('resources/GreenCell6.png'),\
                     pygame.image.load('resources/GreenCell8.png'),\
                     pygame.image.load('resources/GreenCell5.png'),\
                     pygame.image.load('resources/GreenCell7.png'),\
                     pygame.image.load('resources/GreenCell3.png'),]

    def initImgAndMusic(self):
        self.winImgy = -700
        self.loadImageList()       
        # set self.animateCount to zero again
        self.animateCount = 0

    def init(self,level): # level as a number
        self.mode = "Running"
        self.initImgAndMusic()
        self.potential = None # potential target pointing at
        self.lineDrawn,self.grayify = [],False # should we make bg gray?
        self.recordPos = None # a temporary position recorder
        self.order = True # left to right
        self.levelList = [Level_1(),Level_2(),Level_3(),Level_4(),Level_5(),\
                          Level_6(),Level_7()]
        self.levelChosen,self.AIEMB,self.chains = level,None,[]
        self.cellList = self.levelControl(self.levelList[level-1])
        self.maximum = self.levelList[level-1].maximum
        self.dic = dict() # same as above, to record.
        for cell in self.cellList:
            cell.sprite = Target()
            cell.sprite.rect.x = cell.x-cell.radius
            cell.sprite.rect.y = cell.y-cell.radius
            #self.block_list.add(cell.sprite)
            if cell.name == "ATT":
                self.dic[cell] = -1 # by default
        self.redrawAll()

    def levelControl(self,level): # which level?
        return level.cellList
    
    def run(self):
        pygame.init()
        # initialize the screen
        self.screenSize = (700,700)
        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("Tentacle Wars")
        
        # initialize clock
        self.clock = pygame.time.Clock()
        # the cell we are dealing with
        self.dealCell = "ATT"
        self.animateCount = 0
        # self.menuInit()
        self.bgimage = pygame.image.load('resources/StoneAge2.jpg')
        self.init(4)
        while (self.numGames > 0):
            self.timerFired()
        pygame.quit()
                
class Target(pygame.sprite.Sprite):
    """ Fake cells to test collision"""

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("resources/sprite.jpg").convert()

        self.rect = self.image.get_rect()
    
class Cell(object):
    def __init__(self,x,y,value=20,color=GREEN,index=-1):
        self.x,self.y = x,y
        self.index=index
        self.color = color # green by default
        self.radius = 23
        self.outerRadius = self.radius + 8
        self.lastValue = self.value = value
        self.name = "ATT"
        self.state,self.loss = None,0
        self.d,self.avgDelta = dict(),[]
        self.getNeedle = False # being injected?
        self.injectTime = 0 # start counting the time during Injection from 0
        self.fakeNeedle = False # needle left is 0

    def drawCell(self,surface):
        center = (self.x,self.y)
        color = WHITE if not self.getNeedle else GREEN
        pygame.draw.circle(surface,color,center,self.outerRadius,3)
        pygame.draw.circle(surface,self.color,center,self.radius,0)
        ############# possibly a gradient effect ###############
        pygame.draw.circle(surface,(255,255,255),center,self.radius,2)
        self.drawValue(surface)
        if self.value >= 50:
            # more fashion drawing for larger cell
            for i in range(8):
                size = random.randint(2,6)
                self.drawSideCircle(surface,i,size)

    def drawSideCircle(self,surface,ang,radius):
        angle = ang*math.pi/4
        cx,cy = int(round(self.x+self.outerRadius*math.cos(angle))),\
                int(round(self.y+self.outerRadius*math.sin(angle)))
        pygame.draw.circle(surface,self.color,(cx,cy),radius,0)

    def drawValue(self,surface):
        my_font = pygame.font.SysFont("",21,True)
        textObj = my_font.render("%d"%self.value,True,(255,255,255))
        if len(str(self.value)) == 2:
            surface.blit(textObj,(self.x-8,self.y-10.5))
        else:
            surface.blit(textObj,(self.x-5,self.y-10.5))

    def findDistanceInChainUnits(self,targetx,targety):
        # for Artificial Intelligence use
        # chain unit (dot) is of length of 3
        geoDistance = ((targetx-self.x)**2+(targety-self.y)**2)**0.5
        dotNumber = geoDistance/(3*2) # diameter
        valueNeed = dotNumber/2
        return valueNeed


    def findAllies(self,allCellList):
        # in game, allCellList should be self.cellList
        #print("asdsfdf");
        self.alliesList = []
        embList = [] # temporary list for EMB
        allyAvg = 0
        for cell in allCellList:
            if cell.color == self.color: # possibly used for BLUE?
                if cell.name == "ATT" and cell.x != self.x:
                    self.alliesList.append((cell.value,cell.name,cell))
                else:
                    embList.append((cell.value,cell.name,cell))
                allyAvg += cell.value
        # EMB cells first, and then ATT cell, in cell value order
        '''
        print("start");
        print(self.alliesList);
        print(sorted(self.alliesList, key=itemgetter(0)))
        print("end");
        '''
        self.alliesList = sorted(embList, key=itemgetter(0)) + sorted(self.alliesList, key=itemgetter(0))
        if len(self.alliesList) != 0:
            return float(allyAvg)/len(self.alliesList)
        else:
            return 0 # force the cell to be in defense mode

    def findEnemiesWithinDistance(self,allCellList):
        """ return self.grayList and self.allOtherList as a tuple """
        # for AI use. 
        self.enemiesList = []
        grayList = [] # higher priority should be put in front
        enemyAvg = 0
        for cell in allCellList:
            if cell.color != self.color:
                valueNeed = self.findDistanceInChainUnits(cell.x,cell.y)
                if self.value - valueNeed > 10:
                # that is, after reaching out tentacle, value left is 10
                    # gray (neutral) cell should be of priority
                    if cell.color == GRAY:
                        # ATTENTION! No longer cell.name for index 1
                        grayList.append((cell.value,cell.color,cell))
                    elif cell.name == "ATT":
                        # the heuristic here is that estimated distance
                        # to travel plus target's value
                        enemyAvg += cell.value
                        self.enemiesList.append((cell.value+valueNeed,\
                                                 cell.color,cell))
        self.grayList = list(grayList) # just in case, so that no aliasing
        self.allOtherList = list(reversed(sorted(grayList, key=itemgetter(0))))+sorted(self.enemiesList, key=itemgetter(0))
        if len(self.enemiesList) != 0:
            return float(enemyAvg)/len(self.enemiesList)
        else:
            return 100 # force the cell to be in defense mode

    def findEmergencyCell(self):
        alliesList = self.alliesList
        for (allyValue,allyName,ally) in alliesList:
            #print "allyinfo:",ally.x,ally.y
            if ally.state == "Alert" and allyName == "ATT":
                return ally
        return None
                

    def think(self,environment,animateCount):
        # the thinking process refers to the AI
        #####################################################################
        #     Later possibly use value drop in time also to determin. For
        # instance, consider d(C.V.)/dt
        #####################################################################
        enemyAvg = self.findEnemiesWithinDistance(environment)

        allyAvg = self.findAllies(environment)
        count,lowKey,highKey = animateCount,8,16
        delta = self.lastValue - self.value # loss of life in given time. Say 5.
        self.lastValue = self.value
        self.avgDelta.append(delta)
        if len(self.avgDelta) >= 15:
            self.loss = float(sum(self.avgDelta))/15
            self.avgDelta = []
            #print "loss",self.x,self.y,self.loss
        emergency = self.findEmergencyCell()
        #print "emergency:",emergency
        #print allyAvg,enemyAvg
        if animateCount < 100:
            self.state = "Defense"
        elif lowKey < self.value < highKey or 1.4 <= self.loss <=1.5 :
            # lose 6 value points per second
            self.state = "Defense"
        elif self.value <= lowKey or self.loss > 1.5:
            self.state = "Alert"
        elif self.value >= highKey:
            self.considerEmerg(emergency,allyAvg,enemyAvg)
            
                    
    def considerEmerg(self,emergency,allyAvg,enemyAvg):
        lowKey,highKey = 5,15
        if emergency != None:
            if self.value+90 >= enemyAvg:
                self.state = "Attack and Assist"
            elif self.value + allyAvg >= enemyAvg*3./2:
                self.state = "Attack and Assist"
            else:
                self.state = "Defense"
        else:
            if self.value+lowKey >= enemyAvg:
                self.state = "Attack"
            elif self.value + allyAvg >= enemyAvg*3./2:
                self.state = "Attack"
            else:
                self.state = "Defense"

    def update(self,environment,animateCount):
        # update every aspect: camp, current mode, etc.
        # ONLY ENEMY CELL NEEDS TO UPDATE.
        self.think(environment,animateCount)
        #print self.x,self.y,self.state
        # change mode,find friends,find enemies
            
        

    def __hash__(self):
        hashable = (self.x,self.y)
        return hash(hashable)

class Embracer(Cell):
    def __init__(self,x,y,value,color=(47,171,51)):
        super(Embracer,self).__init__(x,y,value,color)
        self.name = "EMB"
        self.increaseCount = 0
        self.moveJudge = False
    
    def drawCell(self,surface):
        #print self.name,self.x,self.y
        center = (x,y) = (self.x,self.y)
        add = int(round(2*self.radius/(3**0.5)))

        pygame.draw.polygon(surface,(255,255,255),((x,y-self.radius-8),\
                            (x+add,y+self.radius-8),\
                            (x-add,y+self.radius-8)))
        pygame.draw.circle(surface,self.color,center,self.radius,0)
        pygame.draw.circle(surface,(255,255,255),center,self.radius,2)
        self.drawValue(surface)

    def setTarget(self,targetx,targety,fps):
        distance = ((targetx-self.x)**2+(targety-self.y)**2)**0.5
        acce = (distance*2/3**2)/fps #acceleration. Complete in 3 sec
        self.speed = int(round((2*acce*distance)**0.5))
        self.speedx = int(round(((targetx-self.x)/distance)*self.speed))
        self.speedy = int(round(((targety-self.y)/distance)*self.speed))
        self.accex = ((targetx-self.x)/distance)*acce
        self.accey = ((targety-self.y)/distance)*acce

    def move(self,targetx,targety,fps):
        self.setTarget(targetx,targety,fps)
        if self.moveJudge:
            self.x += self.speedx
            self.y += self.speedy
            if dist(targetx,targety,self.x,self.y,0.5):
                self.speedx -= int(round(self.accex))
                self.speedy -= int(round(self.accey))
            else:
                self.moveJudge = False #stops
            self.sprite.rect.x = self.x
            self.sprite.rect.y = self.y


    def update(self,environment,animateCount):
        # update every aspect: camp, current mode, etc.
        # ONLY ENEMY CELL NEEDS TO UPDATE.
        self.think(environment,animateCount)
        # change mode,find friends,find enemies
            
class Level_8(object):
    def __init__(self):

        self.c1 = Cell(250,150,12,RED,0)
        self.c2 = Cell(450,150,12,GREEN,1)
        self.cellList = [self.c1,self.c2]
        self.maximum = maxEnergy

class Level_4(object):
    def __init__(self):

        self.c1 = Cell(350,250,60,RED,0)
        self.c2 = Cell(550,250,30,GREEN,1)
        self.c3 = Cell(150,150,40,GREEN,2)
        self.cellList = [self.c1,self.c2,self.c3]
        self.maximum = maxEnergy

class Level_9(object):
    def __init__(self):

        self.c1 = Cell(250,150,12,RED,0)
        self.c2 = Cell(450,150,12,GREEN,1)
        self.c3 = Cell(150,int(round(150+100*math.sqrt(3))),12,RED,2)
        self.c4 = Cell(550,int(round(150+100*math.sqrt(3))),12,RED,3)
        self.c5 = Cell(250,int(round(150+200*math.sqrt(3))),12,RED,4)
        self.c6 = Cell(450,int(round(150+200*math.sqrt(3))),12,RED,5)
        self.c7 = Cell(350,320,60,GREEN,6)
        self.cellList = [self.c1,self.c2,self.c3,self.c4,self.c5,\
                         self.c6,self.c7]
        self.maximum = maxEnergy

class Level_1(object):
    def __init__(self):

        self.c1 = Cell(250,150,60,RED,0)
        self.c2 = Cell(450,150,36,GREEN,1)
        self.c3 = Cell(150,int(round(150+100*math.sqrt(3))),60,RED,2)
        self.c4 = Cell(550,int(round(150+100*math.sqrt(3))),36,GREEN,3)
        self.c5 = Cell(250,int(round(150+200*math.sqrt(3))),36,GREEN,4)
        self.c6 = Cell(450,int(round(150+200*math.sqrt(3))),36,GREEN,5)
        self.c7 = Cell(350,320,90,GREEN,6)
        self.cellList = [self.c1,self.c2,self.c3,self.c4,self.c5,\
                         self.c6,self.c7]
        self.maximum = maxEnergy

class Level_6(object):
    def __init__(self):
        self.c1 = Cell(600,200,5,(55,55,55))
        self.c2 = Cell(500,300,10)
        self.c3 = Cell(240,420,10,(200,0,0))
        self.maximum = 70
        self.cellList = [self.c1,self.c2,self.c3]

class Level_3(object):
    def __init__(self):
        self.c1 = Cell(100,400,5,(55,55,55))
        self.c2 = Cell(500,370,46)
        self.c3 = Cell(240,230,7,(200,0,0))
        self.c4 = Embracer(520,80,3,(200,0,0))
        self.c5 = Embracer(230,125,10,GREEN)
        self.c6 = Cell(650,600,40,(200,0,0))
        self.c7 = Cell(560,500,60)
        self.maximum = 80
        self.cellList = [self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7]

class Level_2(object):
    def __init__(self):
        self.c1 = Cell(350,300,0,(55,55,55))
        self.c2 = Cell(500,370,30)
        self.c3 = Cell(240,230,20,(200,0,0))
        self.c4 = Cell(200,550,15,(200,0,0))
        self.c5 = Embracer(230,125,10,GREEN)
        self.c6 = Cell(450,180,15,(200,0,0))
        self.c7 = Cell(560,500,10,(55,55,55))
        self.maximum = 80
        self.cellList = [self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7]

class Level_5(object):
    def __init__(self):
        self.cellList = []
        for x in range(150,650,150):
            if x%300 == 0:
                cell = Cell(x,x,30,GREEN)
            else:
                cell = Cell(x,x,25,GREEN)
            self.cellList.append(cell)
        for x in range(150,650,150):
            cell = Cell(x,750-x,28,(200,0,0))
            self.cellList.append(cell)
        self.maximum = 90

class Level_7(object):
    def __init__(self):
        blueY = random.randint(350,450)
        self.c1 = Cell(260,180,25)
        self.c2 = Cell(230,280,25)
        self.c3 = Cell(230,500,30)
        self.c4 = Cell(260,600,30)
        self.c5 = Cell(500,200,50,RED)
        self.c6 = Cell(500,550,40,RED)
        self.c7 = Embracer(640,640,15,RED)
        self.c8 = Embracer(640,60,15,RED)
        self.c9 = Cell(550,blueY,80,(0,0,255))
        self.c10 = Embracer(50,350,10,(0,0,255))
        self.c11 = Embracer(100,100,5,(0,0,255))
        self.c12 = Embracer(350,650,8,GREEN)
        self.cellList = [self.c1,self.c2,self.c3,self.c4,self.c5,\
                         self.c6,self.c7,self.c8,self.c9,self.c10,\
                         self.c11,self.c12]
        self.maximum = 90
        
class Lock(object):
    def __init__(self,x,y):
        self.x,self.y = x,y
        self.radius = 30
        self.color = (255,255,0) # yellow

    def drawLock(self,surface):
        #self.drawCirc(surface)
        self.drawArr(surface)

    def drawCirc(self,surface):
        center = (self.x,self.y)
        pygame.draw.circle(surface,self.color,center,self.radius,2)

    def drawArr(self,surface):
        self.drawArrAng(surface,0)
        self.drawArrAng(surface,1)
        self.drawArrAng(surface,2)
        self.drawArrAng(surface,3)

    def drawArrAng(self,surface,angle):
        ang = angle*math.pi/2 + 3*math.pi/4

        if angle % 2 == 1: # topRight/bottomLeft
            tip1 = (self.x+self.radius*math.cos(ang),\
                    self.y-self.radius*math.sin(ang))
            delta = +30 if angle == 1 else -30
            tip2 = (tip1[0]-delta-5,tip1[1]+delta-5)
            tip3 = (tip1[0]-delta+5,tip1[1]+delta+5)

        else:
            tip1 = (self.x+self.radius*math.cos(ang),\
                    self.y-self.radius*math.sin(ang))
            delta = +30 if angle == 2 else -30
            tip2 = (tip1[0]+delta-5,tip1[1]+delta+5)
            tip3 = (tip1[0]+delta+5,tip1[1]+delta-5)
        pygame.draw.polygon(surface,self.color,(tip1,tip2,tip3))

class Chain(object):
    def __init__(self,startx,starty,endx,endy,startCell,endCell,color=GREEN):
        self.color = color
        self.startx,self.starty = startx,starty
        self.endx,self.endy = endx,endy
        self.startCell=startCell
        self.endCell=endCell
        if endx != startx:
            self.tan = float(starty-endy)/(endx-startx)
            if self.tan > 0:
                self.direction = math.atan(self.tan)
                if starty < endy:
                    self.direction += math.pi
            else:
                self.direction = math.atan(self.tan)+math.pi
                if startx < endx:
                    self.direction += math.pi
        else:
            self.direction = math.pi/2 if starty > endy else 3*math.pi/2
        self.chainFinalLen = ((endx-startx)**2+(endy-starty)**2)**0.5
        self.dotNum = (self.chainFinalLen-23)/(3*2) # 23 is cell radius
        # 3 is the radius of the small dot,so 3*2 is the diameter
        self.chainInit = 1 # initially of length (dot) 1.
        self.chainList = [(startx,starty)]
        self.shouldBreak,self.breakInd = False,None
        self.shouldCollapse = False
        self.shouldGrow = True # at first, every chain should grow
        self.growNum = 0 # to control the speed of the growth
        self.lineHalfLength,self.freq = 5.5,5
        self.subtractCellValue = False # growing chain costs life value
        self.determineInd()

    def determineInd(self):
        cutoff1,cutoff2 = 280,450
        if cutoff1 <= self.chainFinalLen <= cutoff2:
            self.IndNum = 3
            self.shiningInd = [-1,-14,-27] # the dot on the chain that shines
        elif self.chainFinalLen > cutoff2:
            self.IndNum = 4
            self.shiningInd = [-1,-14,-27,-40]
        else:
            self.IndNum = 2
            self.shiningInd = [-1,-19]
        # two dots have distance difference of 18
    
    def grow(self):
        startx = self.chainList[-1][0]
        starty = self.chainList[-1][1]
        cycle = 8
        index = len(self.chainList)%cycle
        if index == 0:
            angle = math.atan(1/2.)
        elif index == 1:
            angle = math.atan(math.sqrt(2)/4)
        elif index == 2:
            angle = 0
        elif index == 3:
            angle = math.atan(-math.sqrt(2)/4)
        elif index == 4:
            angle = math.atan(-1/2.)
        elif index == 5:
            angle = math.atan(-math.sqrt(2)/4)
        elif index == 6:
            angle = 0
        elif index == 7:
            angle = math.atan(math.sqrt(2)/4)
        direction,regularRad = self.direction + angle,12
        newx = startx + 6*math.cos(direction)
        newy = starty - 6*math.sin(direction)

        self.chainList.append((newx,newy))
        if len(self.chainList)%2 == 0: # two dots worth one life value of cell
            self.subtractCellValue = True
        if dist(newx,newy,self.endx,self.endy,regularRad):
            self.shouldGrow = False
            self.subtractCellValue = False

    def drawChain(self,surface):
        length = self.lineHalfLength
        angle = self.direction
        for i in range(len(self.chainList)):

            dot = self.chainList[i]
            dotx = int(round(dot[0]))
            doty = int(round(dot[1]))
            color = self.color
            for index in self.shiningInd:
                if index == i:
                    color = (255,255,0)
                    break
            
            # the dot that should shine shines.
            pygame.draw.circle(surface,color,(dotx,doty),3,0)
            
            linestx = int(round(dotx-length*math.sin(math.pi-angle)))
            linesty = int(round(doty+length*math.cos(math.pi-angle)))
            lineEndx = int(round(dotx+length*math.sin(math.pi-angle)))
            lineEndy = int(round(doty-length*math.cos(math.pi-angle)))           
            pygame.draw.line(surface,color,(linestx,linesty),\
                             (lineEndx,lineEndy),2)

def dist(x1,y1,x2,y2,r):
    return ((x1-x2)**2+(y1-y2)**2)**(0.5) <= r+3

my_CellWar = CellWar()
mlAgent = ml2k17(my_CellWar.numCells);
my_CellWar.run()