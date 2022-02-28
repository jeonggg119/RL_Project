# -----------------------------------------------------------------------------------------------
# (Lab) Flappybird_app_lab.py - version 2020.11.23v1 
# -----------------------------------------------------------------------------------------------

import numpy as np
import torch
from torch import nn
from torch import optim

import os
from os import walk
import itertools
import statistics
import time
import requests
import json
import warnings
import random

import gym
from model_dqn import agent, brain, replay_memory


#-------------------------------------------------------
# *** 기본 설정 (변경 O) ***
#-------------------------------------------------------

NICKNAME = '쩡' # 표시용 별명

# Course Progress Tracker 의 Register 페이지에서 Auth Code 항목을 복사한 후 붙여넣음
AUTH_CODE = '3d03274718f3fb9560a648d7a9f7e849'  # 인증코드

# 훈련 도중 지금까지 최고기록 모델 저장 여부
SAVE_BEST_MODEL = True

# 하이퍼파라미터 튜닝 때도 모델 저장 여부
SAVE_BEST_MODEL_ON_TUNING = True

# 게임 화면 스킵 프레임 수
RENDER_SKIP = 100 # (0: 정상속도, 클수록 빠름) (변경 가능)


LINE_STR = '-' * 120


#-------------------------------------------------------
# *** 하이퍼파라미터 (1) (변경 O) ***
#-------------------------------------------------------
NUM_EP_EVAL = 700 # 평가 에피소드 수 (변경 가능)
NUM_EP_TRAIN = 500 # 훈련 에피소드 수 (변경 가능) # 많이 !
NUM_EP_TUNE = 200 # (하이퍼파라미터 튜닝용) 훈련 에피소드 수 (변경 가능)
NUM_ATTEMPTS_TUNE = 10 # 하이퍼파라미터 튜닝 시도 수 (변경 가능)

#-------------------------------------------------------
# *** 하이퍼파라미터 (2) - For Training (변경 O) ***
#-------------------------------------------------------
LEARNING_RATE = 0.0001
BATCH_SIZE = 200
DROPOUT = 0.1
GAMMA = 0.99
REPLAY_MEMORY_CAPACITY = 70000
OPTIMIZER_ID = 'adam'
MODEL_ID = 'model1'

#-------------------------------------------------------
# *** 하이퍼파라미터 (2) - For Tuning (변경 O) ***
#-------------------------------------------------------
RANDOM_LEARNING_RATE_RANGE = ( 0.0001, 0.001 )
RANDOM_BATCH_SIZE_RANGE = ( 100, 200)
RANDOM_DROPOUT_RANGE = ( 0.1, 0.3, 0.5 )
RANDOM_GAMMA_RANGE = ( 0.99, 0.9)
RANDOM_REPLAY_MEMORY_CAPACITY_RANGE = ( 5000, 10000, 50000 )
RANDOM_OPTIMIZER_ID = ('adam', 'sgd')
RANDOM_MODEL_ID = ('model1', 'model2', 'model3')


#-------------------------------------------------------
# *** 하이퍼파라미터 (3) 게임 환경 (변경 O) ***
#-------------------------------------------------------

# 파이프 갭 랜덤 여부 True/False 
IS_RANDOM_GAP = False 

# 파이프 갭 랜덤이 False일 경우 파이프 갭 크기 (120 ~ 170) (변경 가능)
GAP_SIZE = 160 # 모든 파이프 갭에서 전반적으로 잘 나오도록 훈련!(130~170 랜덤도 하나 있음)


# -----------------------------------------------------------------------------------------------
# FlappyBirdApp
# -----------------------------------------------------------------------------------------------
class FlappyBirdApp():

    def __init__(self):

        global AUTH_CODE

        # -----------------------------------------------------------------------------------------------
        # Check Auth Code
        # -----------------------------------------------------------------------------------------------
        if not AUTH_CODE:
            AUTH_CODE = input('Input Auth Code( http://esohn.be/register/ ) > ')
        
        # -----------------------------------------------------------------------------------------------
        # Download FB Helper
        # -----------------------------------------------------------------------------------------------
        if not os.path.isfile('flappybird_helper.py'):
            self.download_fb_helper()


        # Import FB Helper
        import flappybird_helper

        # Create TSP Helper object
        self.helper = flappybird_helper.FlappyBirdHelper( self, nickname=NICKNAME, auth_code=AUTH_CODE,
            num_ep_train=NUM_EP_TRAIN, num_ep_tune=NUM_EP_TUNE, num_ep_eval=NUM_EP_EVAL, num_attempts_tune=NUM_ATTEMPTS_TUNE, 
            is_random_gap=IS_RANDOM_GAP, gap_size=GAP_SIZE, render_skip=RENDER_SKIP, save_best_model=SAVE_BEST_MODEL, save_on_tuning=SAVE_BEST_MODEL_ON_TUNING )

        print( '\n\n{0}  Flappybird Helper Version: {1}  {0}\n\n'.format( '-'*10, self.helper.FB_HELPER_VERSION) )



    def init_agent(self, env, model_to_use=None, random_hyperparameters=False):

        # -----------------------------------------------------------------------------------------------
        # 상태 및 액션 갯수 (변경 X)
        # -----------------------------------------------------------------------------------------------
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # -----------------------------------------------------------------------------------------------
        # 하이퍼파라미터 지정 (변경 X)
        # -----------------------------------------------------------------------------------------------
        if random_hyperparameters:
            # 적정 범위의 랜덤한 값 지정
            self.learning_rate = random.choice( RANDOM_LEARNING_RATE_RANGE )
            self.batch_size = random.choice( RANDOM_BATCH_SIZE_RANGE )
            self.dropout_rate = random.choice( RANDOM_DROPOUT_RANGE )
            self.gamma = random.choice( RANDOM_GAMMA_RANGE )
            self.replay_memory_capacity = random.choice( RANDOM_REPLAY_MEMORY_CAPACITY_RANGE )
            self.optimizer_id = random.choice( RANDOM_OPTIMIZER_ID )
            self.model_id = random.choice( RANDOM_MODEL_ID )
        else:
            # 튜닝이 끝난 하이퍼파라미터를 집중 훈련
            self.learning_rate = LEARNING_RATE
            self.batch_size = BATCH_SIZE
            self.dropout_rate = DROPOUT
            self.gamma = GAMMA
            self.replay_memory_capacity = REPLAY_MEMORY_CAPACITY
            self.optimizer_id = OPTIMIZER_ID
            self.model_id = MODEL_ID

        # -----------------------------------------------------------------------------------------------
        # *** 신경망 모델 (변경 O, 다양한 구조를 가진 모델들을 시도해볼 수 있음) ***
        # -----------------------------------------------------------------------------------------------

        model1 = nn.Sequential(
            nn.Linear( num_states, 64 ),
            nn.ReLU(),
            nn.Linear( 64, 128), 
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            
            nn.Linear( 128, num_actions ),
        )

        model2 = nn.Sequential(
            nn.Linear( num_states, 64 ), 
            nn.ReLU(),
            nn.Linear( 64, 64 ), 
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            
            nn.Linear( 64, num_actions ),
        )

        model3 = nn.Sequential(
            nn.Linear( num_states, 64 ), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout( p=self.dropout_rate ),
            
            nn.Linear( 64, num_actions),
        )


        # -----------------------------------------------------------------------------------------------
        # 모델 Print (변경 X)
        # -----------------------------------------------------------------------------------------------

        if model_to_use:
            model = model_to_use
            self.model_id = 'Loaded model'
            print('Using the loaded model...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model1':
            model = model1
            print('Using the model1...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model2':
            model = model2
            print('Using the model2...\n\n {}\n{}\n'.format(model, LINE_STR))
        elif self.model_id == 'model3':
            model = model3
            print('Using the model3...\n\n {}\n{}\n'.format(model, LINE_STR))


        # -----------------------------------------------------------------------------------------------
        # *** Optimizer (변경 O, Optimizer 추가 등 가능) ***
        # -----------------------------------------------------------------------------------------------

        if self.optimizer_id == 'adam':
            optimizer = optim.Adam( model.parameters(), lr=self.learning_rate )

        elif self.optimizer_id == 'sgd':
            optimizer = optim.SGD( model.parameters(), lr=self.learning_rate )


        # -----------------------------------------------------------------------------------------------
        # Agent 초기화 (변경 X)
        # -----------------------------------------------------------------------------------------------

        memory = replay_memory.ReplayMemory( self.replay_memory_capacity )

        return agent.Agent( num_actions = num_actions, batch_size = self.batch_size, gamma = self.gamma, replay_memory = memory, model = model, optimizer = optimizer )

        # -----------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------
    # FB Helper download - (변경 X)
    # -------------------------------------------------------------------------------------------
    def download_fb_helper(self):

        FB_HELPER_URL = 'https://s3.ap-northeast-2.amazonaws.com/esohn.be/python/flappybird_helper.py'
        
        r = requests.get(FB_HELPER_URL, allow_redirects=True)
       
        if r.status_code == 200:
            with open('flappybird_helper.py', 'wb') as f:
                f.write( r.content )
        else:
            print('ERROR: unable to download fb_helper.py!')



if __name__=="__main__":

    warnings.filterwarnings("ignore")

    app = FlappyBirdApp()

    app.helper.menu()

    app.helper.env.close()



