import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Home: NextPage = () => {

  const wrapper = `import numpy as np
import gym

class StochasticFrameSkip(gym.Wrapper):
  def __init__(self, env, n, stickprob):
      gym.Wrapper.__init__(self, env)
      self.n = n
      self.stickprob = stickprob
      self.curac = None
      self.rng = np.random.RandomState()
      self.supports_want_render = hasattr(env, "supports_want_render")

  def reset(self, **kwargs):
      self.curac = None
      return self.env.reset(**kwargs)

  def step(self, ac):
      done = False
      totrew = 0
      for i in range(self.n):
          # First step after reset, use action
          if self.curac is None:
              self.curac = ac
          # First substep, delay with probability=stickprob
          elif i==0:
              if self.rng.rand() > self.stickprob:
                  self.curac = ac
          # Second substep, new action definitely kicks in
          elif i==1:
              self.curac = ac
          if self.supports_want_render and i<self.n-1:
              ob, rew, done, info = self.env.step(self.curac, want_render=False)
          else:
              ob, rew, done, info = self.env.step(self.curac)
          totrew += rew
          if done: break
      return ob, totrew, done, info`

    const searchSpace = `# Represents all possible moves.
# Converts moves to button presses which can be used with env.step()
# 
# ex:
#	# returns an array of 12 ints representing the button presses for this action
#	buttons = ActionSpace.move_right()	
#	
#	env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
#	env.step(buttons)	# sends move to game emulator
#
# ActionSpace is treated like a namespace. It is not intended to be instantiated.

class ActionSpace:
  # --------------------------------- FIELDS --------------------------------

  # Index of each possible move.
  # These values are constant and should not be change at runtime.
  STAND_STILL = 0
  RIGHT = 1
  JUMP_RIGHT = 2
  JUMP = 3
  JUMP_LEFT = 4
  LEFT = 5
  CROUCH = 6
  ROLL = CROUCH
  # TODO: Do we need one for spin dash
  
  # look up table which maps action specified by an index [0, 7]
  # to a combination of button presses.
  # *** There is no way to control rolling left or right. ***
  # *** Momentum determines direction of roll. ***
  BUTTONS = [
    # 0  1  2  3  4  5  6  7  8  9 10 11
    # A  B  C     ^  v  <  >              
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],	# 0 - stand still
    [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # 1 - right
    [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], # 2 - jump right
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], # 3 - jump		
    [ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ], # 4 - jump left
    [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ], # 5 - left
    [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ], # 6 - crouch/roll
    # TODO: Do we need spin dash?
  ]

  # --------------------------------- METHODS -------------------------------

  # returns button presses which coorespond with standing still
  def stand_still() -> list:
    return ActionSpace.BUTTONS[ActionSpace.STAND_STILL]

  # returns button presses which coorespond with moving/running right
  def move_right() -> list:
    return ActionSpace.BUTTONS[ActionSpace.RIGHT]
      
  def jump_right() -> list:
    return ActionSpace.BUTTONS[ActionSpace.JUMP_RIGHT]
      
  def jump() -> list:
    return ActionSpace.BUTTONS[ActionSpace.JUMP]
      
  def jump_left() -> list:
    return ActionSpace.BUTTONS[ActionSpace.JUMP_LEFT]
    
  def move_left() -> list:
    return ActionSpace.BUTTONS[ActionSpace.LEFT]

  def crouch() -> list:
    return ActionSpace.BUTTONS[ActionSpace.CROUCH]

  def roll() -> list:
    return ActionSpace.BUTTONS[ActionSpace.ROLL]

  # returns button presses as a list/array by index.
  # see class ActionSpace fields for aliases for each index
  def move(index) -> list:
    return ActionSpace.BUTTONS[index]

  # Returns the number of possible moves (7 moves)
  def get_n_moves() -> int:
    return len(ActionSpace.BUTTONS)

  # Converts button presses to a string representing the action
  def to_string(buttons) -> str:
    if buttons == ActionSpace.BUTTONS[ActionSpace.STAND_STILL]:
      return 'X'
    if buttons == ActionSpace.BUTTONS[ActionSpace.RIGHT]:
      return '>'
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP_RIGHT]:
      return '/'
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP]:
      return '|'
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP_LEFT]:
      return '\\'
    if buttons == ActionSpace.BUTTONS[ActionSpace.LEFT]:
      return '<'
    if buttons == ActionSpace.BUTTONS[ActionSpace.CROUCH]:
      return 'o'

  def to_string_big(buttons) -> str:
    if buttons == ActionSpace.BUTTONS[ActionSpace.STAND_STILL]:
      return 'XXXXXXX'
    if buttons == ActionSpace.BUTTONS[ActionSpace.RIGHT]:
      return '    -->'
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP_RIGHT]:
      return '   |-->'
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP]:
      return '   |   '
    if buttons == ActionSpace.BUTTONS[ActionSpace.JUMP_LEFT]:
      return '<--|   '
    if buttons == ActionSpace.BUTTONS[ActionSpace.LEFT]:
      return '<--    '
    if buttons == ActionSpace.BUTTONS[ActionSpace.CROUCH]:
      return 'vvvvvvv'

  # Returns true if buttons is the button press of a jump action
  def is_jump(buttons) -> bool:
    return \
      buttons == ActionSpace.jump() or \
      buttons == ActionSpace.jump_right() or \
      buttons == ActionSpace.jump_left()`

      const rewards = `import os
import sys
import numpy as np
from a_queue import *
from action_space import ActionSpace

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

# Calculates a reward which can be used in reinforcement learning.
# Compliments the reward that is automatically calculated by gym retro.
# class RewardSystem allows adding more rewards to it. 
# And object of RewardSystem should call calc_rewards() every frame in order to calculate rewards accurately.
# Skipping frames might cause some rewards to be over/under calculated.
class RewardSystem:
  
  # --------------------------------- Class XPos ----------------------------
  # This mirrors the contest Xpos reward. Calculates rew based on raw x position @ a rate of 1 pt for each x traveled
  class XPos:
    def __init__(self):
      self.__x_rew = 1						# reward for moving 1 pixel to the right
      self.__x_prev = 0						# coordinate of last x position
    
    def init(self, info) -> None:
      self.__x_prev = info['x']
      
    def calc_reward(self, info, action) -> int:
      # rew is new pos - old pos
      rew = (info['x']  - self.__x_prev) * self.__x_rew # can change scale if interested
      # set old pos to new pos for next iteration
      self.__x_prev = info['x']
      return rew
      
    def to_string(self) -> str:
      return "XPos"

  # --------------------------------- Allow Backtracking -------------------------

  class Backtracking():
    def __init__(self):
      None

    def init(self, info) -> None:
      self.__x_rew = 1						# reward for moving 1 pixel to the right
      self._max_x = 0

    def calc_reward(self, info, action) -> int:
      # rew is maximum x traveled
      rew = max(0, info['x'] - self._max_x) * self.__x_rew
      # set new max for next iteration

      self._max_x = max(self._max_x, info['x'])
      return rew
      
    def to_string(self) -> str:
      return "Allow Backtracking"

  # --------------------------------- Class Contest -------------------------
    
  class Contest:

    def __init__(self):
      None
      
    def init(self, info) -> None:
      self.__end_x = info['screen_x_end']
      self.__prev_progress = 0
      self.__frame = 0

    def calc_progress(self, info):
      
      return info['x'] / self.__end_x 

    def calc_reward(self, info, action) -> int:
      
      progress = self.calc_progress(info)
      rew = (progress - self.__prev_progress) * 9000
      self.__prev_progress = progress

      # Reward for completing level quickly
      if progress >= 1:
        rew = rew + (1 - np.clip(self.__frame / 18000, 0, 1)) * 1000
      self.__frame+=1
      return rew

      
    def to_string(self) -> str:
      return "Contest"
      
  # --------------------------------- Class Complex -------------------------
  
  class Complex:

    def __init__(self):
      self.__frame_counter = 0

      # Reward weights:
      # Specifies how good each action is.
      # good actions are positive
      # bad actions are negative
      self.__ring_rew = 1000					# reward for each ring collected
      self.__ring_loss_rew = 0#-10			# penalty for losing any number of rings
      self.__ring_deficient_rew = -5			# penalty for not having rings (applied every frame we don't have rings)

      self.__ring_count = 0					# how many rings do we have

      self.__robot_rew = 1					# reward for destroying each robot
      self.__robot_count = 0					# how many robots have been destroyed

      self.__score_rew = 10					# 10 point for every new point scored
      self.__score_count = 0					# how many points do we have

      self.__life_rew = 1000					# Reward for collecting an extra life
      self.__life_penalty = -self.__ring_loss_rew	# Penalty for dying
      self.__life_count = 0					# how many lives do we have

      self.__x_rew = 1						# reward for moving 1 pixel to the right
      self.__x_prev = 0						# coordinate of last x position

      self.__x_explore_rew = 10				# reward for exploring 1 pixel further than before
      self.__x_max = 0						# the furthest right we have moved along the x axis.

      self.__y_prev = 0						# coordinate of last y position

      self.__items_rew = 1					# reward for collecting item boxes

      # --- Location Specific Rewards ---
      # self.__location_rewards = { '' }

      self.__jump_rew = -20					# penalty for each jump
      self.__jump_history = AQueue()			# timestamps of most recent jumps.
      self.__jump_tolerance_count = 2			# allows jumping x times without penalty every so many frames
      self.__jump_tolerance_period = 10		# allows jumping x times without penalty every this many frames

    # Sets initial conditions of current epoch. 
    # Some rewards are based on previous actions. 
    # This method sets the initial conditions of the new epoch so that rewards can be based on them.
    # Sets things like, ring count, current x position, and current score.
    # Call this method whenever the game is reset or parts of a level are skipped.
    # ! This is not a constructor !
    def init(self, info) -> None:
      # TODO: More reward/penalty ideas
      # Penalty (For getting stuck): Trying to move right but not increasing 'x'
      # Penalty (For getting stuck): Trying to move left but not decreasing 'x'
      # Penalty: For losing a life. Getting hit without rings. 
      self.__frame_counter = 0
      self.__ring_count = info['rings']
      self.__robot_count = 0	# TODO: ???
      self.__score_count = info['score']
      self.__life_count = info['lives']
      self.__x_prev = info['x']
      self.__x_max = self.__x_prev
      self.__y_prev = info['y']
      self.__jump_history.clear()

    # Calculates reward based on environment
    # env  		- environment of gym retro
    # obs  		- observation - currently rendered frame as numpy.ndarray
    # info 		- contains game state information like position, score, ring count and speed
    # action		- the most recent action made by agent stored as a list of ints. See ActionSpace.
    # returns 		- recalculated reward as an int 
    def calc_reward(self, info, action) -> int:
      self.__frame_counter += 1						# Increment frame counter

      reward = 0

      reward += self.__calc_ring_reward(info)
      reward += self.__calc_robot_reward(info) 		# TODO: this doesn't do anything yet
      reward += self.__calc_score_reward(info)
      reward += self.__calc_life_reward(info)
      reward += self.__calc_x_reward(info)
      reward += self.__calc_items_reward(info)		# TODO: this doesn't do anything yet
      reward += self.__calc_jump_reward(action)		

      return reward

    # Calculates reward for collecting/loosing rings
    def __calc_ring_reward(self, info) -> int:
      rings_curr = info['rings'] 
      ring_diff = rings_curr - self.__ring_count

      self.__ring_count = rings_curr

      # --- Reward for collecting/loosing rings ---
      reward = 0

      if ring_diff >= 0:
        # reward for collecting each ring
        reward += self.__ring_rew * ring_diff	
      else:
        # penalize for losing any number of rings
        reward += self.__ring_loss_rew			

      # --- Penalty for not having rings ---
      if rings_curr == 0:
        reward += self.__ring_deficient_rew

      return reward

    # Calculates reward for destroying a robot
    def __calc_robot_reward(self, info) -> int:
      # TODO: Don't know
      return 0

    # Calculates reward for increasing score (This will overlap with other rewards but it will still work)
    def __calc_score_reward(self, info) -> int:
      score_curr = info['score']
      score_diff = score_curr - self.__score_count	

      self.__score_count = score_curr

      return self.__score_rew * score_diff

    # Calculates a reward for collecting a life (or "one up")
    def __calc_life_reward(self, info) -> int:
      rew = 0

      life_curr = info['lives']
      life_diff = life_curr - self.__life_count

      # Did we gain or lose a life?
      if life_diff >= 0:
        # We gained a life :)
        rew += self.__life_rew * life_diff
      else:
        # We lost a life :(
        rew += self.__life_penalty * life_diff

      self.__life_count = life_curr

      return rew

    # Calculates reward for moving right
    def __calc_x_reward(self, info) -> int:
      x_curr = info['x']				
      x_diff = x_curr	- self.__x_prev			# how much did we move since last frame (same as x velocity)
      x_explored = x_curr - self.__x_max		# how much further right did we move than before

      self.__x_max = (x_curr if x_curr > self.__x_max else self.__x_max)
      self.__x_prev = x_curr

      reward = 0

      # Reward for every new pixel we move right
      if x_explored > 0:
        reward += self.__x_explore_rew * x_explored 

      # Reward/Penalize for every pixel, since last frame, we moved right/left
      reward += self.__x_rew * x_diff 

      return reward

    def __calc_items_reward(self, info) -> int:
      # TODO: I don't know
      return 0

    # Calculates penalty for jumping.
    # Only penalizes for jumping more than allowed number.
    # Agent is allowed to jump without penalty x number of times during any period of y frames.
    # For every extra jump, the agent is penalized
    def __calc_jump_reward(self, action) -> int:
      rew = 0

      # Did we jump?
      if ActionSpace.is_jump(action):
        # Yes. We jumped.

        # Record this jump.
        self.__jump_history.push(self.__frame_counter)	# We jumped at frame x

        # Penalize but only if we jumped too much.
        if self.__jump_history.size() > self.__jump_tolerance_count:
          rew = self.__jump_rew

      # Update jump history. Remove old jumps.
      while self.__jump_history.size() > 0:
        oldest = self.__jump_history.front()

        if oldest + self.__jump_tolerance_period <= self.__frame_counter:
          self.__jump_history.pop()		# remove oldest jump
        else:
          break

      return rew
  
    def to_string(self) -> str:
      return "Complex"`

  return (
    <>
      <Head>
        <title>DeepQ Tuning</title>
        <meta name="description" content="First, we modified our input to our Deep Q Agent to take 4 consecutive concatenated frames as input, which provides richer contextual information to the model such as direction of travel and momentum." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">DeepQ Tuning</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center mb-3">Moving forward with Deep Q Learning as our model for the reinforcement learning agent, the team shifted efforts toward optimization of this model. As Deep Q utilizes images from the emulator environment as input to map button press actions to optimal rewards, we identified three areas in which to experiment: image input, actions, and reward structure. </p>
            <p className="text-center mb-3">The OpenAI Gym Retro emulator processes images at 60 frames per second (fps). However, we posited that this frame rate was faster than humans could process the image, make a decision, and press a button. Therefore, we did not need to map an input to each frame, in the emulator and could take advantage of a reduced frame rate in several ways. First, we modified our input to our Deep Q Agent to take 4 consecutive concatenated frames as input, which provides richer contextual information to the model such as direction of travel and momentum. Next, we applied stochastic frame skipping to these images prior to concatenation. This means that rather than seeing 4 directly consecutive frames, the environment may skip up to 4 frames before concatenating the next image. This effectively reduces our frame rate to 15 fps, which better aligns with human button press times.</p>
            <p className="text-center mb-3">According to research by Google&apos;s DeepMind, Deep Q Learning is optimized around relatively small, discrete action spaces. However, the Sega Genesis controller emulation contains 12 buttons, which can be pressed in 12^2 or 4,096 combinations. Therefore, we opted to reduce the action space to the most viable moves for Sonic to make in the environment. This eliminates redundant combinations such as pressing up and down at the same time. Our final action space is reduced to 7 basic actions: stand still, jump, walk right, jump right, walk left, jump left, and crouch/roll.</p>
            <p className="text-center">The OpenAI Gym Retro environment provides several reward functions as a baseline to inform the agent of the success of its previous move. These include raw “x position”, indicating how far right Sonic has moved in the environment and “contest” which calculates x position relative to environment length and adds a bonus for quickly completing a level. The team added two additional reward functions for experimentation. First, we added a modified version of “contest” which did not penalize backtracking, allowing Sonic to move left to gain momentum in support of overcoming tall obstacles. Additionally, we added a complex reward function which included relative x position, collecting rings, eliminating enemies, and penalized excessive jumping to prioritize forward momentum. Our most successful runs utilized the “backtracking” reward function.</p>
          </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/basic_image_processing"}>Basic Image Processing -&gt;</Link>
          </button>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
            <p>/source/interface/wrappers.py</p>
            <SyntaxHighlighter
              showLineNumbers
              style={vscDarkPlus}
              languag="python">
                {wrapper}
            </SyntaxHighlighter>
            </div>
        </article>
        <article>
          <div className="m-6 md:mx-8 lg:mx-16">
            <p>/source/interface/action_space.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {searchSpace} 
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="m-6 md:mx-8 lg:mx-16">
            <p>/source/learning/reward_system.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {rewards} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
