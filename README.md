<a name="br1"></a> 

Dynamical Tilt Maze Solving with 4 6DoF Franka

Robots Using SARSA with Eligibility Trace

Cankut Bora Tuncer

Busenaz Kerimgil

Onur Alp C¸ ulha

Electrical and Electronics Engineering Electrical and Electronics Engineering Electrical and Electronics Engineering

Bilkent University

Ankara, Turkey

Bilkent University

Ankara, Turkey

Bilkent University

Ankara, Turkey

bora.tuncer@ug.bilkent.edu.tr

busenaz.kerimgil@ug.bilkent.edu.tr

alp.culha@ug.bilkent.edu.tr

¨

Osmanalp Omerog˘lu

Kemal Anıl Sarıtas¸

Electrical and Electronics Engineering

Electrical and Electronics Engineering

Bilkent University

Bilkent University

Ankara, Turkey

Ankara, Turkey

osmanalp.omeroglu@ug.bilkent.edu.tr

anil.saritas@ug.bilkent.edu.tr

Abstract—In reinforcement learning (RL), it is fundamental for

In this project, we implement a reinforcement learning (RL)

non-human agents to be most efﬁcient even in complex situations algorithm to solve tilt mazes. The RL algorithm we selected

like dynamic maze problems. Moreover, a dynamic environment

to utilize is the Sarsa with Eligibility Trace. We trained the

algorithm within our Rai environment setup, which includes

four 6 degrees of freedom (DoF) Franka Robots to manipulate

selection is crucial as the outcomes are relatively versatile like the

real world. This paper focuses on working the four robotic arms

in harmony to tilt the maze throughout the path of the ball with

precise movement. Our methodology involved experimenting with the platform. Using RL, instead of labeling all state-action

six different dynamical tilting mazes with the SARSA algorithm pairs, we established a reward function to update the Q-

with eligibility traces. We expect our ﬁndings to enhance the real-

table and, consequently, the policy. Hence, we overcome the

problems that would occur in classical supervised learning.

To implement the Sarsa with Eligibility Trace algorithm, we

needed to discretize the states since the environment acts as

world applications of RL with the time efﬁciency of our software

experiments.

I. INTRODUCTION

A tilt maze is a unique type of puzzle where the objective is a continuous setting. We have implemented several methods

to move a ball through a labyrinth by tilting the platform it is to ﬁnd a sufﬁcient discretization for our model to learn in

resting on. In other words, the player should tilt the platform a relatively short period of time. Moreover, the environment

so that the ball rolls in the desired direction, avoiding dead setup must be precise to avoid unwanted scenarios, such as the

ends to reach a speciﬁc goal point. This task requires careful ball falling off the table. These types of scenarios slow down

manipulation, a sense of balance, and strategic thinking since the learning and, more importantly, might stop the training

each tilt can signiﬁcantly affect the ball’s journey. This project process as a whole.

aims to build a system with non-human agents to solve tilt

mazes autonomously.

We designed six different maps and trained the algorithm

on these maps. We recorded the collected reward per episode,

The integration of problem-solving with physical interac- the win rate for each starting position, and the policy table for

tion forms an interesting problem to study, illustrating how each map. We compared the collected rewards with human

machines can adapt and react in dynamic environments. The agents. The model’s performance gets very close to that of

idea involves decision-making that affects the state of the humans and even exceeds in several maps.

environment. This concept can be expanded to broader real-

Below is a bullet list of the main contributions and insights

world applications, such as automatic navigation systems gained:

where the system should respond to the dynamic changes in

the environment.

•

Demonstrated that SARSA Q-Trace can be used to train

an agent in a dynamic and challenging environment such

as a tilt maze.

Puzzle-solving in a dynamic and physically interactive en-

vironment requires a lot of practice, even for human agents.

Agents should understand that their actions affect the state

of the environment. Therefore, they should be able to adapt

to the environment and make strategic decisions. Classical

supervised learning techniques are not suitable for this task. A

human labeling every state-action pair is inefﬁcient (or perhaps

impossible) in terms of time and effort.

• Although the environment is continuous, it is proven that

with careful discretization of the state representation, it

is possible to achieve decent performance.

• It is shown that the trained agent can perform similarly

to a human and, in some cases, even better.

• Utilized the Simulation interface and gained know-how

about Rai.



<a name="br2"></a> 

• By training our agent in Rai, this work can be referenced C. Robots Outperforming Humans

for future work for implementing and simulating RL by

using the physics engine in Rai.

Deep RL has recently begun to outperform humans in

various games, including chess, Go [18], and Atari video

games. Aside from that, the performance of 2D and 3D maze

navigation ([19], [20], [21]) began to match or exceed the hu-

man score. However, the amount of success shown in systems

that require real interactions with the physical environment

II. RELATED WORK

A. RL Algorithmn

In this work, we exploit recent advances in model-free RL remains limited. For instance, such as autonomous drones

and its ability to make informed decisions about potentially outperform human world champions in drone racing [22].

successful behaviors by planning for the future [3]. Rein- Mazes which tested in a lab environment are regarded as

forcement Learning stands out as a contemporary method suitable testing grounds for further investigation of the real-

for robotic path planning. It operates as an online learning world capabilities of artiﬁcial intelligence due to their low cost,

technique, dynamically interacting with the environment to wide availability, and tough character. Research conducted

make real-time decisions as needed [2]. Robots, exempli- by Linko¨pings University [23] includes using a controller

ﬁed by the ”Cyberrunner” robot developed at ETH Zurich, to navigate through a real-world maze using a model-based

demonstrate the capability to perform speciﬁc tasks [3]. This method, two servos to control the angles of the board, and a

particular robot is engineered to achieve rapid learning within camera to track the position of a ball. In contrast, the authors

physical environments, navigating and solving labyrinth games of [24] make no direct changes to the game and want to

while computing the optimal path from the start point to solve it in its original condition. Approaches employing RL

the destination [2]. Temporal Difference learning (TD) is a to solve the labyrinth include [25] and [26]. However, the

model-free RL method commonly employed in temporal path validity of these approaches was only veriﬁed in simulation.

detection, where actions are prioritized and rewarded accord- ETH Zurich’s latest work also implemented this approach in a

ingly based on their assigned importance [5]. TD learning real-life tilt maze without modifying the game itself and used

involves bootstrapping the current value function estimate, the original labyrinth with original dynamics [3]. They exploit

enabling iterative updates without relying on a predeﬁned recent advances in model-based RL and its ability to make

model of the environment [6]. The SARSA trace algorithm informed decisions about potentially successful behaviors by

is a TD algorithm that is chosen to perform path planning planning for the future [3].

in a dynamic environment, and this algorithm is a model-

free algorithm, meaning it does not have any information

about its environment [1]. Apart from that, SARSA is an on-

III. METHOD AND ALGORITHM

The SARSA with eligibility traces algorithm was used to

policy learning algorithm. It has proven its contribution in estimate and update the Q-values of state-action pairs in the

many ﬁelds, for example, in swarm intelligence reinforcement given maze environment, which was modeled in Rai. Utilizing

learning algorithms where multiple agents communicate with eligibility traces along with temporal-difference learning, the

each other and exchange information [4].

algorithm continuously improves its understanding of the

expected cumulative rewards associated with carrying out

particular actions in speciﬁed states. The objective of the

SARSA algorithm is to learn more about Q-values, which

represent the expected total reward of fulﬁlling a speciﬁc

condition. Temporal-difference learning involves modifying

these Q-values using the temporal difference error, which is

the difference between the current estimate and a new estimate

that accounts for both the estimated future rewards and the

observed immediate reward. The SARSA update rule, which

modiﬁes the Q-values in accordance with the learning rate

and the temporal difference error, is the basis of this process.

Speciﬁcally, to update the Q-value for a state-action pair, the

multiplication of the temporal difference error and the learning

rate is added. The equation is as follows [27]:

B. Robotic Manipulation

Finding the ideal way to control high-dimensional, contin-

uous systems like humanoid robots has been nearly impos-

sible due to the ”Optimal Control problem” being incredibly

complex [12]. However, recent advancements in motion op-

timization, including algorithms like CHOMP [13], STOMP

[14], iTOMP [15], TrajOpt [16], KOMO [9], and RIEMO [17],

have signiﬁcantly tackled this challenge. These methods have

progressively reﬁned motion optimization, making it a key tool

for generating motions in high-dimensional systems.

In this work, the RAI interface is used to create simulated

environments, including a physics-based simulation environ-

ment as well as a particular solution to cope with the com-

plexity of the k-Order Motion Optimization (KOMO) problem

[7]. KOMO means k-order Markov optimization. KOMO is a

way to formulate path optimization problems [7]. KOMO can

be used in various applications such as motion prediction[8],

control [9], and decision-making [10], [11]. However, in this

project, KOMO is used to calculate the joint states of the Panda

robots using inverse kinematics.

Q<sub>t+1</sub>(s, a) = Q (s, a) + αδ e (s, a)

t

t

t

for all s,a where,

δ<sub>t</sub> = r<sub>t+1</sub>γQ (s + 1, a + 1) − Q (s , a ),

t

t

t

t

SARSA incorporates eligibility traces to improve credit

assignments and make learning easier in situations where



<a name="br3"></a> 

rewards are delayed. These traces record the eligibility of state- building blocks, the collision of the ball with which the tile

action pairs for updates. A state-action pair’s eligibility trace is gives the position of the ball is checked. It is also possible

ﬁrst increased and then progressively decays over time when to get the exact position of the ball and then discretize it,

it is visited at each time step. This decay is controlled by but using the collision feature returned the most accurate

a discount factor, which means that in the learning process, measurements. The velocity is found by taking the difference

the most recent state-action pairs are given more weight. The between the current position and the previous position and

eligibility trace update equation is as follows [27]:

taking its unit vector.

(

γe<sub>t−1</sub>(s, a) + 1, if s = s<sub>t</sub> and a = a<sub>t</sub>

e<sub>t</sub> =

γe<sub>t−1</sub>(s, a),

otherwise

In our model, the states were chosen as the position and

the velocity of the ball in the given maze. The actions are

the movements of the four 6DoF Franka Robots. Using these

state-action pairs, the model was trained using the pseudo-code

given below [27].

Initialize Q(s, a) arbitrarily and e(s, a) = 0 for all s, a

repeat{for each episode}

Initialize s

repeat{for each step of episode}

Choose a from s using policy derived from Q

Take action a, observe r, s′

Fig. 1. The markers that are located near the grasp location.

The reward function can be found below. The reward path

is found by hand when designing the maze.

δ ← r + γQ(s , a ) Q(s, a)

′

<sup>′</sup> −

e(s, a) ← e(s, a) + 1

for all s, a do

TABLE I

REWARD FUNCTION

Q(s, a) ← Q(s, a) + αδe(s, a)

e(s, a) ← γλe(s, a)

end for

Condition

Won

Lost (Fall into the hole)

On reward path

Visited state

Reward

10

-10

s ← s

′

until s is terminal

until convergence

1

-0.4

-0.25

Living Cost

IV. EXPERIMENTAL SETUP

Rai is utilized to simulate the environment. There are six

different maps, and each map is harder to solve than the

previous one. The maps are generated from a 15x15 array

where 0 is free space, 1 is a wall, 2 is a reward path, 3 is a

hole, 4 is the starting position, and 5 is the goal point. The

generateMap function creates the map by using unit cubes,

where the red blocks are the reward path, green blocks are

starting positions, and blue blocks are the goal positions. The

ball has a unit length diameter. The map is built on top of the

tray named maze base. Four 6DoF Panda robots are placed

at each edge of the maze base, where each arm moves up,

down, and middle, tilting the maze at four different angles.

There are 12 markers (3 for each arm), which are placed at the

same level, above, and below relative to the contact point. The

KOMO (K-Order Markov Optimizer) is used to calculate the

four joint state conﬁgurations where the robotic arms tilt the

maze forward, backward, left, and right using those markers

[6].

The Simulation framework in Rai is quite useful in appli-

cations of RL since it can work at higher speeds than real-

time and the environment can be reset without destroying the

environment, different from the BotOP framework [7]. We

trained and tested our agent in the Rai environment.

TABLE II

S<sub>IMULATION</sub> P<sub>ARAMETERS</sub>

Parameter Name

Learning Rate

Value

0\.1

Epsilon

0\.8

Reward Margin

Minimum Reward

Exploration Decay

Exploitation Decay

10

Living Cost\*Reward Margin\*Reward Path Length

0\.999

0\.999

The use of exploitation decay was critical in the perfor-

After thorough experimentation, it was found that using the mance of the agent. What it does is decrease the minimum

state’s current position and velocity returns the best perfor- reward margin in each epoch. By doing so, in a higher epoch,

mance. The current position is found by using the collision the agent does not diverge from the ideal trajectory and goes

feature in Rai. Since the maze is generated using unit-size for the suboptimal paths.



<a name="br4"></a> 

V. RESULTS

is seen that moving the ball forward makes the ball hit the wall

and reach the end goal. With higher epochs reaching 2000, the

agent would ﬁnd similar trajectories to the human.

The source code <sup>1</sup> and demo video<sup>2</sup> for our experiments are

available. <sup>1 2</sup>

We ran our algorithm and simulated the agent in six different

maps. Furthermore, we compared the agent’s performance

with human performance. To assess the human benchmark, we

gave ﬁve different people to give it a try, and after repeating

each map ﬁve times, we took the average result. The robotic

arms are controlled using the keyboard.

As for performance metrics, we used three different plots:

episode reward vs. epoch, win rate vs. epoch, and Q-Table

representation. We compared the episode reward with the

human benchmark and tried to observe whether the agent’s

performance converged or even exceeded the human per-

formance. The win rate shows how many tries the agent

consistently completed the maze. It is expected that when the

difﬁculty of the mazes increases, the later the agent manages

to complete the maze. Last but not least, we represented the Q

table to get a more in-depth understanding of how the policy

selects the actions in each state. Each tile is divided into nine

cells since there are nine different velocity conﬁgurations. The

arrows represent the action taken in each cell. If the cell is

empty, it means that the agent hasn’t explored such a state.

The black tile in the plot represents the end goal.

Fig. 3. Map 1 performance metrics.

A. Map 1 - Agent vs Human

Fig. 4. Map 1 Q-Table.

B. Map 2 - Agent vs Human

Fig. 2. Map 1.

The ﬁrst map is fairly simple. It follows a single path where

the agent has to make a zig-zag motion to make the ball reach

the goal position. This map is trained for 500 epochs, and

it took about 1.5 hours to train. Due to the simplicity of the

map, the human performs slightly better than the trained agent.

But after 500 epochs, the agent consistently solves the maze.

When the Q-Table is analyzed, at ﬁrst glance, it may seem

some moves of the agent are counter-intuitive. This is due to

the environment being dependent on dynamic constraints. For

instance, near the goal position, the agent moves the tilt maze

forward when the goal is just below. When simulated in Rai, it

Fig. 5. Map 2.

<sup>1</sup>https://github.com/CankutBoraTuncer/Dynamical-Tilt-Maze-Solving-with-

4-6DoF-Franka-Robots-Using-SARSA-Q-Table

<sup>2https://youtu.be/y47Ss</sup><sub>5</sub>cLc0

The second map is again a simple map, slightly harder than

the ﬁrst map. Same with the previous map this map is trained



<a name="br5"></a> 

for 500 epochs, and it took about 1.5 hours to train. When following the walls, the agent has to move between the gaps.

compared with the human benchmark, the agent’s performance The agent is trained for 500 epochs and it took 2 hours to

converges with the human performance.

train. As the difﬁculty increases, the difference between the

agent’s performance and human performance decreases.

Fig. 6. Map 2 performance metrics.

Fig. 9. Map 3 performance metrics.

Fig. 7. Map 2 Q-Table.

Fig. 10. Map 3 Q-Table.

C. Map 3 - Agent vs Human

D. Map 4 - Agent vs Human

Fig. 8. Map 3.

Fig. 11. Map 4.

With the third map, there is a signiﬁcant increase in the

The fourth map has intricate gaps and passageways with

difﬁculty in terms of the dynamics of the map. Rather than different alternative routes. In this map, it is observed that



<a name="br6"></a> 

the shortest path is not always the quickest. The human actor

selects the shortest by following the red line. However, the

agent discovered that going around the narrow passage on top

and going from below is faster, thus performing better than

a human. This map is trained for 1500 epochs, and it took 4

hours to train.

Fig. 14. Map 5.

Fig. 12. Map 4 performance metrics.

Fig. 15. Map 5 performance metrics.

Fig. 13. Map 4 Q-Table.

E. Map 5 - Agent vs Human

What makes the ﬁfth map harder than the previous maps is

the pitfalls. The agent has little room to make errors, such as

falling into a pit, which results in a penalty and restart. The

human performance is expected to be better in this scenario as

the ideal path is dynamically achievable with trivial skills. But

the agent also performed similarly to the human as the action

taken in each state can be seen in the q table. This agent is

trained for 2000 epochs and it took 4 hours.

F. Map 6 - Agent vs Human

The last map has narrow passageways and traps, making

following the path quite difﬁcult. The agent exceeds the

human performance by a slight margin. Finding the right

combinatorics of actions while following the path was quite

challenging, even for our group members to execute. We

Fig. 16. Map 5 Q-Table.



<a name="br7"></a> 

map complexity is low, the human performs better than the

agent. However, in maps where dynamical manipulation is

crucial, the agent has the upper hand. It is seen that agents

can ﬁnd nontrivial alternative routes that end up with higher

returns than the shortest path.

The main limitation we encountered was the simulation

speed. In the previous version of our project, rather than using

the Simulation framework, we used BotOP. In BotOP, it was

not possible to increase the simulation speed. However, we

managed to increase the speed by 4 when using the Simulation

framework. Still, the training speed was not high enough to

train the agent for 20,000 or 30,000 epochs, which would take

approximately up to 2 or 3 days.

Fig. 17. Map 6.

VII. FUTURE WORK

Due to our choice of algorithm being SARSA Q-Table, the

states had to be discretized. Without needing any discretiza-

tion, if the states were continuous, the agent could ﬁnd slightly

more optimized paths. As for future work, rather than using

value function methods, deep RL algorithms such as PPO and

DDQN can be used. Moreover, in this project, we had the full

state information; however, if this project is to be applied in

a real-life environment, this is not trivial. We tried to gather

the state information from a top-down camera simulated in Rai

rather than using the collision feature. Due to time constraints,

we did not have time to test the camera. But the code is

available in our code.

managed to achieve such a result after many tries. The agent

is trained for almost 4000 epochs; we only displayed only a

signiﬁcant portion of it. It took 10 hours to train.

VIII. CONCLUSION

We have approached the problem of solving a dynamic tilt

maze by discretizing the environment as much as possible

in terms of position and velocity. Using SARSA with Q-

Table was a lightweight and high-performing algorithm that

we utilized in this project. Using an exploitation decay next

to an exploration decay improved the performance of the agent

\- in the previous interactions, we had a static minimum reward

threshold, and after a while agent diverged from the optimal

path. Rather than comparing the results as it is, we compared

them with the human performance and investigated whether

the trained agent can outperform the human agent. The results

indicate that the train agent performs similarly to the human

agent - even better in harder maps. The reason is that the

agent can exploit the dynamic environment and select the ideal

actions in each situation better than the human agent.

Fig. 18. Map 6 performance metrics.

ACKNOWLEDGMENT AND CONTRIBUTIONS

This project is presented in the CS449/549 Learning for

Robotics course in the Fall Semester of 2023. We are thankful

for the guidance of our course coordinator Asst. Prof. Ozgur

S. Oguz and teaching assistant Berk Cicek.

Cankut Bora Tuncer and Busenaz Kerimgil were responsible

for constructing the simulation environment, designing the

maps, actuating the panda robots, and adjusting the reward

function. Onur Alp and Osmanalp found the RL algorithm and

implemented it for our simulation. Kemal wrote the interface

Fig. 19. Map 6 Q-Table.

VI. DISCUSSION AND LIMITATIONS

When the agent performance is compared with human to control the Panda robots with keyboard inputs and the

performance in six different maps, it is found that when the function used to display the Q-table.



<a name="br8"></a> 

REFERENCES

[23] E. Frid and F. Nilsson, ‘Path Following Using Gain Scheduled LQR

Control: with applications to a labyrinth game’, Dissertation, 2020.

[1] N. Ravishankar and M. Vijayakumar, ”Reinforcement Learning Al- [24] N. A. Andersen, O. Ravn, and A. T. Sørensen, “Real-time vision-based

gorithms: Survey and Classiﬁcation,” Indian Journal of Science and

Technology, vol. 10, no. 1, 2017. DOI: 10.17485/ijst/2017/v10i1/109385.

[2] Q. Zhang, M. Li, X. Wang, and Y. Zhang, ”Reinforcement Learning

control of servomechanical systems,” in Experimental Robotics II: The

2nd International Symposium, Toulouse, France, June 25–27, 1991, pp.

388–402, Springer, 1993.

in Robot Path Optimization,” Journal of Software, vol. 7, pp. 657-662, [25] L. Abdenebaoui, E. A. Kirchner, Y. Kassahun, and F. Kirchner, “A

2012\. DOI: 10.4304/jsw.7.3.657-662.

connectionist architecture for learning to play a simulated brio labyrinth

game,” in KI 2007: Advances in Artiﬁcial Intelligence, Springer, 2007,

pp. 427–430.

[3] T. Bi and R. D’Andrea, “Sample-Efﬁcient Learning to Solve

a

Real-World Labyrinth Game Using Data-Augmented Model-

Based Reinforcement Learning,” arXiv.org, 2023. Available: [26] J. H. Metzen, E. Kirchner, L. Abdenebaoui, and F. Kirchner, “The

https://arxiv.org/abs/2312.09906.

Brio labyrinth game - A testbed for reinforcement learning and for

studies on sensorimotor learning,” in Multidisciplinary Symposium on

Reinforcement Learning, 2009.

[4] H. Iima and Y. Kuroe, ”Swarm reinforcement learning algorithms based

on Sarsa method,” in Proceedings of the SICE Annual Conference

(SICE), 2008, pp. 2045-2049. DOI: 10.1109/SICE.2008.4654998.

[5] D. S. Nair and P Supriya, “Comparison of Temporal Difference Learn-

ing Algorithm and Dijkstra’s Algorithm for Robotic Path Planning,”

Jun. 2018, doi: https://doi.org/10.1109/iccons.2018.8663020. Available:

https://ieeexplore.ieee.org/document/8663020.

[27] 7.5

Sarsa(),”

Incompleteideas.net,

2023\.

Available:

http://incompleteideas.net/book/ebook/node77.html. [Accessed: Dec.

20, 2023]

[6] G.

Gammon,” ICGA Journal, vol. 18, no. 2, pp. 88–88, Jun.

1995, doi: https://doi.org/10.3233/icg-1995-18207. Available:

https://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf

Tesauro,

“Temporal

Difference

Learning

and

TD-

[7] M. Toussaint, “Newton methods for k-order Markov Constrained Motion

Problems,” arXiv.org, 2014. Available: https://arxiv.org/abs/1407.0414.

[Accessed: Jan. 03, 2024]

[8] A. M. Lehrmann, P. V. Gehler, S. Nowozin, Efﬁcient nonlinear markov

models for human motion, in: 2014 IEEE Conference on Computer

Vision and Pattern Recognition, IEEE, Columbus, OH, USA, 2014, pp.

1314–1321. doi:10.1109/CVPR.2014. 171.

[9] D. Driess, O. Oguz, and M. Toussaint, ”Hierarchical Task and Motion

Planning using Logic-Geometric Programming (HLGP),” 2019.

[10] Y. Feng and B. Wu, ”Robot path planning algorithm based on partially

observable Markov,” Journal of Intelligent Fuzzy Systems, pp. 1-6,

2017\. DOI: 10.3233/JIFS-169337.

[11] Y. Oh, M. Toussaint, and J. Mainprice, “A System for Traded Control

Teleoperation of Manipulation Tasks using Intent Prediction from Hand

Gestures,” arXiv.org, 2021. https://arxiv.org/abs/2107.01829

[12] A. Doerr, N. Ratliff, J. Bohg, M. Toussaint, and S. Schaal, ”Direct Loss

Minimization Inverse Optimal Control,” in Proceedings of Robotics:

Science and Systems (RSS), 2015. DOI: 10.15607/RSS.2015.XI.013.

[13] N. Ratliff, M. Zucker, J. Andrew Bagnell, and S. Srinivasa, “CHOMP:

Gradient Optimization Techniques for Efﬁcient Motion Planning -

PREPRINT.”

[14] Mrinal Kalakrishnan, Sachin Chitta, E. A. Theodorou, P. Pa´stor, and

S. Schaal, “STOMP: Stochastic trajectory optimization for motion

planning,” May 2011, doi: https://doi.org/10.1109/icra.2011.5980280.

[15] C. Park, J. Pan, and D. Manocha, ”ITOMP: Incremental Trajectory

Optimization for Real-time Replanning in Dynamic Environments,”

in Proceedings of the 22nd International Conference on Automated

Planning and Scheduling (ICAPS), 2012.

[16] J. Schulman, J. Ho, A. Lee, I. Awwal, H. Bradlow, and P. Abbeel,

“Finding Locally Optimal, Collision-Free Trajectories with Sequential

Convex Optimization.” Accessed: Jan. 03, 2024. [Online]. Available:

https://www.roboticsproceedings.org/rss09/p31.pdf

[17] N. Ratliff, J. Nvidia, D. Kappler, S. Nvidia, and D. Nvidia, “Rieman-

nian Motion Policies.” Accessed: Jan. 03, 2024. [Online]. Available:

https://arxiv.org/pdf/1801.02854.pdf

[18] D. Silver et al., “Mastering the game of Go without human

knowledge,” Nature, vol. 550, no. 7676, pp. 354–359, Oct.

2017,

doi:

https://doi.org/10.1038/nature24270.

Available:

https://www.nature.com/articles/nature24270.

[19] A. Juliani et al., “Unity: A General Platform for Intelligent Agents,”

arXiv.org, 2018. Available: https://arxiv.org/abs/1809.02627. [Accessed:

Jan. 03, 2024]

[20] A. Devo, G. Costante, and P. Valigi, “Deep reinforcement learning for

instruction following visual navigation in 3d maze-like environments,”

IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1175–1182,

2020\.

[21] E. Parisotto and R. Salakhutdinov, “Neural map: Structured memory for

deep reinforcement learning,” arXiv preprint arXiv:1702.08360, 2017.

[22] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun, and ¨

D. Scaramuzza, “Champion-level drone racing using deep reinforcement

learning,” Nature, vol. 620, no. 7976, pp. 982–987, 2023.


