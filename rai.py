import robotic as ry
import random 
import numpy as np
import heapq
import matplotlib.pyplot as plt

class RaiEnv():

    def __init__(self, maze_dif = 1):
        random.seed(42)
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

        self.C = ry.Config()
        
        self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'), namePrefix="l_")
        self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'), namePrefix="r_")
        self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'), namePrefix="f_")
        self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'), namePrefix="b_")
    
        self.C.frame("l_l_panda_link0_0").setColor([0, 0, 1]) 
        self.C.frame("r_l_panda_link0_0").setColor([0, 1, 0])
        self.C.frame("f_l_panda_link0_0").setColor([1, 1, 0])
        self.C.frame("b_l_panda_link0_0").setColor([1, 0, 0])

        self.C.delFrame("l_table")
        self.C.delFrame("r_table")
        self.C.delFrame("f_table")
        self.C.delFrame("b_table")

        self.C.delFrame("l_cameraTop")
        self.C.delFrame("r_cameraTop")
        self.C.delFrame("f_cameraTop")
        self.C.delFrame("b_cameraTop")

        # Initialize states
        self.maze_dif = maze_dif
        self.unit_size = 0.075
        self.direction = [[0,1], [0,-1], [-1, 0], [1,0]]
        self.tile_color = [0.831372549, 0.733333, 0.4941176]
        self.wall_color = [0.5098039216, 0.2862745098, 0.0431372549] #[1,.1*3,1-.1*3]
        self.end_color = [68/255, 51/255, 153/255]
        self.reward_color = [1, 0, 0]#[153/255, 51/255, 51/255]
        self.start_color = [116/255, 153/255, 51/255]
        self.base_color = [0.5098039216, 0.2862745098, 0.0431372549] #[0,0,0]
        self.top_color = [1, 1, 1, 0.1]
        self.ball_color = [0, 1, 1]
        self.visited_color = [0.7, 0.7, 0.2]
        
        self.S = None
        self.rewards = []
        self.start_x = 0
        self.start_y = 0
        self.base_x = 0
        self.base_y = 0
        self.base_z = 0.4
        self.over = False
        
        maze_data = RaiEnv.create_maze(self.maze_dif)
        self.maze = maze_data[0]
        self.start = maze_data[1]
        self.goal_position = maze_data[2]
        self.maze_dim_x = len(self.maze)
        self.maze_dim_y = len(self.maze[0])

        self.base_w = (self.maze_dim_x) * self.unit_size 
        self.base_h = (self.maze_dim_y) * self.unit_size
        self.base_l = self.unit_size/2

        self.C.frame("l_l_panda_base") .setPosition([0, self.base_w/2 + 0.5, .0 ])  .setQuaternion([0.7038453156522361, 0.0, 0.0, -0.7103532724176078])
        self.C.frame("r_l_panda_base") .setPosition([0, -self.base_w/2 - 0.5, .0 ]) .setQuaternion([0.7038453156522361, 0.0, 0.0, 0.7103532724176078])
        self.C.frame("f_l_panda_base") .setPosition([self.base_w/2 + 0.5, 0, .0 ])  .setQuaternion([0.0007963267107332633, 0.0, 0.0, 0.9999996829318346])
        self.C.frame("b_l_panda_base") .setPosition([-self.base_w/2 - 0.5, 0, .0 ]) .setQuaternion([1.0, 0.0, 0.0, 0.0])

        self.start_x = -int(len(self.maze) / 2) * self.unit_size
        self.start_y = -int(len(self.maze[0]) / 2) * self.unit_size

        self.init_position = self.start
        self.current_position = self.start
        self.previous_position = self.start
        self.velocity = [0, 0]

        self.reward_path = self.maze_solver()
        self.last_visited_reward = 0
    
        self.generateMap()

        self.init_ball = self.indexToCoordinate(self.start[0], self.start[1])
        self.ball_position = self.init_ball
        self.ball = self.addBall(self.init_ball)

        self.motions = self.calculateMotions()
        self.init_arm_state = self.motions[0][-1]
        self.init_frame_state = self.C.getFrameState()

        self.C.setJointState([.02], ['l_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['r_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['f_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['b_l_panda_finger_joint1']) #only cosmetics

        self.goal = self.goal_position
        
        self.Q = dict()
        self.visited = set()
        self.visited.add(tuple(self.current_position))
                
        self.allowed_states = np.asarray(np.where(self.maze == 0)).T.tolist()

        self.empty_points = []
        for coord in np.argwhere(np.array(self.maze) == 0).tolist():
            if(coord[0] % 3 == 1 and coord[1] % 3 == 1):
                self.empty_points.append(tuple(coord))
        self.empty_points.append((4, 1))
        
        self.action_map = {
                           0: [0, 1],
                           1: [0,-1],
                           2: [1, 0],
                           3: [-1,0],
                           }
        
        self.directions = {
                           0: '→',
                           1: '←',
                           2: '↓ ',
                           3: '↑'}
        
        self.reward_path = self.maze_solver()
        self.initSim()

        self.f = self.C.addFrame("cameraTop", parent="maze_base", args= "shape: marker, size: [.1], focalLength: 4, width: 600, height: 600, zRange: [.5, 100]")

        self.f.setRelativePosition([0, 0, 10])
        self.f.setRelativeQuaternion([0.0007963242222135884, 0.9999965579344531, 1.9908147030663304e-06, -0.002499996603164559])
        
        self.cam_pos = [0,0,0]
        self.C.view_setCamera(self.f)
        self.view()


    def printCam(self):
        self.C.view_setCamera(self.f)
        rgb = self.C.view_getRgb()
        depth = self.C.view_getDepth()

        self.ax1.clear()
        self.ax1.imshow(rgb)
        self.ax2.clear()
        self.ax2.imshow(depth)

        plt.draw()
        plt.pause(0.001)
        
        b = rgb[:,:,2]
        g = rgb[:,:,1]
        re = rgb[:,:,0]
        for r in range(0, len(b)):
            for c in range(0, len(b[0])):
                if(b[r,c] > 0.80*255 and g[r,c] > 0.80*255 and re[r,c] < 0.2*255):
                    self.cam_pos = [c, r, self.ball_position[2]]
                    return
        print("Not found!")

    def addBall(self, pos):
        return self.C.addFrame(name = "ball") \
                .setShape(ry.ST.sphere, size = [self.unit_size/3]) \
                .setPosition(pos) \
                .setColor(self.ball_color) \
                .setContact(True)\
                .setMass(0.001)
        
    def IK(self, gripper, target):
        komo = ry.KOMO(self.C, 100, 1, 0, False)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.poseDiff, [gripper[0], target[0]], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.poseDiff, [gripper[1], target[1]], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.poseDiff, [gripper[2], target[2]], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.poseDiff, [gripper[3], target[3]], ry.OT.eq, [1e1])
        ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
        return [komo.getPath(), ret]

    def calculateMotions(self):
        
        a = ["l_l_gripper","r_l_gripper", "f_l_gripper", "b_l_gripper"]
        neutral, _ = self.IK(a, ["marker_l", "marker_r", "marker_f", "marker_b"])
        
        b1 = ["marker_l", "marker_r", "marker_up", "marker_down"]
        b2 = ["marker_l", "marker_r", "marker_up_reset", "marker_down_reset"]
        b3 = ["marker_left", "marker_right", "marker_f", "marker_b"]
        b4 = ["marker_left_reset", "marker_right_reset", "marker_f", "marker_b"]
        #up down left right
        b = [b1, b2, b3, b4]

        motions = [neutral] # neutral, up, down, left, right

        for i in range(0, 4):
            qf, _ = self.IK(a, b[i])
            motions.append(qf)
            self.C.setJointState(neutral[-1])
            self.view()
            
        return motions

    def reset(self, start_cell):
            
        self.current_position = start_cell
        self.previous_position = start_cell

        self.init_position = start_cell
        self.ball_position = self.indexToCoordinate(start_cell[0], start_cell[1])

        self.C.frame("maze_base").setPosition([self.base_x, self.base_y, self.base_z]).setQuaternion([1, 0, 0, 0])
        self.C.setJointState(self.init_arm_state)

        self.ball.setPosition(self.ball_position)
        self.reward_path = self.maze_solver()
        
        self.C.setJointState([.02], ['l_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['r_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['f_l_panda_finger_joint1']) #only cosmetics
        self.C.setJointState([.02], ['b_l_panda_finger_joint1']) #only cosmetics

        self.visited = set()
        self.visited.add(tuple(self.current_position))
        self.over = False
        
        self.S.setState(self.C.getFrameState())
        return self.state()

    def state_update(self, action):
        isgameon = True
        self.over = False
        if(action != 4):
            move = self.action_map[action]
        else:
            move = [0,0]
            
        self.moveBall(move)
        self.ball_position = self.ball.getPosition()
        cur_state = self.getPositionIndex()
        next_position = np.asarray([cur_state[1], cur_state[2]])
        self.previous_position = self.current_position
        self.current_position = next_position
        self.velocity = self.normalizeVelocity(self.current_position - self.previous_position)

        if cur_state[0] == 1:
            reward = 10
            isgameon = False
            self.over = True
            return [self.state(), reward, isgameon, 1]
        
        elif self.ball_position[2] < 0.48:
            reward = -10
            isgameon = False
            self.over = True
            return [self.state(), reward, isgameon, 0]
        
        else:
            if tuple(self.current_position) in self.visited:
                reward = -0.4
            elif tuple(self.current_position) in self.reward_path:
                i = self.reward_path.index(tuple(self.current_position))
                if i > self.last_visited_reward:
                    self.last_visited_reward  = i
                    reward = 1
                else:
                    reward = -0.25
            else:
                reward = -0.25

        self.visited.add(tuple(self.current_position))
        return [self.state(), reward, isgameon, 0]

    def state(self):
        return [*self.current_position, *self.velocity]
    
    def view(self):
        self.C.view()

    def initSim(self):
        self.S = ry.Simulation(self.C, ry.SimulationEngine.physx, verbose=0)

    def normalizeVelocity(self, velocity):
        max_value = max(velocity)
        min_value = min(velocity)
        normalized_vector = [0, 0]
        for i in range(len(velocity)):
            if (max_value - min_value != 0):
                normalized_vector[i] = (velocity[i] - min_value) / (max_value - min_value) * 2 - 1
        return normalized_vector
    
    def moveToVelocity(self, move):
        vx = 1 * self.unit_size * move[0]
        vy = 1 * self.unit_size * move[1]
        return [vx, vy]
    
    def moveToMotion(self, move):
        #up down left right
        if(move[0] == 0 and move[1] == 1):
            #print("Move down")
            return self.motions[4]
        elif(move[0] == 0 and move[1] == -1):
            #print("Move up")
            return self.motions[3]
        elif(move[0] == 1 and move[1] == 0):
            #print("Move left")
            return self.motions[2]
        elif(move[0] == -1 and move[1] == 0):
            #print("Move right")
            return self.motions[1]
        else:
            #print("Not Found Move")
            return self.motions[0]

    def moveBall(self, move):
        m = self.moveToMotion(move)
        tau = .01
        for i in range(100):  
            self.S.step(m[i], tau,  ry.ControlMode.position)
            self.C.view()
        
    def getPositionIndex(self):
        self.C.computeCollisions()
        collision = self.C.getCollisions()
        for i in range(1, len(collision)):
            if(collision[i][0] == "ball"):
                a = collision[i][2]
                pos = collision[i][1].split("_")
                if(pos[0] == "t" and float(a) < 0.001):
                    return [0, int(pos[1]), int(pos[2])]
                elif(pos[0] == "e" and float(a) < 0.001):
                    print(collision[i])
                    return [1, int(pos[1]), int(pos[2])]

        return [0, self.current_position[0], self.current_position[1]]
    
    def indexToCoordinate(self, i0, i1):
        x = i0 * self.unit_size + self.start_x
        y = i1 * self.unit_size + self.start_y
        return [x , y, self.base_l + self.unit_size*3 + self.base_z ]

    def updateMap(self):
        self.maze[self.previous_position[0], self.previous_position[1]] = 0
        self.maze[self.current_position[0], self.current_position[1]] = 2

    def addTile(self, name, pos, color, parent):
        len = self.unit_size
        size = [len, len, 0.25*self.unit_size*8, len/10000]
        self.C.addFrame(name = name,parent=parent) \
            .setShape(ry.ST.ssBox, size = size) \
            .setRelativePosition(pos) \
            .setColor(color) \
            .setContact(True)\
            .setMass(0.0001)
        
    def addWall(self, name, pos, color,parent):
        len = self.unit_size
        size = [len, len, 0.25*(self.unit_size)*16, len/10000]
        self.C.addFrame(name = name,parent=parent) \
            .setShape(ry.ST.ssBox, size = size) \
            .setRelativePosition(pos) \
            .setColor(color) \
            .setContact(True)\
            .setMass(0.0001)
    
    def addBase(self, name, pos, color):
        size = [self.base_w, self.base_h, self.base_l, .02]
        self.maze_base = self.C.addFrame(name = name) \
            .setShape(ry.ST.ssBox, size = size) \
            .setPosition(pos) \
            .setColor(color) \
            .setQuaternion([1.0, 0.0, 0.0, 0.0]) \
            .setMass(0.001) \
            .setContact(True)
        
    def addNewBase(self):
        color = [0.5098039216, 0.2862745098, 0.0431372549]
        s = self.unit_size*16
        name = "maze_base"
        self.C.addFrame(name = name) \
            .setShape(ry.ST.ssBox, size = [1*s, 1*s, 0.05*s, 0.001]) \
            .setPosition([0 ,0, self.base_z]) \
            .setContact(True)\
            .setColor([0,0,0])\
            .setMass(0.001) \
            .setContact(True)

        self.C.addFrame(name = "w_base_left", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.05*s, 1*s, 0.25*s, 0.001]) \
            .setRelativePosition([0.5*s ,0, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_base_right", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.05*s, 1*s, 0.25*s, 0.001]) \
            .setRelativePosition([-0.5*s, 0, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_base_front", parent=name) \
            .setShape(ry.ST.ssBox, size = [1*s, 0.05*s, 0.25*s, 0.001]) \
            .setRelativePosition([0 ,0.5*s, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_base_back", parent=name) \
            .setShape(ry.ST.ssBox, size = [1*s, 0.05*s, 0.25*s, 0.001]) \
            .setRelativePosition([0 ,-0.5*s, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_handle_front", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.15*s, 0.2*s, self.unit_size/2, 0.001]) \
            .setRelativePosition([0.575*s,0, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_handle_back", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.15*s, 0.2*s, self.unit_size/2, 0.001]) \
            .setRelativePosition([-0.575*s ,0, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_handle_left", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.2*s, 0.15*s, self.unit_size/2, 0.001]) \
            .setRelativePosition([0 ,0.575*s, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)

        self.C.addFrame(name = "w_handle_right", parent=name) \
            .setShape(ry.ST.ssBox, size = [0.2*s, 0.15*s, self.unit_size/2, 0.001]) \
            .setRelativePosition([0 ,-0.575*s, 0.125*s]) \
            .setContact(True)\
            .setColor(self.tile_color)\
            .setMass(0.001)\
            .setContact(True)
        
    def addMarker(self):
        m = 0.03
        j = 0.045
        #  Blue     - LEFT  ARM
        #  Green    - RIGHT ARM
        #  Yellow   - FRONT ARM
        #  Red      - BACK  ARM
        self.C.addFrame(name = "marker_f", parent="w_handle_front") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([j, 0, 0]) \
            .setQuaternion([0.7038453156522361, 0.0, 0.7103532724176078, 0.0])
        self.C.addFrame(name = "marker_up", parent="w_handle_front") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([j, 0, m]) \
            .setQuaternion([0.7038453156522361, 0.0, 0.7103532724176078, 0.0])
        self.C.addFrame(name = "marker_up_reset", parent="w_handle_front") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([j, 0, -m]) \
            .setQuaternion([0.7038453156522361, 0.0, 0.7103532724176078, 0.0])
        #----------------------------------------------------------------------#
        self.C.addFrame(name = "marker_b", parent="w_handle_back") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([-j, 0, 0]) \
            .setQuaternion([0.7038453156522361, 0.0, -0.7103532724176078, 0.0]) 
        self.C.addFrame(name = "marker_down", parent="w_handle_back") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([-j, 0, -m]) \
            .setQuaternion([0.7038453156522361, 0.0, -0.7103532724176078, 0.0])        
        self.C.addFrame(name = "marker_down_reset", parent="w_handle_back") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([-j, 0, m]) \
            .setQuaternion([0.7038453156522361, 0.0, -0.7103532724176078, 0.0])  
        #----------------------------------------------------------------------#
        self.C.addFrame(name = "marker_l", parent="w_handle_left") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, j, 0]) \
            .setQuaternion([0.5053980585291337, -0.4999708601149831, -0.4999708601149831, -0.4946019414708662])
        self.C.addFrame(name = "marker_left", parent="w_handle_left") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, j, m]) \
            .setQuaternion([0.5053980585291337, -0.4999708601149831, -0.4999708601149831, -0.4946019414708662])
        self.C.addFrame(name = "marker_left_reset", parent="w_handle_left") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, j, -m]) \
            .setQuaternion([0.5053980585291337, -0.4999708601149831, -0.4999708601149831, -0.4946019414708662])
        #----------------------------------------------------------------------#
        self.C.addFrame(name = "marker_r", parent="w_handle_right") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, -j, 0]) \
            .setQuaternion([0.5053980585291337, 0.4999708601149831, 0.4999708601149831, -0.4946019414708662])
        self.C.addFrame(name = "marker_right", parent="w_handle_right") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, -j, -m]) \
            .setQuaternion([0.5053980585291337, 0.4999708601149831, 0.4999708601149831, -0.4946019414708662])    
        self.C.addFrame(name = "marker_right_reset", parent="w_handle_right") \
            .setShape(ry.ST.marker, size=[.1]) \
            .setRelativePosition([0, -j, m]) \
            .setQuaternion([0.5053980585291337, 0.4999708601149831, 0.4999708601149831, -0.4946019414708662])    
        
    def generateMap(self):
        self.addNewBase()
        self.addMarker()
        
        for r in range(0,len(self.maze)):
            for c in range(0,len(self.maze[r])):
                if(self.maze[r][c] == 0):
                    name = "t" + "_" + str(r) + "_" + str(c)
                    pos = [self.start_x  + self.unit_size*r, self.start_y + self.unit_size*c, 0.25*self.unit_size*4]
                    self.addTile(name, pos, self.tile_color,"maze_base")   
                
                elif(self.maze[r][c] == 1):
                    name = "w" + "_" + str(r) + "_" + str(c)
                    pos = [self.start_x  + self.unit_size*r, self.start_y + self.unit_size*c, 0.25*(self.unit_size)*8]
                    self.addWall(name, pos, self.wall_color,"maze_base")

                elif(self.maze[r][c] == 2):
                    name = "t" + "_" + str(r) + "_" + str(c)
                    pos = [self.start_x  + self.unit_size*r, self.start_y + self.unit_size*c, 0.25*self.unit_size*4]
                    self.addTile(name, pos, self.reward_color,"maze_base")  

                elif(self.maze[r][c] == 4):
                    name = "t" + "_" + str(r) + "_" + str(c)
                    pos = [self.start_x  + self.unit_size*r, self.start_y + self.unit_size*c, 0.25*self.unit_size*4]
                    self.addTile(name, pos, self.start_color,"maze_base")  

                elif(self.maze[r][c] == 5):
                    name = "e" + "_" + str(r) + "_" + str(c)
                    pos = [self.start_x  + self.unit_size*r, self.start_y + self.unit_size*c, 0.25*self.unit_size*4]
                    self.addTile(name, pos, self.end_color,"maze_base")  

    def maze_solver(self):
        step = 1
        start = (self.init_position[0], self.init_position[1])
        end = (self.goal_position[0], self.goal_position[1])
        rows, cols = len(self.maze), len(self.maze[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def is_valid(x, y):
            return 0 <= x < rows and 0 <= y < cols and (self.maze[x][y] == 0 or self.maze[x][y] == 2 or self.maze[x][y] == 4 or self.maze[x][y] == 5)

        visited = [[False] * cols for _ in range(rows)]
        distance = [[float('inf')] * cols for _ in range(rows)]
        prev = [[None] * cols for _ in range(rows)]  # To store the previous node in the path
        distance[start[0]][start[1]] = 0
        heap = [(0, start)]

        while heap:
            current_dist, (x, y) = heapq.heappop(heap)
            if (x, y) == end:
                break
            if visited[x][y]:
                continue
            visited[x][y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    new_dist = distance[x][y] + 1
                    if new_dist < distance[nx][ny]:
                        distance[nx][ny] = new_dist
                        prev[nx][ny] = (x, y)
                        heapq.heappush(heap, (new_dist, (nx, ny)))

        # Reconstruct the path
        path = []
        current = end
        i = 0
        while current is not None:
            if i % step == 0:
                path.append(current)
            current = prev[current[0]][current[1]]
            i += 1
        path.reverse()  # Reverse to get the path from start to end
        path = path[1:]
        return path
    
    @staticmethod
    def create_maze(dif):
        if(dif == 1):
            maze = [[1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,4 ,4 ,4 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [5 ,5 ,5 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]]
            
            start = [0, 13]
            end   = [14, 1]

        elif(dif == 2):
            maze = [[0 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,1 ,4 ,4 ],
                    [0 ,2 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,2 ,1 ,1 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,1 ,1 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,1 ,1 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,2 ,1 ,1 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,2 ,2 ,2 ,1 ,1 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,1 ,1 ,2 ,2 ,2 ,2 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,1 ,1 ,2 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,1 ,1 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,1 ,1 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [0 ,2 ,1 ,1 ,2 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,2 ,0 ],
                    [5 ,5 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,0 ]]
            
            start = [0, 13]
            end   = [14, 1]

        elif(dif == 3):
            maze = [[1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,4 ,4 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,2 ,2 ,2 ,2 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,1 ,1 ,2 ,0 ,1 ,2 ,0 ,1 ,1 ,0 ,2 ,1 ],
                    [1 ,0 ,0 ,0 ,0 ,2 ,0 ,1 ,2 ,0 ,0 ,0 ,0 ,2 ,1 ],
                    [1 ,2 ,2 ,2 ,2 ,2 ,0 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,5 ,5 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]]
            
            start = [0, 13]
            end   = [14, 1]

        elif(dif == 4):
            maze = [[1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,4 ,4 ,1 ],
                    [1 ,1 ,1 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,1 ],
                    [0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,2 ,2 ,2 ,2 ,1 ],
                    [0 ,0 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,0 ,2 ,1 ,0 ,0 ,1 ],
                    [0 ,0 ,1 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,0 ,0 ,1 ],
                    [0 ,0 ,1 ,1 ,1 ,2 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,0 ,1 ],
                    [0 ,2 ,2 ,2 ,2 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ],
                    [0 ,2 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ],
                    [1 ,2 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,0 ],
                    [0 ,2 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ],
                    [0 ,2 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ],
                    [5 ,5 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]]
            
            start = [0, 13]
            end   = [14, 1]

        elif(dif == 5):
            maze = [[1 ,2 ,2 ,2 ,2 ,1 ,1 ,2 ,2 ,2 ,2 ,1 ,1 ,4 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ],
                    [0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ],
                    [1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ],
                    [3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ],
                    [1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ],
                    [0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ],
                    [1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ],
                    [3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ,1 ,2 ,1 ,3 ,2 ,0 ],
                    [1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ,0 ,2 ,3 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ],
                    [1 ,5 ,1 ,1 ,2 ,2 ,2 ,2 ,1 ,1 ,2 ,2 ,2 ,2 ,1 ]]
            
            start = [0, 13]
            end   = [14, 1]

        elif(dif == 6):
            maze = [[1 ,0 ,1 ,1 ,0 ,0 ,0 ,1 ,0 ,1 ,1 ,1 ,1 ,4 ,1 ],
                    [0 ,0 ,0 ,1 ,0 ,1 ,0 ,1 ,0 ,2 ,2 ,2 ,2 ,2 ,1 ],
                    [1 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,2 ,1 ,0 ,1 ,1 ,1 ],
                    [1 ,2 ,2 ,2 ,2 ,1 ,0 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,0 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,0 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,0 ],
                    [1 ,2 ,1 ,1 ,2 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,1 ,2 ,1 ],
                    [0 ,2 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,5 ,1 ,1 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,2 ,1 ],
                    [1 ,2 ,1 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,1 ,2 ,1 ,1 ],
                    [0 ,2 ,2 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ],
                    [1 ,1 ,2 ,1 ,1 ,2 ,0 ,1 ,1 ,2 ,2 ,2 ,2 ,1 ,1 ],
                    [1 ,1 ,2 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ],
                    [1 ,1 ,2 ,2 ,2 ,2 ,1 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ]]
            
            start = [0, 13]
            end   = [7, 10]

        elif(dif == 7):
            maze = [[3 ,2 ,2 ,2 ,1 ,2 ,2 ,4 ,1 ,3 ,0 ,3 ,1 ,3 ,0 ],
                    [1 ,2 ,1 ,2 ,2 ,2 ,1 ,1 ,1 ,0 ,2 ,2 ,2 ,0 ,1 ],
                    [2 ,2 ,0 ,3 ,1 ,3 ,1 ,3 ,1 ,1 ,2 ,1 ,2 ,0 ,3 ],
                    [2 ,1 ,3 ,1 ,2 ,2 ,2 ,2 ,1 ,3 ,2 ,1 ,2 ,1 ,0 ],
                    [2 ,2 ,2 ,1 ,2 ,3 ,1 ,2 ,1 ,1 ,2 ,1 ,2 ,2 ,2 ],
                    [1 ,1 ,2 ,1 ,2 ,2 ,1 ,2 ,1 ,3 ,2 ,1 ,3 ,1 ,2 ],
                    [2 ,2 ,2 ,3 ,1 ,2 ,3 ,2 ,1 ,1 ,2 ,1 ,1 ,1 ,5 ],
                    [2 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,3 ,1 ,2 ,3 ,1 ,1 ,1 ],
                    [2 ,2 ,2 ,1 ,2 ,2 ,1 ,2 ,1 ,1 ,2 ,2 ,1 ,1 ,3 ],
                    [3 ,1 ,2 ,3 ,2 ,1, 3, 2, 2, 1, 1, 2, 2, 2, 0 ],
                    [2 ,2 ,2 ,1 ,2 ,2, 1, 1, 2, 1, 1, 1, 1, 2, 2 ],
                    [2 ,1 ,3 ,1 ,1 ,2, 1, 2, 2, 1, 1, 3, 1, 1, 2 ],
                    [2 ,2 ,1 ,1 ,2 ,2, 1, 2, 3, 1, 2, 2, 2, 3, 2 ],
                    [1 ,2 ,1 ,2 ,2 ,3, 1, 2, 2, 1, 2, 1, 2, 1, 2 ],
                    [3 ,2 ,2 ,2 ,1 ,1, 1, 3, 2, 2, 2, 3, 2, 2, 2 ]]
            
            start = [0, 7]
            end   = [6, 14]

        else:
            maze = [[1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,4 ,4 ,4 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [1 ,0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [0 ,2 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                    [5 ,5 ,5 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]]
            
            start = [0, 13]
            end   = [14, 1]
            
        return maze, start, end 
    
    def q(self, state):
        if type(state) == np.ndarray:
            state = tuple(state.flatten())
        return np.array([self.Q.get((state, action), 0.0) for action in list(self.action_map.keys())])

    def predict(self, state):
        q = self.q(state)

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
    
    def plot_policy_map(self, filename = 'figures/figure.png', ):
        _, ax = plt.subplots()
        ax.imshow(self.maze, 'Greys')
        offset = [0.35,-0.3]
        for r in range(0, len(self.maze)):
            for c in range(0, len(self.maze[r])):
                if(self.maze[r][c] != 1):
                    free_cell = (r, c)
                    action = self.predict(free_cell)
                    policy = self.directions[action]
                    ax.text(free_cell[1] + offset[1], free_cell[0] + offset[0], policy)

        ax = plt.gca()

        plt.xticks([], [])
        plt.yticks([], [])

        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.show()
