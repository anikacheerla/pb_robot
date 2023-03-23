#/usr/bin/env python
# -*- coding: utf-8 -*-

'''TPRM between two random configurations'''
import numpy as np
import random
import time
from scipy import spatial
import networkx as nx 
import numpy
from . import util
import pb_robot
from .plannerTypes import GoalType, ConstraintType
from .prm import TPRM

class TPRMPlanner(object):
    '''My implementation of cbirrt, for now without constraining,
    for toying around'''
    def __init__(self):
        ## Constants 
        self.TOTAL_TIME = 100.0
        self.SHORTEN_TIME = 2.0 # For video level, 4 seconds
        self.PSAMPLE = 0.2 
        self.QSTEP = 1
        self.tstart = None
        self.MAX_OBSTACLE_WIDTH = 4.0

        self.goal = None
        self.goal_type = None
        self.constraints = None
        self.grasp = None
        self.manip = None

        self.handles = []
    
    def collision_fn(self, q):
        return not self.manip.IsCollisionFree(q, obstacles=self.static_obstacles, self_collisions=True)

    def PlanToConfiguration(self, manip, start, goal_config, **kw_args):
        '''Plan from one joint location (start) to another (goal_config) with
        optional constraints. 
        @param manip Robot arm to plan with 
        @param start Joint pose to start from
        @param goal_config Joint pose to plan to
        @return OpenRAVE joint trajectory or None if plan failed'''


        # TODO: split up static and dynamic obstacles to pass into prm 
        path = self.TPRMPlanner(manip, start, goal_config, GoalType.JOINT, **kw_args)
        return util.generateTimedPath(path)

    def PlanToEndEffectorPose(self, manip, start, goal_pose, **kw_args):
        '''Plan from one joint location (start) to an end effector pose (goal_pose) with
        optional constraints. The end effector pose is converted to a point TSR
        and passed off to a higher planner
        @param manip Robot arm to plan with 
        @param start Joint pose to start from
        @param goal_pose End effector pose to plan to
        @return OpenRAVE joint trajectory or None if plan failed'''
        tsr_chain = util.CreateTSRFromPose(manip, goal_pose)
        return self.PlanToEndEffectorTSR(manip, start, [tsr_chain], **kw_args)

    def PlanToEndEffectorPoses(self, manip, start, goal_poses, **kw_args):
        '''Plan from one joint location (start) to any pose within a set of 
        end effector pose (goal_poses) with optional constraints. The end effector 
        poses are converted into a chain of point TSRs and passed off to a
        higher planner
        @param manip Robot arm to plan with 
        @param start Joint pose to start from
        @param goal_poses Set of end effector pose to plan to
        @return OpenRAVE joint trajectory or None if plan failed'''
        chains = []
        for i in range(len(goal_poses)):
            tsr_chain = util.CreateTSRFromPose(manip, goal_poses[i])
            chains.append(tsr_chain)
        return self.PlanToEndEffectorTSR(manip, start, chains, **kw_args)
    
    def PlanToEndEffectorTSR(self, manip, start, goal_tsr, constraints=None, **kw_args):
        '''Plan from one joint location (start) to any pose defined by a TSR
        on the end effector (goal_tsr) with optional constraints. There is also
        the option to "back in" to the goal position - instead of planning directly
        to the goal we plan to a pose slightly "backed up" from it and then execute
        the straight short motion from our backed up pose to our goal pose. This 
        has shown to make collision checking easier and work well when approaching
        objects. This process involves planning two paths, which are then stitched
        together. 
        @param manip Robot arm to plan with 
        @param start Joint pose to start from
        @param goal_poses Set of end effector pose to plan to
        @return OpenRAVE joint trajectory or None if plan failed'''

        # static and dynamic obstacles in **kw_args
        path = self.TPRMPlanner(manip, start, goal_tsr, GoalType.TSR_EE, constraints=constraints, **kw_args)
        return util.generateTimedPath(path)

    def distance_fn(self, q1, q2):
        tform_1 = self.manip.ComputeFK(q1)
        tform_2 = self.manip.ComputeFK(q2)
        e1 = np.array(tform_1)[:3, 3]
        e2 = np.array(tform_2)[:3, 3]
        return np.linalg.norm(np.subtract(e1, e2))


    def TPRMPlanner(self, manip, start, goalLocation, goal_type, roadmap = None, static_obstacles=None, dynamic_obstacles = None, 
                        constraints=None, grasp=None, num_samples=20):
        '''Given start and end goals, plan a path
        @param manip Arm to plan wit
        @param start start joint configuration
        @param goalLocation end goal, either configuration or TSR
        @param goal_type Configuration, TSR or Tool TSR
        @param constraints List of constraint functions given in form
                 [fn() ConstraintType]. List can have multiple of same type
        @param grasp The transform of the hand in the tool frame
        @param backupAmount Quantity, in meters to, to back up. If zero, we
                      do not back up in the goal position
        @param backupDIr Direction to back. Currently we only accept a single
                      direction ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        @param path Given as series of waypoints to be converted to OpenRave trajectory'''
 
        self.manip = manip
        self.goal = goalLocation
        self.goal_type = goal_type
        self.constraints = constraints
        self.grasp = grasp 
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.edge_collision_fn =self.checkEdgeCollision
        # self.distance_fn = lambda q1, q2: numpy.linalg.norm(numpy.subtract(q2, q1))        
        
         # distance between end effectors
        original_pose = manip.GetJointValues()
        if self.goal_type == GoalType.TSR_TOOL and self.grasp is None:
            raise ValueError("Planning calls that operate on the tool require the grasp is given")

        # If our goal is a joint configuration, that is our final goal.
        # However, if we have a TSR we sample a goal node


        if self.goal_type is not GoalType.JOINT:
            self.goal = self.addRootConfiguration()
            if not self.goal: # no configuration for grasp was found
                return None 
        
        if roadmap == None:
            samples = [numpy.array(start), numpy.array(self.goal)] + [self.randomConfig() for _ in range(num_samples)]
            print("Done sampling")
            roadmap = TPRM(self.manip, self.distance_fn, self.constrainedExtend, self.collision_fn, self.static_obstacles, self.dynamic_obstacles, samples=samples, edge_collision_fn=self.edge_collision_fn)
            print("Done generating roadmap")
        # Reset DOF Values
        manip.SetJointValues(original_pose) 

        # Return an trajectory
        return roadmap(numpy.array(start), numpy.array(self.goal))
    
    def makeRoadmap(self, manip, static_obstacles=None, dynamic_obstacles=None, num_samples=100, constraints=None, connect_distance=0.5):
        self.manip = manip
        self.constraints = constraints
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.edge_collision_fn =self.checkEdgeCollision

        samples = [self.randomConfig() for _ in range(num_samples)]
        print("Done sampling")
        roadmap = TPRM(self.manip, self.distance_fn, self.constrainedExtend, self.collision_fn, self.static_obstacles, self.dynamic_obstacles, samples=samples, edge_collision_fn=self.edge_collision_fn)
        print("Done generating roadmap")
        return roadmap

    def addRootConfiguration(self):
        '''Add goal configurations if the goal is not a single configuration. 
        We sample the end effector set and compute IK, only adding it if
        it satifies the relevant constraints
        @param T tree to add to
        @param T tree with new goal node'''
        if self.goal_type is GoalType.JOINT:
            raise TypeError("Cant add root configuration for non tsr-goal")

        searching = True
        while searching and (time.time() - self.tstart) < self.TOTAL_TIME:
            # Sample a TSR and then sample an EE pose from that TSR
            pose = util.SampleTSRForPose(self.goal)
           
            # Transform by grasp if needed
            if self.goal_type is GoalType.TSR_TOOL:
                ee_pose = numpy.dot(pose, self.grasp)
            else:
                ee_pose = pose

            # If there is an ee constraint, check it
            if self.evaluateConstraints(ConstraintType.GOAL_EE, pose=ee_pose):
                config = self.manip.ComputeIK(ee_pose)
                if config is not None:
                    # if there is a joint constraint, check it
                    if self.evaluateConstraints(ConstraintType.GOAL_JOINT, config=config, pose=ee_pose):
                        searching = False

        # Timed out, no root to be added
        if searching:
            return None

        # Found goal configuration, add it to the tree
        return config
    
    def checkEdgeCollision(self, q, q_parent, obstacles=None):
        '''Check if path from q_first to q_second is collision free
        @param q Joint configuration
        @return collisionFree (boolean) True if collision free''' 
        # Reject if end point is not collision free
        if not self.manip.IsCollisionFree(q, obstacles):
            return False
        cdist = util.cspaceLength([q_parent, q])
        count = int(cdist / 0.1) # Check every 0.1 distance (a little arbitrary)

        # linearly interpolate between that at some step size and check all those points
        interp = [numpy.linspace(q_parent[i], q[i], count+1).tolist() for i in range(len(q))]
        middle_qs = numpy.transpose(interp)[1:-1] # Remove given points
        return all((self.manip.IsCollisionFree(m, obstacles) for m in middle_qs)) 

    def evaluateConstraints(self, constraintType, **kw_args):
        '''Given the list of constraints and the constraint time, 
        find all constraints of that type and evaluate them
        @param constraintType type to check
        @param kw_args the parameters of the constraint function
        @return True if constraints are satified'''
        # If there are no constraints, then they are trivially satified
        if self.constraints is None:
            return True

        # Grab all the relevant constraints
        relevantConstraints = [self.constraints[i][0] for i in range(len(self.constraints)) if self.constraints[i][1] == constraintType]
        
        # If there are no relevant constraints, also satified
        if len(relevantConstraints) == 0:
            return True

        # Evaluate constraints
        return all((fn(**kw_args) for fn in relevantConstraints))

    def randomConfig(self):
        '''Sample a random configuration within the reachable c-space.
        Random values between joint values
        @return Random configuration within joint limits'''
        (lower, upper) = self.manip.GetJointLimits()
        joint_values = numpy.zeros(len(lower))
        for i in range(len(lower)):
            joint_values[i] = random.uniform(lower[i], upper[i])
        if self.manip.IsCollisionFree(joint_values, obstacles=self.static_obstacles):
            return joint_values
        return self.randomConfig()
    
    def constrainedExtend(self, q_near, q_target):
        # TODO: implement with circular difference
        qs = q_near
        qs_old = q_near
        while True:
            if numpy.array_equal(q_target, qs):
                return qs # Reached target
            elif numpy.linalg.norm(numpy.subtract(q_target, qs)) > numpy.linalg.norm(numpy.subtract(qs_old, q_target)):
                return qs_old # Moved further away

            qs_old = qs
            dist = numpy.linalg.norm(numpy.subtract(q_target, qs))
            qs_config_proposed = qs + min(self.QSTEP, dist)*(numpy.subtract(q_target, qs) / dist)
            qs_config = self.approveNewNode(qs_config_proposed, qs)
            if qs_config is not None:
                # qs = self.getNextIdx(T)
                qs = qs_config
                yield qs_config
            else:
                return qs_old

    def approveNewNode(self, qs_proposed, qs_parent):
        '''We will only approve the new node on several conditions. We
        constrain it to be within joint limits. We reject if its in 
        collision or if it violents any path wide constraints
        @param qs_proposed joint position of proposed node
        @param qs_parent joint positon of parent node
        @param qs_config joint configuration is approved,
                otherwise None'''
        qs_config = self.clampJointLimits(qs_proposed)
        #collision_free = self.checkEdgeCollision(qs_config, qs_parent)

        ee_pose = self.manip.ComputeFK(qs_config)
        ee_constraint = self.evaluateConstraints(ConstraintType.PATH_EE, pose=ee_pose)
        joint_constraint = self.evaluateConstraints(ConstraintType.PATH_JOINT, config=qs_config, pose=ee_pose)
        if ee_constraint and joint_constraint:
            return qs_config
        else:
            return None

    def clampJointLimits(self, qs):
        '''Given a proposed next joint space location, check that it is within
        joint limits. If it is not, clamp it to the nearest joint limit
        (joint wise)
        @param qs Joint configuration
        @param qs_new Joint configuration'''
        (lower, upper) = self.manip.GetJointLimits()
        qs_new = [max(lower[i], min(qs[i], upper[i])) for i in range(len(lower))]
        return qs_new