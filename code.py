# USE SPYDER TO VISUALISE THE PLOTS BETTER

from math import sqrt, sin, cos, acos, pi
import numpy as np
import matplotlib.pyplot as plt
import random

# Function that returns the 2D rotation matrix to rotate a vector by theta radians ccw
def rotmat(theta):
    return np.array([[cos(theta),-sin(theta)], [sin(theta),cos(theta)]])

# Function that returns the distance between two points
def dist(p1, p2):
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Function that returns the vectors of points of intersections of two circles given their centres and radii
def circle_intersect(c1, r1, c2, r2, choose_one=0):
    del_c = dist(c1, c2)
    if del_c < abs(r2 - r1): return []
    elif del_c > r1 + r2: return []
    else:
        x1 = 0.5*(c1[0] + c2[0]) + ((r1**2 - r2**2)/(2*(del_c**2)))*(c2[0] - c1[0]) + 0.5*sqrt(2*((r1**2 + r2**2)/(del_c**2)) - (((r1**2 - r2**2)**2)/(del_c**4)) - 1)*(c2[1] - c1[1])
        y1 = 0.5*(c1[1] + c2[1]) + ((r1**2 - r2**2)/(2*(del_c**2)))*(c2[1] - c1[1]) + 0.5*sqrt(2*((r1**2 + r2**2)/(del_c**2)) - (((r1**2 - r2**2)**2)/(del_c**4)) - 1)*(c1[0] - c2[0])
        x2 = 0.5*(c1[0] + c2[0]) + ((r1**2 - r2**2)/(2*(del_c**2)))*(c2[0] - c1[0]) - 0.5*sqrt(2*((r1**2 + r2**2)/(del_c**2)) - (((r1**2 - r2**2)**2)/(del_c**4)) - 1)*(c2[1] - c1[1])
        y2 = 0.5*(c1[1] + c2[1]) + ((r1**2 - r2**2)/(2*(del_c**2)))*(c2[1] - c1[1]) - 0.5*sqrt(2*((r1**2 + r2**2)/(del_c**2)) - (((r1**2 - r2**2)**2)/(del_c**4)) - 1)*(c1[0] - c2[0])
        if choose_one == 1: return [np.array([x1, y1])]
        elif choose_one == 2: return [np.array([x2, y2])]
        else: return [np.array([x1, y1]), np.array([x2, y2])]

# Solves for the position of joint 3, by finding the point of intersection of the circles centred at joints 1 and 2
def joint3_solver(joints, b, c):
    joint1 = joints[1]  
    joint2 = joints[2]
    joint3 = circle_intersect(joint2, b, joint1, c, choose_one=2)
    if len(joint3) == 0: return []
    return joint3[0]

# Performs position synthesis for a given set of link lengths, ternary link angle, output orientation and input angle
def pos_syn(links, alpha2, theta8, theta2):
    a, b, c, d, e, f, g, h = links                                             # unpack link lengths except f
    theta2 = theta2 * pi/180                                                   # convert the given angles to radians
    theta8 = theta8 * pi/180
    joints = {}                                                                # dictionary that stores all the joint coordinates
    joints[0] = np.array([0, 0])                                               # joint 0 is at the origin
    joints[1] = np.array([d, 0])                                               # joint 1 is at (d, 0) (ground link is horizontal)
    joints[2] = np.array([a*cos(theta2), a*sin(theta2)])                       # joint 2 (crank link's second joint) is at (acos(theta2), asin(theta2))
    joints[3] = joint3_solver(joints, b, c)                                    # joint 3 is obtained using the circle_intersection function
    if len(joints[3]) == 0: return None, 0                                     # check if a valid joint 3 was obtained
    joints[4] = joints[0] + (e/a) * (rotmat(alpha2) @ (joints[2] - joints[0])) # joint 4 (crank link's third joint) is obtained by rotating vector a by the ternary link angle and scaling it
    joints[5] = joints[1] + (h/c) * (joints[3] - joints[1])                    # joint 5 (the first joint on the output link) obtained by scaling the follower by the link length ratio
    joints[6] = joints[5] - np.array([g*cos(theta8), g*sin(theta8)])           # joint 6 (the second joint on the output link) obtained by using the fact that output link is a constant vector
    if abs(f - dist(joints[4], joints[6])) > 1.0: return None, 0               # length of f link is measured to validate the links
    theta4 = acos((joints[5][0] - joints[1][0])/dist(joints[5], joints[1]))    # get the output angle
    return joints, theta4

# Function that takes in the joint coordinates and returns angles of the crank link, coupler link and the f link
def get_angles(joints, alpha2):
    angles = {}
    angles[1] = acos((joints[2][0] - joints[0][0])/dist(joints[2], joints[0]))     # angle between crank link (a) and x axis
    angles[2] = acos((joints[3][0] - joints[2][0])/dist(joints[3], joints[2]))     # angle between coupler link (b) and x axis
    angles[4] = acos((joints[3][0] - joints[1][0])/dist(joints[3], joints[1]))     # angle between ouput link (h) and x axis
    angles[5] = angles[2] - (alpha2*pi/180)                                                        # angle between the other crank link (e) and x axis
    angles[6] = acos((joints[6][0] - joints[4][0])/dist(joints[6], joints[4]))     # angle between the link of length f and the x axis
    angles[7] = acos((joints[6][0] - joints[5][0])/dist(joints[6], joints[5]))
    return angles

# Function iterates through all the input angles, stepping by 0.01 degree at a time and validates that the given link lengths ensure complete range of motion
def check_validity(links, alpha2, theta8):
    joints_t = []
    theta2_min = alpha2
    theta2_max = 90
    theta2s = [i/100 for i in range(int(theta2_min*100), int(theta2_max*100))]
    theta4_min = 90
    theta4_max = 0
    for theta2 in theta2s:
        joints, theta4 = pos_syn(links, alpha2, theta8, theta2)
        if joints is None:
            return False, 0
        else:
            joints_t.append(joints)
            if theta4 > theta4_max:
                theta4_max = theta4
            if theta4 < theta4_min:
                theta4_min = theta4
    return joints_t, theta4_max - theta4_min

# Function that plots the linkage for all input angles, stepping by 1 degree at a time
def animate_motion(links, alpha2, theta8, plot=True, plot_cg=False):
    joints_t = []                                                                 # list to store the joint coordinates for all the input angles
    angles_t = []
    theta2_min = alpha2
    theta2_max = 90
    theta2s = [i for i in range(int(theta2_min), int(theta2_max))]
    
    cg_points = {}
    cg_points[0] = []
    cg_points[1] = []
    cg_points[2] = []
    cg_points[3] = []
    cg_points[4] = []
    
    for theta2 in theta2s:
        joints, _ = pos_syn(links, alpha2, theta8, theta2)
        angles = get_angles(joints, alpha2)
        joints_t.append(joints)
        angles_t.append(angles)
        joints = []
        for joint in joints_t[-1]:
            joints.append(rotmat((-90-theta8)*pi/180) @ joints_t[-1][joint])      # rotate the whole mechanism by -90 - theta8 so that the output link is vertical
        if plot:
            fig, ax = plt.subplots()
            ax.set_xlim([-5, 115])
            ax.set_ylim([-100, 30])
            ax.set_aspect('equal')
            ax.plot([joints[0][0], joints[1][0]], [joints[0][1], joints[1][1]], 'k')
            ax.plot([joints[0][0], joints[2][0]], [joints[0][1], joints[2][1]], 'b')
            ax.plot([joints[2][0], joints[3][0]], [joints[2][1], joints[3][1]], 'y')
            ax.plot([joints[1][0], joints[5][0]], [joints[1][1], joints[5][1]], 'g')
            ax.plot([joints[0][0], joints[4][0]], [joints[0][1], joints[4][1]], 'r')
            ax.plot([joints[2][0], joints[4][0]], [joints[2][1], joints[4][1]], 'g')
            ax.plot([joints[4][0], joints[6][0]], [joints[4][1], joints[6][1]], 'k')
            ax.plot([joints[5][0], joints[6][0]], [joints[5][1], joints[6][1]], 'b')
        else: plot_cg = False    
        
        # find centre of gravity coordinates
        cg_points[0].append((1/3)*(joints[0] + joints[2] + joints[4]))         
        cg_points[1].append((1/2)*(joints[2] + joints[3]))
        cg_points[2].append((1/2)*(joints[1] + joints[5]))
        cg_points[3].append((1/2)*(joints[4] + joints[6]))
        cg_points[4].append((1/2)*(joints[5] + joints[6]))
        
        # plot the centre of gravity trajectories
        if plot_cg:
            colour = ['r.', 'y.', 'g.', 'k.', 'b.']
            for key in cg_points:
                for point in cg_points[key]:
                    ax.plot(point[0], point[1], colour[key], markersize=0.5)
        
        if plot:
            plt.show()
            plt.close(fig)
        
    return joints_t, angles_t
    

# Function that iterates through random lengths of links within 5 units of a given set of link lengths
# Iterates until it obtains 100 valid link lengths

def link_length_iterator(alpha2, theta8):
    max_iter = 100
    iter = 0
    valid_iter = 0
    successful_link_lengths = []

    a = 25.76
    b = 63.26
    c = 40.44
    d = 49.26
    e = 19.49
    f = 89.90
    g = 42.25
    h = 80.96

    while(valid_iter < max_iter):                                              # generate random link lengths within 5 units of the original link lengths
        if iter != 0:
            b = 63.26 + (10*random.random() - 5)
            c = 40.44 + (10*random.random() - 5)
            d = 49.26 + (10*random.random() - 5)
            e = 19.49 + (10*random.random() - 5)
            f = 89.90 + (10*random.random() - 5)
            g = 42.25 + (10*random.random() - 5)
            h = 80.96 + (10*random.random() - 5)
        iter += 1
        links = [a, b, c, d, e, f, g, h]
        valid, theta4 = check_validity(links, alpha2, theta8)
        if not valid: continue                                                 # if valid, append to successful_link_lengths and return at the end
        else:
            print(f"Iteration: {valid_iter}")
            valid_iter = valid_iter + 1
            successful_link_lengths.append([links, theta4])

    return successful_link_lengths

# performs velocity analysis for a given linkage 
def vel_analysis(links, angles_t, alpha2):
    ang_vels_t = {}
    ang_vels_t[2] = []
    ang_vels_t[4] = []
    ang_vels_t[6] = []
    ang_vels_t[7] = []
    for angles in angles_t:
        ang_vels = {}
        r1, r2, r3, _, r5, r6, r7, r4 = links
        r4 = r4 - r3
        # obtain angular velocities by solving the matrix equation obtained from velocity analysis
        vector = np.array([r1*cos(angles[1]), r1*sin(angles[1]), r5*cos(angles[5]), r5*sin(angles[5])]).T
        matrix = np.array([[-r2*cos(angles[2]), -r3*cos(angles[4]), 0, 0],
                          [-r2*sin(angles[2]), -r3*sin(angles[4]), 0, 0],
                          [0, (r3 + r4)*cos(angles[4]), -r6*cos(angles[6]), -r7*cos(angles[7])],
                          [0, (r3 + r4)*sin(angles[4]), -r6*sin(angles[6]), -r7*sin(angles[7])]])
        res = np.linalg.inv(matrix) @ vector
        ang_vels[2], ang_vels[4], ang_vels[6], ang_vels[7] = res.flatten()
        ang_vels_t[2].append(ang_vels[2])
        ang_vels_t[4].append(ang_vels[4])
        ang_vels_t[6].append(ang_vels[6])
        ang_vels_t[7].append(ang_vels[7])
    
    # plot the angular velocity curves
    theta2s = [i for i in range(int(alpha2), int(90.0))]
    fig, ((ax2, ax4), (ax6, ax7)) = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("Angular Velocity Curves")
    ax2.plot(theta2s, ang_vels_t[2])
    ax2.set_xlabel(r"Input Angle $\theta_1$")
    ax2.set_ylabel(r"Angular Velocity of" + '\n' + r" Coupler Link ($\omega_2$)")
    ax2.grid(True)
    ax4.plot(theta2s, ang_vels_t[4])
    ax4.set_xlabel(r"Input Angle $\theta_1$")
    ax4.set_ylabel(r"Angular Velocity of" + '\n' + r" Rocker Link ($\omega_4$)")
    ax4.grid(True)
    ax6.plot(theta2s, ang_vels_t[6])
    ax6.set_xlabel(r"Input Angle $\theta_1$")
    ax6.set_ylabel(r"Angular Velocity of" + '\n' + r" f-Link ($\omega_6$)")
    ax6.grid(True)
    ax7.plot(theta2s, ang_vels_t[7])
    ax7.set_xlabel(r"Input Angle $\theta_1$")
    ax7.set_ylabel(r"Angular Velocity of" + '\n' + r" Output Link ($\omega_7$)")
    ax7.grid(True)
    plt.show()
    plt.close(fig)
    return ang_vels_t

# performs acceleration analysis for a given linkage (perform velocity analysis to get angular velocities first)
def accl_analysis(links, angles_t, ang_vels_t, alpha2):
    ang_accls_t = {}
    ang_accls_t[2] = []
    ang_accls_t[4] = []
    ang_accls_t[6] = []
    ang_accls_t[7] = []
    for i in range(len(angles_t)):
        angles = angles_t[i]
        ang_vels = {}
        ang_vels[2] = ang_vels_t[2][i]
        ang_vels[4] = ang_vels_t[4][i]
        ang_vels[6] = ang_vels_t[6][i]
        ang_vels[7] = ang_vels_t[7][i]
        ang_accls = {}
        r1, r2, r3, _, r5, r6, r7, r4 = links
        r4 = r4 - r3
        # obtain angular velocities by solving the matrix equation obtained from acceleration analysis
        vector = np.array([(-1)*r1*cos(angles[1]) + (ang_vels[4]**2)*(r3*cos(angles[4])) - (ang_vels[2]**2)*(r2*cos(angles[2])),
                           (-1)*r1*sin(angles[1]) + (ang_vels[4]**2)*(r3*sin(angles[4])) - (ang_vels[2]**2)*(r2*sin(angles[2])),
                           (-1)*r5*cos(angles[5]) + (ang_vels[4]**2)*((r3+r4)*cos(angles[4])) - (ang_vels[6]**2)*(r6*cos(angles[6])) - (ang_vels[7]**2)*(r7*cos(angles[7])),
                           (-1)*r5*sin(angles[5]) + (ang_vels[4]**2)*((r3+r4)*sin(angles[4])) - (ang_vels[6]**2)*(r6*sin(angles[6])) - (ang_vels[7]**2)*(r7*sin(angles[7]))]).T
        matrix = np.array([[-r2*cos(angles[2]), -r3*cos(angles[4]), 0, 0],
                          [-r2*sin(angles[2]), -r3*sin(angles[4]), 0, 0],
                          [0, (r3 + r4)*cos(angles[4]), -r6*cos(angles[6]), -r7*cos(angles[7])],
                          [0, (r3 + r4)*sin(angles[4]), -r6*sin(angles[6]), -r7*sin(angles[7])]])
        res = np.linalg.inv(matrix) @ vector
        ang_accls[2], ang_accls[4], ang_accls[6], ang_accls[7] = res.flatten()
        ang_accls_t[2].append(ang_accls[2])
        ang_accls_t[4].append(ang_accls[4])
        ang_accls_t[6].append(ang_accls[6])
        ang_accls_t[7].append(ang_accls[7])
    
    # plot the angular acceleration curves
    theta2s = [i for i in range(int(alpha2), int(90.0))]
    fig, ((ax2, ax4), (ax6, ax7)) = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("Angular Acceleration Curves\n")
    ax2.plot(theta2s, ang_accls_t[2])
    ax2.set_xlabel(r"Input Angle $\theta_1$")
    ax2.set_ylabel(r"Angular Acceleration of" + '\n' + r" Coupler Link ($\alpha_2$)")
    ax2.grid(True)
    ax4.plot(theta2s, ang_accls_t[4])
    ax4.set_xlabel(r"Input Angle $\theta_1$")
    ax4.set_ylabel(r"Angular Acceleration of" + '\n' + r" Rocker Link ($\alpha_4$)")
    ax4.grid(True)
    ax6.plot(theta2s, ang_accls_t[6])
    ax6.set_xlabel(r"Input Angle $\theta_1$")
    ax6.set_ylabel(r"Angular Acceleration of" + '\n' + r" f-Link ($\alpha_6$)")
    ax6.grid(True)
    ax7.plot(theta2s, ang_accls_t[7])
    ax7.set_xlabel(r"Input Angle $\theta_1$")
    ax7.set_ylabel(r"Angular Acceleration of" + '\n' + r" Output Link ($\alpha_7$)")
    ax7.grid(True)
    plt.show()
    plt.close(fig)
    
    return ang_accls_t

def main():
    link_lengths = link_length_iterator(37.2, 329.7)                           # obtain 100 valid link length combinations
    if len(link_lengths) == 0: print("empty")
    max_out = 0
    max_links = []
    for entry in link_lengths:                                                 # among the 100 valid link lengths, find the one that has the highest range of motion, i.e. output angle
        links, theta4 = entry
        if max_out < theta4:
            max_out = theta4
            max_links = links
        print(f"a: {links[0]}, b: {links[1]}, c: {links[2]}, d: {links[3]}, e: {links[4]}, f: {links[5]}, g: {links[6]}, h: {links[7]}")
    
    
    # Comment the following line if you want to generate a new solution
    max_links = [25.76, 65.89452485607082, 39.59148985494131, 50.86133087296963, 19.323308548183956, 89.43484732760035, 41.430820084078746, 82.7499333103564]
    
    # Output the obtained link lengths
    print()
    print("====================================================================")
    print(f"a: {max_links[0]}, b: {max_links[1]}, c: {max_links[2]}, d: {max_links[3]}, e: {max_links[4]}, f: {max_links[5]}, g: {max_links[6]}, h: {max_links[7]}, theta4: {max_out}")
    print("====================================================================")
    
    
    joints, angles = animate_motion(max_links, 37.2, 329.7, plot=True, plot_cg=True)         # animate the linkage
    ang_vels_t = vel_analysis(max_links, angles, 37.2)                         # plot angular velocity curves
    accl_analysis(max_links, angles, ang_vels_t, 37.2)                         # plot angular acceleration curves
    

if __name__ == "__main__":
    main()

        
        



