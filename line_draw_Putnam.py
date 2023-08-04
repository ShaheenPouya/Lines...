import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
from scipy.ndimage import gaussian_filter1d

#plt.gca().set_aspect('equal')

# Load the data
inner_line = pd.read_csv('inner.csv', header=None, names=['lat', 'lon'])
outer_line = pd.read_csv('outer.csv', header=None, names=['lat', 'lon'])
zones = pd.read_csv('zones.csv')


# Shifts the starting point of the DataFrames to the specified index
# Useful since one of them is out of sync
def shift_start(df, new_start):
    # Create a new index that starts at new_start and wraps around to the beginning
    new_index = list(range(new_start, len(df))) + list(range(new_start))
    # Reindex the DataFrame with the new index, and reset the index
    return df.reindex(new_index).reset_index(drop=True)

# Reverse outer line so their ordering matches
#outer_line = outer_line.iloc[::-1].reset_index(drop=True)

# Shift the start of the DataFrames to index 240
inner_line = shift_start(inner_line, 35)
outer_line = shift_start(outer_line, 35)


# Necessary since the lines do not loop
# Forces the inner/outer lines to loop around.
# inner_line.append(inner_line.iloc[0])
# outer_line.append(outer_line.iloc[0])

# Create a figure and plot the lines
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(inner_line['lon'], inner_line['lat'], 'b-', label='Inner Line')
ax.plot(outer_line['lon'], outer_line['lat'], 'r-', label='Outer Line')

# Add title and labels
ax.set_title('Race Track')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add legend
ax.legend()

# Create a line object for the connections, initially with no data
line, = ax.plot([], [], 'k-')

def slope(p1, p2):
    # Calculate the slope between two points
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def dot_product(p1, p2, p3):
    # Calculate the dot product of the vectors from p1 to p2 and from p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1
    return v1[0]*v2[0] + v1[1]*v2[1]

def l_size(p1,p2):
    if type(p1) == list:
        p1 = np.array(p1)
        p2 = np.array(p2)
    # Calculate the distance between two points
    v1 = p2 - p1
    return np.sqrt(v1[0]**2 + v1[1]**2)

def perp_point(p1, p2, outer_points):
    # Find the point in outer_points that forms a line perpendicular to the line from p1 to p2
    acceotable_points = []
    for p in outer_points:
        if l_size(p1,p) < 0.0002:
            acceotable_points.append(p)
    dot_products = [dot_product(p1, p2, p) for p in acceotable_points]
    sizer = [l_size(p1, p) for p in acceotable_points]

    return acceotable_points[np.argmin(np.abs(np.array(dot_products)))]


def line_locator_inner(i):
    # Get the current point on the inner line and the next point
    line, = ax.plot([], [], 'b-')
    p1 = inner_line[['lon', 'lat']].iloc[i].to_numpy()
    p2 = inner_line[['lon', 'lat']].iloc[(i+1) % len(inner_line)].to_numpy() # wrap around to the start if necessary
    # Get all points on the outer line
    outer_points = outer_line[['lon', 'lat']].iloc[[j % len(outer_line) for j in range(i-30, i+30)]].to_numpy()
    # Find the point on the outer line that forms a line perpendicular to the line from p1 to p2
    p3 = perp_point(p1, p2, outer_points)
    line.set_data([p1[0], p3[0]], [p1[1], p3[1]])
    return line,

def line_locator_outer(i):
    # Get the current point on the outer line and the next point
    line, = ax.plot([], [], 'r-')
    p1 = outer_line[['lon', 'lat']].iloc[i].to_numpy()
    p2 = outer_line[['lon', 'lat']].iloc[(i+1) % len(inner_line)].to_numpy() # wrap around to the start if necessary
    # Get all points on the outer line
    inner_points = inner_line[['lon', 'lat']].iloc[[j % len(outer_line) for j in range(i-30, i+30)]].to_numpy()
    # Find the point on the outer line that forms a line perpendicular to the line from p1 to p2
    p3 = perp_point(p1, p2, inner_points)
    line.set_data([p1[0], p3[0]], [p1[1], p3[1]])
    return line,

## To check that the starting points of inner and outer are same place!
# line = line_locator_inner(10)
# line = line_locator_outer(0)
# plt.show()
# quit()

def calculate_lines(inner_line, outer_line):
    # Initialize an empty list to store line segments
    lines = []
    # Iterate over the points in the inner line
    for i in range(len(inner_line)):
        # Get the current point on the inner line and the next point
        p1 = inner_line[['lon', 'lat']].iloc[i].to_numpy()
        p2 = inner_line[['lon', 'lat']].iloc[(i+1) % len(inner_line)].to_numpy() # wrap around to the start if necessary
        # Get all points on the outer line
        outer_points = outer_line[['lon', 'lat']].iloc[[j % len(outer_line) for j in range(i-30, i+30)]].to_numpy()
        # Find the point on the outer line that forms a line perpendicular to the line from p1 to p2
        p3 = perp_point(p1, p2, outer_points)
        # Append the line segment to the list of lines
        lines.append(((p1[0], p3[0]), (p1[1], p3[1])))
    return lines

def make_middle(lines):
    # To create and store middle points for each line
    df_line = pd.DataFrame(lines)
    df_line.columns = ['x', 'y']
    # Creating two new columns for middle points
    df_line = df_line.assign(mid_x=pd.Series(np.zeros(len(df_line))).values)
    df_line = df_line.assign(mid_y=pd.Series(np.zeros(len(df_line))).values)
    # Calculating middle points and adding them to data frame
    offset = 30
    steps = 111
    for i in range (len(inner_line)):
        x1 = df_line.iloc[i, 0][0]
        x2 = df_line.iloc[i, 0][1]
        if x1 == x2:
            x2 = x1+ 1e-7
        y1 = df_line.iloc[i, 1][0]
        y2 = df_line.iloc[i, 1][1]
        slope = (y2 - y1) / (x2 - x1)
        st = y1 - slope * x1
        mid_xx = np.linspace(x1, x2, steps)
        mid_yy = np.linspace(y1, y2, steps)
        mid_x = mid_xx[offset:-offset]
        mid_y = mid_yy[offset:-offset]
        df_line.loc[[i], 'mid_x'] = pd.Series([[mid_x]], index=df_line.index[[i]])
        df_line.loc[[i], 'mid_y'] = pd.Series([[mid_y]], index=df_line.index[[i]])
    return df_line

def select_points(df_line):
    # Select a point from each segment's midline
    # Here, we'll simply select the midpoint of the midline for each segment
    selected_points = []
    for i in range(len(df_line)):
        mid_x = df_line.loc[i, "mid_x"][0]
        mid_y = df_line.loc[i, "mid_y"][0]
        selected_points.append((mid_x[len(mid_x)//2], mid_y[len(mid_y)//2]))
    return selected_points


def calculate_curvature(line):
    # Calculate curvature of initial racing line
    curvatures = []
    for i in range(1, len(line) - 1):  # skip first and last points
        # Get the three points that form the current bend
        p1 = np.array(line[i - 1])
        p2 = np.array(line[i])
        p3 = np.array(line[i + 1])

        # Calculate the angle (in degrees) formed by these points
        angle = np.degrees(np.arccos(np.dot(p3 - p2, p1 - p2) / (np.linalg.norm(p3 - p2) * np.linalg.norm(p1 - p2))))

        # Approximate curvature as inverse of angle
        # 1e-5 is added to avoid division by zero
        curvature = 1 / (angle + 1e-5)
        curvatures.append(curvature)
    return curvatures


def make_points(start,finish,condition):
    #Reads the condition and assigns points from vertical line as start middle and end of racing line in that zone
    midle = int((start + finish) / 2)
    st,mid,end = [0,0],[0,0],[0,0]

    if list(condition)[0] == "i":
        st_l = 0
    elif list(condition)[0] == "o":
        st_l = -1
    elif list(condition)[0] == "m":
        st_l = int(len(list(df_line.loc[start, "mid_x"][0]))/2)+1
    st[0] = list(df_line.loc[start, "mid_x"][0])[st_l]
    st[1] = list(df_line.loc[start, "mid_y"][0])[st_l]

    if len(condition) == 3:
        if list(condition)[1] == "i":
            md_l = 0
        elif list(condition)[1] == "o":
            md_l = -1
        elif list(condition)[1] == "m":
            md_l = int(len(list(df_line.loc[start, "mid_x"][0])) / 2) + 1
        mid[0] = list(df_line.loc[midle, "mid_x"][0])[md_l]
        mid[1] = list(df_line.loc[midle, "mid_y"][0])[md_l]

    if list(condition)[-1] == "i":
        en_l = 0
    elif list(condition)[-1] == "o":
        en_l = -1
    elif list(condition)[-1] == "m":
        en_l = int(len(list(df_line.loc[start, "mid_x"][0]))/2)+1
    end[0] = list(df_line.loc[finish, "mid_x"][0])[en_l]
    end[1] = list(df_line.loc[finish, "mid_y"][0])[en_l]

    return st,mid,end

def circler(st,mid,end):
    #Creates a circle using three points
    temp = mid[0]**2 + mid[1]**2
    bc = (st[0]**2 + st[1]**2 - temp) / 2
    cd = (temp - end[0]**2 - end[1]**2) / 2
    det = (st[0] - mid[0]) * (mid[1] - end[1]) - (mid[0] - end[0]) * (st[1] - mid[1])
    if abs(det) < 1.0e-10:
        return None
    cx = (bc*(mid[1] - end[1]) - cd*(st[1] - mid[1])) / det
    cy = ((st[0] - mid[0]) * cd - (mid[0] - end[0]) * bc) / det
    radius = ((cx - st[0])**2 + (cy - st[1])**2)**.5
    return radius,cx,cy

def curver(st,end,condition):
    #Uses the circle to assign closest points from vertical lines to that imaginary circle
    racing_points =[]
    curv_st, curv_mid, curv_end = make_points(st, end, condition)
    radius, cx, cy = circler(curv_st, curv_mid, curv_end)
    center = [cx,cy]
    for i in range(st,end):
        checking_points = []
        for p in range(len(list(df_line.loc[i, "mid_x"][0]))):
            point = [0, 0]
            point[0] = list(df_line.loc[i, "mid_x"][0])[p]
            point[1] = list(df_line.loc[i, "mid_y"][0])[p]
            checking_points.append(point)

        distances = [(l_size(point, center) - radius) for point in checking_points]
        racing_points.append(checking_points[np.argmin(np.abs(np.array(distances)))])
    return racing_points

lines = calculate_lines(inner_line, outer_line)
df_line = make_middle(lines)

# st = 0
# end = 95
# condition ="oim"
# racing = curver(st,end,condition)
# for point in racing:
#         ax.scatter(*point, c='g',s=.1)  # plot selected points in yellow color
# line = line_locator_inner(st)
# line = line_locator_inner(end)
# plt.show()
# quit()

to_race = []
for i in range(len(zones)):
    st = zones.loc[i,'Start']
    end = zones.loc[i,'End']
    condition = zones.loc[i, 'Condition']
    racing = curver(st, end, condition)
    to_race.append(racing)
    #line = line_locator_inner(st)

def smooth_points(points, sigma=1):
    # Smooth the x and y coordinates separately
    xs = gaussian_filter1d([p[0] for p in points], sigma)
    ys = gaussian_filter1d([p[1] for p in points], sigma)
    return list(zip(xs, ys))

racing_line =[]
for zone in to_race:
    for point in zone:
        #ax.scatter(*point, c='g',s=.1)  # plot selected points in yellow color
        racing_line.append(point)


smoothed_points = smooth_points(racing_line, sigma=3)

x_smooth, y_smooth = zip(*smoothed_points)
plt.plot(x_smooth, y_smooth, color='g', label='Smooth Racing Line')
plt.savefig('Racing_Line.png',dpi = 300)
plt.show()
rl = pd.DataFrame(racing_line)
rl.to_csv('Racing_Line.csv')

