#!/usr/bin/env python

import cv2
import numpy as np
import random
from progressbar import *
import math
import argparse
import json

args = {}

# merge two dictionaries into a new one
def dict_merge(a,b):
    c = a.copy()
    c.update(b)
    return c

#print function (to be overridden)
def println(*args, **kwargs):
    pass

# parse arguments and set verbosity if main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connect-The-Dot Generator')
    parser.add_argument('config', help="The .json configuraiton file")
    parser.add_argument('image', help="The image file to create from")
    parser.add_argument('--verbosity', metavar='V', type=int, default=1, help='how much output (integers -1 through 5)')
    args = vars(parser.parse_args())
    print_level = args['verbosity']
    # step-in replacement for print, with custom indent-and-verbosity level.
    def p(string, level=0):
        if level <= args['verbosity']:
            print('  '*int(level) + str(string))
    println = p
    println("Loading Elements")

# return json loaded from file
def loadJSON(filename):
    println("Opening JSON file- {}...".format(filename), 1)
    with file(filename, "r") as f:
        println("Reading JSON data from file {}...".format(filename), 4)
        return json.loads(f.read())

# self explanatory
class IgnorableException(Exception):
    pass

# given an image where values are accessable as [y][x], draw a line between points p1~[x,y] and p2~[x,y], adding c every unit along the distance
def draw_line(canvas, p1, p2, c):
    dx = float(p2[0]-p1[0])
    dy = float(p2[1]-p1[1])
    max_disp = max(abs(dx), abs(dy))
    length = math.sqrt(dx*dx + dy*dy)
    dxom = dx/max_disp
    dyom = dy/max_disp
    clom = (c*length)/max_disp
    for i in range(int(max_disp)):
        x = int(p1[0] + i*dxom)
        y = int(p1[1] + i*dyom)
        canvas[x][y] = canvas[x][y] + clom

# core metaclass - keeps a dictionary of subclasses.
def CoreMetaGen(parent_name):
    class CoreMeta(type):
        def __init__(cls, name, bases, dct):
            println("Registering new {} - {}".format(parent_name, name), 4)
            cls._subclasses = {}
            cls._name = name
            cls._bases = [b for b in bases if hasattr(b, "__metaclass__") and b.__metaclass__ == CoreMeta]
            def append_subclass(cls, bases):
                for b in bases:
                    assert cls._name not in b._subclasses, "there are decendants with the same name!"
                    b._subclasses[cls._name] = cls
                    append_subclass(cls, b._bases)
            append_subclass(cls, cls._bases)
    return CoreMeta

# represents a transformation that can occur at a given index in a list of points, return the resultant new point (does not alter points themself)
class Action(object):
    __metaclass__ = CoreMetaGen("Action")
    def __init__(self, points):
        println("Instantiating new Action {}".format(self._name), 4)
        self.points = points
    def get(self, index):
        raise NotImplementedError

# move a point at index i 'up' on the image by step
class Up(Action):
    def __init__(self, *args, **kwargs):
        super(Up, self).__init__(kwargs['points'])
        self.step = kwargs['step']
    def get(self, index):
        return [self.points[index][0] - self.step, self.points[index][1]]

# move a point at index i 'down' on the image by step
class Down(Action):
    def __init__(self, *args, **kwargs):
        super(Down, self).__init__(kwargs['points'])
        self.step = kwargs['step']
    def get(self, index):
        return [self.points[index][0] + self.step, self.points[index][1]]

# move a point at index i 'left' on the image by step
class Left(Action):
    def __init__(self, *args, **kwargs):
        super(Left, self).__init__(kwargs['points'])
        self.step = kwargs['step']
    def get(self, index):
        return [self.points[index][0], self.points[index][1] - self.step]

# move a point at index i 'right' on the image by step
class Right(Action):
    def __init__(self, *args, **kwargs):
        super(Right, self).__init__(kwargs['points'])
        self.step = kwargs['step']
    def get(self, index):
        return [self.points[index][0], self.points[index][1] + self.step]

# move a point at index i toward its next neighbor by 'factor'
class Forward(Action):
    def __init__(self, *args, **kwargs):
        super(Forward, self).__init__(kwargs['points'])
        self.factor = kwargs['factor']
    def get(self, index):
        if index >= len(self.points)-1:
            raise IgnorableException()
        return [(1-self.factor)*self.points[index][0] + self.factor*self.points[index+1][0], (1-self.factor)*self.points[index][1] + self.factor*self.points[index+1][1]]

# move a point at index i toward its previous neighbor by 'factor'
class Backward(Action):
    def __init__(self, *args, **kwargs):
        super(Backward, self).__init__(kwargs['points'])
        self.factor = kwargs['factor']
    def get(self, index):
        if index < 1:
            raise IgnorableException()
        return [(1-self.factor)*self.points[index][0] + self.factor*self.points[index-1][0], (1-self.factor)*self.points[index][1] + self.factor*self.points[index-1][1]]

# move a point at index i towards the average of its immediate neighbors by 'factor'
class Supress(Action):
    def __init__(self, *args, **kwargs):
        super(Supress, self).__init__(kwargs['points'])
        self.factor = kwargs['factor']
    def get(self, index):
        if (index < 1) or (index >= len(self.points)-1):
            raise IgnorableException()
        return [(1-self.factor)*self.points[index][0] + (self.factor/2)*self.points[index+1][0] + (self.factor/2)*self.points[index-1][0], 
                (1-self.factor)*self.points[index][1] + (self.factor/2)*self.points[index+1][1] + (self.factor/2)*self.points[index-1][1]]

# move a point at index i away from the average of its immediate neighbors by 'factor'
class Extremize(Action):
    def __init__(self, *args, **kwargs):
        super(Extremize, self).__init__(kwargs['points'])
        self.factor = -kwargs['factor']
    def get(self, index):
        if (index < 1) or (index >= len(self.points)-1):
            raise IgnorableException()
        return [(1-self.factor)*self.points[index][0] + (self.factor/2)*self.points[index+1][0] + (self.factor/2)*self.points[index-1][0], 
                (1-self.factor)*self.points[index][1] + (self.factor/2)*self.points[index+1][1] + (self.factor/2)*self.points[index-1][1]]

# a filtering class defines what alterations are permissable, passes in the existing points, and the doFilter takes the new_point returns None if filtered out, otherwise returns the permitted point (these are applied in a chain)
class Filter(object):
    __metaclass__ = CoreMetaGen("Filter")
    def __init__(self, points, *args, **kwargs):
        println("Instantiating new Filter {}".format(self._name), 4)
        self.points = points
    def doFilter(self, new_point):
        raise NotImplementedError

# filters the point out if it is outside the border of the image.
class BoundaryFilter(Filter):
    def __init__(self, *args, **kwargs):
        super(BoundaryFilter, self).__init__(kwargs['points'])
        self.border = kwargs['border']
        self.width = kwargs['width']
        self.height = kwargs['height']
    def doFilter(self, new_point):
        if ((not new_point) or
           (new_point[0] < self.border) or
           (new_point[1] < self.border) or
           (new_point[0] >= self.width-self.border) or
           (new_point[1] >= self.height-self.border)):
            return None
        return new_point

# filters the point out if it allready exists as a point on the image
class RepeatFilter(Filter):
    def doFilter(self, new_point):
        if new_point and new_point in self.points:
            return None
        return new_point

# accepts all points, but modifies them to be permissable - as occuring on a square grid of size 'step'
class DescretiseFilter(Filter):
    def __init__(self, *args, **kwargs):
        super(DescretiseFilter, self).__init__(kwargs['points'])
        self.step = kwargs['step']
    def doFilter(self, new_point):
        if new_point:
            return [new_point[0] - new_point[0] % self.step, new_point[1] - new_point[1] % self.step]
        return None

# the fundamental image for the image blurring (as by convolutions through openCV), passes the width and height of the kernel image, as well as its 'radius'-r. gen() returns kernel
class Kernel(object):
    __metaclass__ = CoreMetaGen("Kernel")
    def __init__(self, k_width, k_height, r, *args, **kwargs):
        println("Instantiating new Kernel {}".format(self._name), 4)
        self.k_width = k_width
        self.k_height = k_height
        self.r = r
    def gen(self):
        raise NotImplementedError

# the canomical blur-kernel
class GaussianKernel(Kernel):
    def gen(self):
        kernel = np.zeros([self.k_width,self.k_height], np.float32)
        for x in range(self.k_width):
            for y in range(self.k_height):
                kernel[x][y] = np.float32(math.exp(-((x-(self.k_width-1)/2.0)**2+(y-(self.k_height-1)/2.0)**2)/(self.r**2)))
        kernel = kernel / np.sum(kernel)
        return kernel

# a blur-kernel more adept at keeping higher frequencies/image-sharpness
class LorentzianKernel(Kernel):
    def gen(self):
        kernel = np.zeros([self.k_width,self.k_height], np.float32)
        for x in range(self.k_width):
            for y in range(self.k_height):
                kernel[x][y] = np.float32(math.exp(-math.sqrt((x-(self.k_width-1)/2.0)**2+(y-(self.k_height-1)/2.0)**2)/self.r))
        kernel = kernel / np.sum(kernel)
        return kernel

# a class defining an input of points for the simulation to run on, simply returns an array of points (an array of 2-arrays)
class PointLoader(object):
    __metaclass__ = CoreMetaGen("PointLoader")
    def __init__(self, *args, **kwargs):
        println("Instantiating new PointLoader {}".format(self._name), 4)
    def get(self):
        raise NotImplementedError

# returns points within border, generated by a random-walk
class RandomPointLoader(PointLoader):
    def __init__(self, length, width, height, border, *args, **kwargs):
        super(RandomPointLoader, self).__init__()
        self.length = length
        self.width = width
        self.height = height
        self.border = border
    def get(self):
        points = []
        max_dist = min(self.width, self.height)/8
        x = self.width/2
        y = self.height/2
        for i in range(self.length):
            points.append([y,x])
            if random.randint(0,1):
                x = x + random.randint(1,max_dist)
            else:
                x = x - random.randint(1,max_dist)
            if random.randint(0,1):
                y = y + random.randint(1,max_dist)
            else:
                y = y - random.randint(1,max_dist)
            if x < self.border:
                x = self.border
            elif x >= self.width-self.border:
                x = self.width - self.border - 1
            if y < self.border:
                y = self.border
            elif y >= self.height-self.border:
                y = self.height - self.border - 1
        return points

# returns points within a border by randomising on x-and-y (statistically the lines tend to pass through the middle... -_- )
class TotalRandomPointLoader(PointLoader):
    def __init__(self, length, width, height, border, *args, **kwargs):
        super(TotalRandomPointLoader, self).__init__()
        self.length = length
        self.width = width
        self.height = height
        self.border = border
    def get(self):
        points = []
        for i in range(self.length):
            points.append([random.randint(self.border,self.height-1-self.border),random.randint(self.border,self.width-1-self.border)])
        return points

# returns points as loaded directly from a file (perhaps from a previous simulation?)
class JSONPointLoader(PointLoader):
    def __init__(self, filename, *args, **kwargs):
        super(JSONPointLoader, self).__init__()
        self.filename = filename
    def get(self):
        return loadJSON(self.filename)

# a class defining a numeric measure of 'good'/'bad' for a set of points (positive is 'badness')
class Metric(object):
    __metaclass__ = CoreMetaGen("Metric")
    def __init__(self, *args, **kwargs):
        println("Instantiating new Metric {}".format(self._name), 4)
    def get(self, points):
        raise NotImplementedError

# measures the dissimilarity between the lines and the image, by bluring both, and returning the mean of the absolute difference in intensity
class LinearBlurCompare(Metric):
    def __init__(self, img, kernel_name, *args, **kwargs):
        super(LinearBlurCompare, self).__init__()
        self.kernel = Kernel._subclasses[kernel_name](*args, **kwargs).gen()
        self.img = img.astype(np.float32, copy=False)
        self.img = cv2.filter2D(self.img,-1,self.kernel)
        self.img = 255-self.img
        self.img = self.img / np.std(self.img)
        self.canvas = np.zeros(self.img.shape, np.float32)
    def get(self, points):
        self.canvas.fill(0.0)
        for i in range(len(points)-1):
            draw_line(self.canvas, points[i], points[i+1], 255)
        self.canvas = cv2.filter2D(self.canvas,-1,self.kernel)
        self.canvas = self.canvas / np.std(self.canvas)
        diff_img = self.img - self.canvas
        return np.sum(np.absolute(diff_img))/(diff_img.shape[0]*diff_img.shape[1])

# measures the dissimilarity between the lines and the image, by bluring both, and returning the root-mean-square (RMS) of the difference in intensity
class RMSBlurCompare(Metric):
    def __init__(self, img, kernel_name, *args, **kwargs):
        super(RMSBlurCompare, self).__init__()
        self.kernel = Kernel._subclasses[kernel_name](*args, **kwargs).gen()
        self.img = img.astype(np.float32, copy=True)
        self.img = cv2.filter2D(self.img,-1,self.kernel)
        self.img = 255-self.img
        self.img = self.img / np.std(self.img)
        self.canvas = np.zeros(self.img.shape, np.float32)
    def get(self, points):
        self.canvas.fill(0.0)
        for i in range(len(points)-1):
            draw_line(self.canvas, points[i], points[i+1], 255)
        self.canvas = cv2.filter2D(self.canvas,-1,self.kernel)
        self.canvas = self.canvas / np.std(self.canvas)
        diff_img = self.img - self.canvas
        return np.linalg.norm(diff_img)/math.sqrt((diff_img.shape[0]*diff_img.shape[1]))

# measures the 'badness' by returning the average absolute difference in length between the points and desired-length
class AvgLengthCompare(Metric):
    def __init__(self, img, desired_length):
        super(AvgLengthCompare, self).__init__()
        self.desired_length = desired_length
    def get(self, points):
        sizes = []
        for i in range(len(points)-1):
            sizes.append(abs(math.sqrt((points[i][0]-points[i+1][0])**2 + (points[i][1]-points[i+1][1])**2) - self.desired_length))
        return sum(sizes)/len(sizes)

# a class defining a means of output of the point data
class Outputter(object):
    __metaclass__ = CoreMetaGen("Outputter")
    def __init__(self, *args, **kwargs):
        println("Instantiating new Outputter {}".format(self._name), 4)
    def output(self, points):
        raise NotImplementedError

# directly output connect-the-dot points to simple postscript
class PostScriptOutputter(Outputter):
    def __init__(self, filename, width, height, page_width=595, page_height=842, page_border_x=30, page_border_y=30, font="Arial", font_size=3, circle_size=0.8, opacity=0.5, t_offset_x=-2, t_offset_y=-2, *args, **kwargs):
        super(PostScriptOutputter, self).__init__()
        self.filename = filename
        self.width = width
        self.height = height
        self.page_width = page_width
        self.page_height = page_height
        self.page_border_x = page_border_x
        self.page_border_y = page_border_y
        self.font = font
        self.font_size = font_size
        self.circle_size = circle_size
        self.opacity = opacity
        self.t_offset_x = t_offset_x
        self.t_offset_y = t_offset_y
    def output(self, points, cycle=''):
        println("Writing data to Postscript file {}...".format(self.filename.format(cycle)), 4)
        f = open(self.filename.format(cycle), "w")
        f.write("%!\n")
        f.write("/cc {{newpath {} 0 360 arc closepath fill}} bind def\n".format(self.circle_size))
        f.write("/tt {newpath moveto show} bind def\n")
        f.write("{} setgray\n".format(self.opacity))
        f.write("/{} findfont {} scalefont setfont\n".format(self.font, self.font_size))
        sx = (self.page_width-self.page_border_x*2.0)/self.width
        sy = (self.page_height-self.page_border_y*2.0)/self.height
        scale = min(sx,sy)
        if sx < sy:
            ox = self.page_border_x
            oy = self.page_height/2.0 + self.height*sx/2.0
        else:
            ox = self.page_width/2.0 - self.width*sy/2.0
            oy = self.page_height - self.page_border_y
        for i,p in enumerate(points):
            f.write("{} {} cc\n".format(ox+p[1]*scale, oy-p[0]*scale))
            f.write("({}) {} {} tt\n".format(i+1, ox+p[1]*scale+self.t_offset_x, oy-p[0]*scale-self.t_offset_y))
        f.write("showpage\n")
        f.close()

# directly output points to file as JSON
class JSONOutputter(Outputter):
    def __init__(self, filename, *args, **kwargs):
        super(JSONOutputter, self).__init__()
        self.filename = filename
    def output(self, points, cycle=''):
        println("Writing JSON data from file {}...".format(self.filename.format(cycle)), 4)
        f = open(self.filename.format(cycle), "w")
        f.write(json.dumps(points))
        f.close()

# directly format and output the points as image-file.
class ImageOutputter(Outputter):
    def __init__(self, width, height, filename, *args, **kwargs):
        super(ImageOutputter, self).__init__()
        self.filename = filename
        self.width = width
        self.height = height
        self.canvas = np.zeros([height, width], np.float32)
    def output(self, points, cycle=''):
        self.canvas.fill(0.0)
        for i in range(len(points)-1):
            draw_line(self.canvas, points[i], points[i+1], 255.0)
        assert self.canvas.max > 0.0, '..no points supplied?'
        self.canvas = self.canvas * (255.0 / self.canvas.max())
        println("Writing Image data from file {}...".format(self.filename.format(cycle)), 4)
        cv2.imwrite(self.filename.format(cycle), 255-self.canvas)

# directly format and output the points as image-file - as blurred by a kernel.
class BlurredImageOutputter(Outputter):
    def __init__(self, width, height, filename, *args, **kwargs):
        super(BlurredImageOutputter, self).__init__()
        self.filename = filename
        self.width = width
        self.height = height
        self.canvas = np.zeros([height, width], np.float32)
        self.kernel = Kernel._subclasses[kwargs['kernel_name']](*args, **kwargs).gen()
    def output(self, points, cycle=''):
        self.canvas.fill(0.0)
        for i in range(len(points)-1):
            draw_line(self.canvas, points[i], points[i+1], 255.0)
        assert self.canvas.max > 0.0, '..no points supplied?'
        self.canvas = cv2.filter2D(self.canvas,-1,self.kernel)
        self.canvas = self.canvas * (255.0 / self.canvas.max())
        println("Writing Blurred Image data from file {}...".format(self.filename.format(cycle)), 4)
        cv2.imwrite(self.filename.format(cycle), 255-self.canvas)

# defines a 'stage' of the simulation, a set of actions,filters,metrics, (and outputters once complete) for the simulation to go through for a number of cycles
class Stage(object):
    def __init__(self, points, img, name):
        println("Instantiating new Stage {}".format(name), 4)
        self.actions = []
        self.filters = []
        self.outputters = []
        self.metrics = []
        self.metric_weights = []
        self.name = name
        self.img = img
        self.points = points
    def addAction(self, action_name, *args, **kwargs):
        self.actions.append(Action._subclasses[action_name](points=self.points, *args, **kwargs))
    def addFilter(self, filter_name, *args, **kwargs):
        self.filters.append(Filter._subclasses[filter_name](points=self.points, *args, **kwargs))
    def addOutputter(self, outputter_name, *args, **kwargs):
        self.outputters.append(Outputter._subclasses[outputter_name](*args, **kwargs))
    def addMetric(self, metric_name, weight, *args, **kwargs):
        self.metrics.append(Metric._subclasses[metric_name](self.img, *args, **kwargs))
        self.metric_weights.append(weight)
    def run(self, cycles, cycles_done): # run a stage of the simulation, is a random jitter+filter (acording to actions/filters) and relaxation of the points in the string according to the metrics, and output at the end.
        dissimilarity = float("inf")
        num_points = len(self.points)
        if cycles*num_points > 0:
            progress = ProgressBar(widgets=['{}: '.format(self.name),Percentage(),' ',Bar(marker='='),' ',ETA()], maxval=cycles*num_points).start()
            for iteration in range(cycles):
                for pi in range(num_points):
                    random.shuffle(self.actions)
                    for action in self.actions:
                        try:
                            new_point = action.get(pi)
                        except IgnorableException:
                            continue
                        for fil in self.filters:
                            new_point = fil.doFilter(new_point)
                        if new_point:
                            old_point = self.points[pi]
                            self.points[pi] = new_point
                            new_dissimilarity = 0
                            for m_i, m in enumerate(self.metrics):
                                new_dissimilarity += self.metric_weights[m_i] * m.get(self.points)
                            if (new_dissimilarity < dissimilarity):
                                dissimilarity = new_dissimilarity
                                break
                            else:
                                self.points[pi] = old_point
                    progress.update(iteration*num_points + pi)
            progress.finish()
        println("Outputting Stage Data", 2)
        for o in self.outputters:
            o.output(self.points, cycles+cycles_done)

# controls the execution of the whole simulation, sets up initial loading, the stages and their execution and final outputs.
class Executor(object):
    def __init__(self, config, img_filename):
        println("Instantiating the Executor", 4)
        self.config = config
        self.img = None
        self.stages = []
        self.stage_cycles = []
        self.cycles = None
        self.img_filename = img_filename
        self.points = []
        self.outputters = []
    def initialise(self): #Load everything...
        println("Loading input image - {}".format(self.img_filename), 1)
        self.img = cv2.imread(self.img_filename,0)
        println("Reading input image dimensions - {}".format(self.img.shape), 2)
        self.config['globals'].update({"width":self.img.shape[0], "height":self.img.shape[0]})
        self.cycles = self.config['cycles']
        println("Loading Initial points from - {}".format(self.config['PointLoader']['name']), 1)
        self.points = PointLoader._subclasses[self.config['PointLoader']['name']](**dict_merge(self.config['globals'], self.config['PointLoader']['config'])).get()
        println("Loading Simulation Outputters", 2)
        for outputter in self.config['outputters']:
            self.outputters.append(Outputter._subclasses[outputter['name']](**dict_merge(self.config['globals'], outputter['config'])))
        for stage_config in self.config['Stages']:
            println("Initiating Stage - {}".format(stage_config['name']), 1)
            new_stage = Stage(self.points, self.img, stage_config['name'])
            println("Initiating Stage Actions", 2)
            for action in stage_config['actions']:
                new_stage.addAction(action['name'], **dict_merge(self.config['globals'], action['config']))
            println("Initiating Stage Filters", 2)
            for fil in stage_config['filters']:
                new_stage.addFilter(fil['name'], **dict_merge(self.config['globals'], fil['config']))
            println("Initiating Stage Metrics", 2)
            for metric in stage_config['metrics']:
                new_stage.addMetric(metric['name'], metric['weight'], **dict_merge(self.config['globals'], metric['config']))
            println("Initiating Stage Outputters", 2)
            for outputter in stage_config['outputters']:
                new_stage.addOutputter(outputter['name'], **dict_merge(self.config['globals'], outputter['config']))
            println("Registering Stage", 3)
            self.stages.append(new_stage)
            self.stage_cycles.append(stage_config['cycles'])
    def run(self): #Iteratively run though the stages until total cycles runs out.
        cycles_done = 0
        stage_index = 0
        while cycles_done < self.cycles:
            println("Simulation Executing from cycle {} - with {} for {} cycles".format(cycles_done, self.stages[stage_index].name, min(self.stage_cycles[stage_index], self.cycles-cycles_done)), 3)
            self.stages[stage_index].run(min(self.stage_cycles[stage_index], self.cycles-cycles_done), cycles_done)
            cycles_done += self.stage_cycles[stage_index]
            stage_index = (stage_index+1) % len(self.stages)
        println("Outputting Final Simulation Data", 1)
        for o in self.outputters:
            o.output(self.points, self.cycles)


if __name__ == '__main__':
    println("Loading Config...")
    E = Executor(loadJSON(args['config']), args['image'])
    println("Initialising Simulation...")
    E.initialise()
    println("Executing Simulation...")
    E.run()
    println("Done.")



