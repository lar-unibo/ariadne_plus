
from numpy.lib.arraysetops import unique
from numpy.lib.function_base import append
import igraph as ig
import numpy as np
import cv2
import math
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops
import itertools
import torch
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')
import arrow 

debug = False

class ImageSegmentator(object):

    def __init__(self, num_segments=200, compactness=100):
        self.num_segments = num_segments
        self.compactness = compactness
        self.sigma = 0.5

    def segments(self, image, mask):
        return slic(image, mask=mask, n_segments=self.num_segments, compactness=self.compactness, sigma=self.sigma, slic_zero=True,
                    enforce_connectivity=True, multichannel=True, convert2lab=True)


class Ariadne(object):

    def __init__(self, image, image_mask, num_segments=100, plot_graph_flag = False):

        self.image = image
        self.image_mask = image_mask
        self.image_masked = self.maskSourceImage()
        self.num_segments = num_segments

        ##################################
        # Superpixel Segmentation
        ##################################

        self.segmentator = ImageSegmentator(num_segments=num_segments, compactness=100)

        self.labels = self.segmentator.segments(self.image, self.image_mask)

        self.regions = self.measureRegions(self.labels)


        start_time = arrow.utcnow()

        ##################################
        # Graph Generation
        ##################################
        self.graph_init = AriadneGraph(self.image, self.labels, self.regions)

        #print("GENERATION: ", (arrow.utcnow() - start_time).total_seconds() * 1000)

        if plot_graph_flag: self.graph_init.plotGraph("graph_init.png")

        start_time = arrow.utcnow()

        ##################################
        # Graph Simplification
        ##################################
        self.new_labels = self.graph_init.simplifyGraph(self.labels)
        self.regions_up = self.measureRegions(self.new_labels)

        #print("SIMPLIFICATION: ", (arrow.utcnow() - start_time).total_seconds() * 1000)

        self.graph = AriadneGraph(self.image, self.new_labels, self.regions_up)

        #self.graph.cycleIntersections()

        ##################################
        # Graph Clustering
        ##################################
        self.graph.clustering()


        #self.graph.plotGraph("graph.png")
        if plot_graph_flag: self.graph.plotGraph("graph.png")


    def measureRegions(self, labels_img):
        return regionprops(label_image=labels_img, intensity_image=self.image_mask, cache=True)

    def generateBoundaryImage(self, image, color=(0, 0, 1)):
        if image is None:
            return mark_boundaries(self.image, self.new_labels, color=color)
        else:
            return mark_boundaries(image, self.new_labels, color=color, mode="thick")

    def generateBoundaryImageInit(self, image, color=(1, 1, 0)):
        if image is None:
            return mark_boundaries(self.image, self.labels, color=color)
        else:
            #return mark_boundaries(image, self.labels, color=color, outline_color = color, mode="thick")
            return mark_boundaries(image, self.labels, color=color, mode="thick")


    def maskSourceImage(self):
        img = self.image.copy()
        img[self.image_mask < 127] = [0, 0, 0]
        return img

class AriadneGraph(object):

    def __init__(self, image, labels, regions):

        self.image = image
        self.labels = labels
        self.labels_boundary = []

        self.labels_unique = np.unique(self.labels)
        self.labels_unique = self.labels_unique[self.labels_unique != 0]
        self.regions = regions

        # Graph Creation
        self.g = ig.Graph()
        self.g.add_vertices(len(self.labels_unique))
        
        # Attributes initialization
        vs_labels = []
        vs_position = []
        vs_intersection = []
        for i, _ in enumerate(ig.VertexSeq(self.g)):
            vs_labels.append(self.labels_unique[i])
            vs_position.append([int(self.regions[i]["centroid"][1]), int(self.regions[i]["centroid"][0])])
            vs_intersection.append(0)

        self.g.vs["label"] = vs_labels
        self.g.vs["position"] = vs_position
        self.g.vs["intersection"] = vs_intersection

        # Edges addition
        edges = self.getEdgesFromLabelMask()
        self.g.add_edges(edges)

        # update intersection property
        self.assignIntersections()  

        # get list of unique intersections
        self.unique_ints_list = self.getUniqueIntersections()
        #print("unique intersections: ", self.unique_ints_list)


    def simplifyGraph(self, labels_img):

        '''
        ###############################################
        new_unique_ints_list = []
        ids_to_delete = []
        for i in range(len(self.unique_ints_list)):

            if i in ids_to_delete:
                continue

            for j in range(len(self.unique_ints_list)):
                if i == j:
                    continue
            
                if j in ids_to_delete:
                    continue
            
                th = self.unique_ints_list[i].shape[0] + self.unique_ints_list[j].shape[0] / 2
                print("th -> ", th)
                path = self.shortestPathNoInt(self.unique_ints_list[i][0], self.unique_ints_list[j][0])

                if len(path) <= th:
                    new_unique_ints_list.append(np.concatenate((self.unique_ints_list[i], self.unique_ints_list[j], np.array(path))))
                    if i not in ids_to_delete:
                        ids_to_delete.append(i)
                    if j not in ids_to_delete:
                        ids_to_delete.append(j)
        #print(new_unique_ints_list)
        print(ids_to_delete)

        for i in sorted(ids_to_delete, reverse=True):
            del self.unique_ints_list[i]
            
        for el in new_unique_ints_list:
            self.unique_ints_list.append(el)
        ###############################################
        
        '''

        labels_copy = labels_img.copy()
        for arr in self.unique_ints_list:
            for a in arr:
                labels_copy[labels_img == a] = arr[0]


        # remove spurious nodes
        to_delete_labels = [v["label"] for v in self.g.vs if v.degree() == 0]
        for el in to_delete_labels:
            labels_copy[labels_img == el] = 0

        return labels_copy


    def clustering(self):
        # clustering
        self.clusters_dict = {}
        cluster_output = self.g.clusters()
        for i, r in enumerate(cluster_output):
            self.clusters_dict[i] = [self.getLabelFromId(r_val) for r_val in r]
        
        # extract clusters without and with intersections
        self.free_clusters, self.inters_clusters = self.splitClusters()
        #print("free clusters: ", self.free_clusters)
        #print("inters clusters: ", self.inters_clusters)

    
    
    def splitClusters(self):

        free_intersection_ids = []
        intersection_ids = []
        for key, val in self.clusters_dict.items():
            for el in self.unique_ints_list:
                inter = set(el).intersection(set(val))
                if len(inter) == 0:
                    free_intersection_ids.append(key)
                else:
                    intersection_ids.append(key)

        return np.unique(free_intersection_ids), np.unique(intersection_ids)


    # list of list where each list contains a cluster of intersecting nodes identified by their (superpixel) label
    def getUniqueIntersections(self):
        vertices =  ig.VertexSeq(self.g)
        id_list = [v.index for v in vertices if self.g.vs[v.index]["intersection"] == 1]
        label_list = [self.getLabelFromId(id) for id in id_list]

        initial_nn_list = []
        for l in label_list:
            nn = self.getNeighborsFromLabels(l)
            nn_int = set(label_list).intersection(set(nn))
            nn_int.add(l)
            #label_nn_list.append(list(nn_int))
            #print("label nn list: \n", label_nn_list)

            not_expanded = True
            for nn_list in initial_nn_list:
                nn_intersection = set(nn_list).intersection(set(nn_int))
                if nn_intersection:
                    nn_not_intersection = [n for n in nn_int if n not in nn_intersection]
                    if nn_not_intersection: nn_list.update(set(nn_not_intersection))
                    not_expanded = False

            if not_expanded:
                initial_nn_list.append(nn_int)

        
        label_nn_list = [list(n) for n in initial_nn_list]
        
        # get individual intersections
        intersection_list = []
        for counter, r in enumerate(label_nn_list):

            int_tmp = r
            for i,l in enumerate(label_nn_list):
                if i == counter: # skip otherwise stuck!
                    continue

                path_int = set(int_tmp).intersection(set(l))
                if path_int:
                    for v in l:
                        int_tmp.append(v)

            intersection_list.append(np.unique(int_tmp))

        unique_intersections = []
        for arr in intersection_list:
            if not any(np.array_equal(arr, unique_arr) for unique_arr in unique_intersections):
                unique_intersections.append(arr.astype(np.int64))
        
        return unique_intersections


    def getIdFromLabel(self, label):
        for id, l in enumerate(self.g.vs["label"]):
            if l == label:
                return id

    def getLabelFromId(self, id):
        return self.g.vs["label"][id]

    def getCentroidFromId(self, id):
        c = self.g.vs["position"][id]
        return tuple([int(c[0]), int(c[1])])
    
    def getCentroidFromLabel(self, label):
        id = self.getIdFromLabel(label)
        return self.getCentroidFromId(id)

    def getIntersectionFromId(self, id):
        c = self.g.vs["intersection"][id]
        return int(c)

    def getIntersectionFromLabel(self, label):
        id = self.getIdFromLabel(label)
        if id == None:
            return 1
        else:
            return self.getIntersectionFromId(id)

    def getVertices(self):
        vertices = ig.VertexSeq(self.g)
        return [vertex.index for vertex in vertices]
    
    def getDegrees(self):
        vertices = self.getVertices()
        return self.g.degree(vertices)


    def assignIntersections(self):
        vertices = ig.VertexSeq(self.g)
        for v in vertices:
            if len(self.getNeighborsFromId(v)) > 2:
                self.g.vs[v.index]["intersection"] = 1
        
        # second pass -> nodes with only intersection nn nodes
        for v in vertices:
            nn = self.getNeighborsFromId(v)
            nn_f = [n for n in nn if self.getIntersectionFromId(self.getIdFromLabel(n)) == 0]
            if len(nn_f) == 0 and len(nn) > 1:
                self.g.vs[v.index]["intersection"] = 1
        
        
    def cycleIntersections(self):
        vertices = ig.VertexSeq(self.g)
        for v in vertices:
            if self.getIntersectionFromId(v.index) == 1:
                nn = self.getNeighborsFromId(v)
                for n in nn:
                    paths = self.g.get_all_simple_paths(v,self.getIdFromLabel(n), cutoff=3)
                    if len(paths) > 1:
                        self.g.vs[self.getIdFromLabel(n)]["intersection"] = 1


    def isBoundary(self, v):
        if not self.labels_boundary:
            l = np.unique(self.labels[0:1, :])
            r = np.unique(self.labels[-2:-1, :])
            t = np.unique(self.labels[:, 0:1])
            b = np.unique(self.labels[:, -2:-1])
            self.labels_boundary = list(np.unique(np.concatenate((l, r, t, b), axis=0)))
            #print("self.labels_boundary ", self.labels_boundary)
        
        if self.g.vs[v.index]["label"] in self.labels_boundary:
            return True

        return False

    def getNeighborsFromId(self, id):
        nn = np.unique(self.g.neighbors(id))
        return [self.getLabelFromId(n) for n in nn]

    def getNeighborsFromLabels(self, l):
        id = self.getIdFromLabel(l)
        return self.getNeighborsFromId(id)

    def getNeighborsIntersectionFromId(self, id):
        label = self.getLabelFromId(id)
        return self.getNeighborsIntersectionFromLabel(label)

    def getNeighborsIntersectionFromLabel(self, label):
        intersection_found = []
        for arr in self.unique_ints_list:
            for el in arr:
                if el == label:
                    intersection_found = arr
                    break
                            
        nn_int = [self.getNeighborsFromLabels(n) for n in intersection_found]
        if len(nn_int) > 0:
            nn_int = np.unique(np.hstack(nn_int))
        return nn_int


    def maskImage(self, l):
        mask = np.zeros(self.labels.shape, dtype=np.uint8)
        mask[self.labels == l] = 255
        return mask
 
 
    def getEdgesFromLabelMask(self):

        all_edges = []
        kernel = np.ones((5, 5), np.uint8)
        #vertices = ig.VertexSeq(self.g)  #vertices = self.g.get_vertices()

        for v in self.g.vs:
            mask = np.zeros(self.image.shape[:2], dtype="uint8")

            val = self.getLabelFromId(v.index)
            mask[self.labels == val] = 255

            gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

            indices = []
            for px in cv2.findNonZero(gradient):

                lval = self.labels[px[0][1], px[0][0]]
                if lval != val and lval != 0:
                    indices.append(self.getIdFromLabel(lval))


            neighbors = np.unique(indices)

            edges = np.arange(len(neighbors) * 2).reshape(len(neighbors), 2)
            edges[:, 0] = v.index
            edges[:, 1] = neighbors

            all_edges.append(edges)

        return np.vstack(all_edges)

    def get2DPoints(self, path):
        points = []
        for p in path:
            centroid = self.getCentroidFromLabel(p)
            points.append(tuple([int(centroid[1]), int(centroid[0])]))
        return points

    def get2DPointsInv(self, path):
        points = []
        for p in path:
            centroid = self.getCentroidFromLabel(p)
            points.append(tuple([int(centroid[0]), int(centroid[1])]))
        return points

    def nearestNode(self, point):
        label = self.labels[point[1], point[0]]
        return label

    def shortestPath(self, v1, v2):
        id1 = self.getIdFromLabel(v1)
        id2 = self.getIdFromLabel(v2) 
        path = self.g.get_shortest_paths(v=id1, to=id2)[0]
        path_filt = path[1:]
        return [self.getLabelFromId(p) for p in path_filt]


    def shortestPathNoInt(self, v1, v2):
        id1 = self.getIdFromLabel(v1)
        id2 = self.getIdFromLabel(v2) 
        path = self.g.get_shortest_paths(v=id1, to=id2)[0]      
        return [self.getLabelFromId(p) for p in path if self.g.vs[p]["intersection"] != 1]


    def plotGraph(self, name = None):
        vs_pos = self.g.vs["position"]
        vs_int_up = self.g.vs["intersection"]


        c = []
        for el in vs_int_up:
            if el==0:
                c.append('#FFCF78')
            else:
                c.append("#F04747")

        try:
            for key, val in self.clusters_dict.items():
                for k in self.free_clusters:
                    if k == key:
                        nodes = self.clusters_dict[key]
                        for n in nodes:
                            id = self.getIdFromLabel(n)
                            #c[id] = "#3492eb"
        except:
            print("PlotGraph: clustering not available!")

        visual_style = {}  
        visual_style["vertex_color"] = c
        visual_style["layout"] = vs_pos
        visual_style["vertex_size"] = 40

        if name is not None:
            ig.plot(self.g, name, bbox=(0, 0, 1280, 720), **visual_style)
        else:
            ig.plot(self.g, bbox=(0, 0, 1280, 720), **visual_style)


###################################################################
#
#                   AriadnePath
#
###################################################################

class AriadnePath(object):

    def __init__(self, ariadne, cross_net=None, triplet_net=None, curvature_pred=None, name_file = None, device = "cuda"):
        
        self.device = device
        self.ariadne = ariadne
        self.network = cross_net
        self.network_pred = triplet_net
        self.graph = self.ariadne.graph

        self.curvature_predictor = curvature_pred

        # CANDIDATES
        self.candidates, self.boundary_candidates = self.getCandidates()
        #print("candidates: ", self.candidates, self.boundary_candidates)

        # BOUNDARY CHECK
        #self.boundaryCheck()
        self.candidates, self.boundary_candidates = self.getCandidates()
        
        #print("candidates final: ", self.candidates)

        # LINK CANDIDATES TO CLUSTERS
        self.clusters_candidates_link = self.linkCandidatesToClusters()
        self.inters_candidates = self.getIntersectionCandidates()

        self.free_candidates = self.simplifyFreeCandidates()
        #print("self.clusters_candidates_link ", self.clusters_candidates_link)
        #print("self.inters_candidates ", self.inters_candidates)

        # initialization of weight matrices
        #print("unique_ints_list ", self.graph.unique_ints_list)
        nn_int = []
        for n in self.graph.unique_ints_list:
            nn_int.append(self.graph.getNeighborsFromLabels(n))

        if nn_int:
            nn_int = np.unique(np.hstack(nn_int))
        #print("nn_int ", nn_int)

        self.wm = np.full(shape=(len(nn_int), len(nn_int)), fill_value=-1.0) # weight matrix 
        self.wm_ids = {c: i for i,c in enumerate(nn_int)} # we use this dict to link the rows of the matrix to the nodes
        self.wm_done = np.full(shape=(len(nn_int), len(nn_int)), fill_value=False) # weight matrix 


        ########################
        # COMPUTE PATCHES AND PREDICTIONS FOR ALL
        self.patches = {n: self.getPatch(n) for n in nn_int}
        if self.patches:
            values = self.tripletnetSinglePred(list(self.patches.values()))
            self.outputs = {}
            for i, n in enumerate(nn_int):
                self.outputs[n] = values[i]



    # return list of candidate end-points of cables, identified by their label
    def getCandidates(self):
        degrees = self.graph.getDegrees()
        candidates = np.argwhere(degrees == np.amin(degrees))
        candidates = np.unique(np.hstack(candidates))

        end_points = [self.graph.getLabelFromId(c) for c in candidates]

        boundary_candidates = [v.index for v in ig.VertexSeq(self.graph.g) if self.graph.isBoundary(v) == True]
        boundary_points = [self.graph.getLabelFromId(b) for b in boundary_candidates if b not in candidates]

        return end_points, boundary_points

    def boundaryCheck(self):
        threshold = 0.01
        for b in self.boundary_candidates:

                nn = self.graph.getNeighborsFromLabels(b)
                #print(b, nn)

                if len(nn) != 2:
                    continue

                path_tmp1 = [nn[1], b, nn[0]]
                curvature_score = self.curvature_predictor.computeScore(path_tmp1)          
                visual_score = self.visual_predictor.computeScore(b, path_tmp1[-1])
                comulative_score1 = visual_score * curvature_score  
                
                path_tmp2 = [nn[0], b, nn[1]]
                curvature_score = self.curvature_predictor.computeScore(path_tmp1)          
                visual_score = self.visual_predictor.computeScore(b, path_tmp1[-1])
                comulative_score2 = visual_score * curvature_score  

                if comulative_score1 > comulative_score2: 
                    comulative_score = comulative_score2
                    path_tmp = path_tmp2

                else: 
                    comulative_score = comulative_score1
                    path_tmp = path_tmp1
                                    
                if comulative_score < threshold:
                    id1 = self.graph.getIdFromLabel(b)
                    id2 = self.graph.getIdFromLabel(path_tmp[-1])

                    self.graph.g.delete_edges([(id1, id2)])
                    self.graph.g.delete_edges([(id2, id1)])

                    #print("removing edge: ", id1, id2)


        #self.graph.plotGraph("edges_removed.png")

    def linkCandidatesToClusters(self):
        if debug:
            print("Cluster: ", self.graph.clusters_dict)
        link = {}
        for key, val in self.graph.clusters_dict.items():
            c_queue = []
            for c in self.candidates:
                if c in val:
                    c_queue.append(c)
            link[key] = c_queue 

        return link      

    def getIntersectionCandidates(self):
        
        output = []
        for key, val in self.clusters_candidates_link.items():
            for el in self.graph.inters_clusters:
                if key == el:
                    output.append(val)                
        
        if not output:
            return []
        else:
            return np.hstack(output)

    def simplifyFreeCandidates(self):
        '''
        for each intersection-free cluster, simplify the set of candidates (keep only 1 of the 2)
        '''
        free_candidates = []
        for key0 in self.graph.free_clusters:
            for key, val in self.clusters_candidates_link.items():
                if key == key0:
                    val.pop()
                    if len(val) > 0:
                        free_candidates.append(val[0])
        return free_candidates


    def mainPathFinder(self):

        # nodes neighbours of intersections
        nn_int = [self.graph.getNeighborsFromLabels(n) for n in self.graph.unique_ints_list]

        if nn_int:
            nn_int = list(np.unique(np.hstack(nn_int)))


        ee_nodes = list(self.candidates)

        self.pred_matrix = np.full(shape=(len(nn_int), len(nn_int)), fill_value=-1.0) # predictions matrix 
        self.pred_ids = {c: i for i,c in enumerate(nn_int)} # we use this dict to link the rows of the matrix to the nodes
        self.pred_done = np.full(shape=(len(nn_int), len(nn_int)), fill_value=False) # prediction done matrix 

        self.predictorIntersection()
        #print("PREDICTION MATRIX: \n", self.pred_matrix)

        self.pred_matrix_copy = self.pred_matrix.copy()

        paths_completed = []
        self.paths_tbc = []
        ee_nodes_copy = ee_nodes.copy()
        while len(ee_nodes_copy) > 0:

            path = self.pathFromEndpoint(ee_nodes_copy[0])
            ee_nodes_copy.pop(0)

            if path[-1] in ee_nodes_copy and len(path) > 2:
                paths_completed.append(path)
            elif path[-1] in nn_int:
                self.paths_tbc.append(path)
                nn_int.remove(path[-1])


        while len(nn_int) > 0:

            path = self.pathFromIntNN(nn_int[0])
            nn_int.pop(0)

            if path[-1] in nn_int:
                self.paths_tbc.append(path)
                nn_int.remove(path[-1])


        flag_loop = True
        while flag_loop:
        
            init_path = self.initIntersectionPath()
            if init_path is not None:
                n1, cross, n2 = init_path

                path1 = []
                path2 = []
                for p_tbc in self.paths_tbc:
                    if (n1 in p_tbc):
                        path1 = p_tbc 
                    if (n2 in p_tbc):
                        path2 = p_tbc 
                
                try:
                    self.paths_tbc.remove(path1)
                except:
                    pass

                try:
                    self.paths_tbc.remove(path2)
                except:
                    pass

                if path1:
                    if n1 == path1[0]:
                        path1 = path1[::-1] # reversed
                
                if path2:
                    if n2 == path2[-1]:
                        path2 = path2[::-1] # reversed

                path_merged = path1 + [cross] + path2

                if path_merged[0] in ee_nodes and path_merged[-1] in ee_nodes:
                    paths_completed.append(path_merged)
                else:
                    self.paths_tbc.append(path_merged)

            else:
                flag_loop = False

        #print("COMPLETE: ", paths_completed)
        #print("TMP: ", self.paths_tbc)

        # if elements inside paths_tbc, provide also them as partial solution
        paths_completed = paths_completed + self.paths_tbc

        return paths_completed

    ###########################################################################################
    def pathFromEndpoint(self, v0, condition_angle_enabled=True):

        path_vertices = []
        path_vertices.append(v0)
        
        max_iters = len(self.graph.getVertices())
        it = 0
        while it < max_iters:
            it += 1
            #print("--> ", i)

            tip_v = path_vertices[-1]

            if tip_v in self.candidates and len(path_vertices) > 2: #intersection free clusters return condition
                return path_vertices

            nn = self.graph.getNeighborsFromLabels(tip_v)
            nn_f = [n for n in nn if n not in path_vertices]

            if len(nn_f) == 0: # end of the path reached
                return path_vertices

            elif len(nn_f) == 1: # only one neighbor
                #print("length nn filtered is one!")
                single_nn, = nn_f
                if (self.graph.getIntersectionFromLabel(single_nn) == 0): # no intersection node

                    # *******************************
                    if condition_angle_enabled:
                        path_tmp = [path_vertices[-2], path_vertices[-1], single_nn]
                        angle = self.pathAngle(path_tmp)

                        if angle > 1.5709:
                            path_vertices.append(single_nn)
                        #print(path_vertices)
                        else:
                            return path_vertices
                
                    else:
                        path_vertices.append(single_nn)

                else:
                    return path_vertices #end condition for path completion limit
                  
 
        return path_vertices

    def pathAngle(self, path):
        directions = []
        points = self.graph.get2DPoints(path)
        for i in range(len(points) - 2, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        try:
            angle = math.acos(np.dot(directions[0], directions[1]))
        except:
            angle = 0.0
        return angle


    def pathFromIntNN(self, v0):

        path_vertices = []
        path_vertices.append(v0)
        
        flag_loop = True
        while flag_loop:

            tip_v = path_vertices[-1]

            nn_f = [n for n in self.graph.getNeighborsFromLabels(tip_v) if n not in path_vertices]
            nn_f2 = [n for n in nn_f if self.graph.getIntersectionFromLabel(n) == 0]

            if len(nn_f2) > 0:
                path_vertices.append(nn_f2[0])  
            else:
                flag_loop = False

        return path_vertices


    def initIntersectionPath(self):

        try:
            arg_max = self.pred_matrix_copy.argmax()
        except:
            return None

        if arg_max == 0:
            return None


        id1, id2 = np.unravel_index(self.pred_matrix_copy.argmax(), self.pred_matrix_copy.shape)
        self.pred_matrix_copy[:, id2] = 0
        self.pred_matrix_copy[:, id1] = 0
        self.pred_matrix_copy[id1, :] = 0
        self.pred_matrix_copy[id2,:] = 0    


        n1 = [k for k, v in self.pred_ids.items() if v == id1][0]
        n2 = [k for k, v in self.pred_ids.items() if v == id2][0]

        nn1 = self.graph.getNeighborsFromLabels(n1)
        nn2 = self.graph.getNeighborsFromLabels(n2)

        set_result = set(list(nn1)).intersection(set(list(nn2)))
        if len(set_result) == 0:
            return None

        cross = list(set_result)[0]

        return n1, cross, n2
    

    '''
    def predictorIntersection(self):
        crosses_list = [np.asscalar(el) for el in self.graph.unique_ints_list]
        ints_dict = {n: self.graph.getNeighborsFromLabels(n) for n in crosses_list}

        #print(ints_dict)

        for _, val in ints_dict.items():
        
            combinations_list = list(map(list, itertools.combinations(val, 2)))
        
            for n1, n2 in combinations_list:
                
                nn1 = self.graph.getNeighborsFromLabels(n1)
                nn2 = self.graph.getNeighborsFromLabels(n2)
                (cross, ) = set(list(nn1)).intersection(set(list(nn2)))

                # tripletnet prediction
                pred = self.computeDist(self.outputs[n1], self.outputs[n2])
                pred = 1 - torch.sigmoid(torch.tensor(pred)).float()
                pred = float(pred) 

                #pred = 0.0 if pred > 3 else (3 - pred) / 3 # here we assusme 3 as bound max value for pred

                # curvature prediction
                target_path = []
                if len(nn1) > 1 and len(nn2) > 1:
                    nn1.remove(cross)
                    n1_1, = nn1
                    nn2.remove(cross)
                    n2_2, = nn2
                    target_path = [n1_1, n1, n2, n2_2]
                if len(nn1) > 1:
                    nn1.remove(cross)
                    n1_1, = nn1
                    target_path = [n1_1, n1, n2]
                elif len(nn2) > 1:
                    nn2.remove(cross)
                    n2_2, = nn2
                    target_path = [n1, n2, n2_2]                    
                else:
                    target_path = [n1, cross, n2] 

                curvature_prob = self.curvature_predictor.computeScore(target_path)
                print(pred, curvature_prob)
                # total score
                tot_score = pred * curvature_prob
                #print(n1, n2, tot_score, pred, curvature_prob)

                self.pred_matrix[self.pred_ids[n1], self.pred_ids[n2]] = tot_score
    '''
    def predictorIntersection(self):
        crosses_list = [np.asscalar(el) for el in self.graph.unique_ints_list]
        ints_dict = {n: self.graph.getNeighborsFromLabels(n) for n in crosses_list}

        #print(ints_dict)

        tripletnet_dict = {}
        for _, val in ints_dict.items():
        
            combinations_list = list(map(list, itertools.combinations(val, 2)))

            tmp_dict = {}
            max_pred = 0
            for n1, n2 in combinations_list:
                
                # tripletnet prediction
                pred = self.computeDist(self.outputs[n1], self.outputs[n2])
                pred = 1 - torch.sigmoid(torch.tensor(pred)).float()
                pred = float(pred)

                tmp_dict[(n1, n2)] = pred 
                if pred > max_pred: max_pred = pred

            for key, val in tmp_dict.items():
                tripletnet_dict[key] = val / max_pred


        for _, val in ints_dict.items():
        
            combinations_list = list(map(list, itertools.combinations(val, 2)))
        
            for n1, n2 in combinations_list:

                nn1 = self.graph.getNeighborsFromLabels(n1)
                nn2 = self.graph.getNeighborsFromLabels(n2)

                cross_int = set(list(nn1)).intersection(set(list(nn2)))
                #print(cross_int)
                cross = list(cross_int)[0]

                # curvature prediction

                target_path = []
                '''
                if len(nn1) > 1 and len(nn2) > 1:
                    nn1.remove(cross)
                    n1_1, = nn1
                    nn2.remove(cross)
                    n2_2, = nn2
                    target_path = [n1_1, n1, n2, n2_2]
                if len(nn1) > 1:
                    nn1.remove(cross)
                    n1_1, = nn1
                    target_path = [n1_1, n1, n2]
                elif len(nn2) > 1:
                    nn2.remove(cross)
                    n2_2, = nn2
                    target_path = [n1, n2, n2_2]                    
                else:
                    target_path = [n1, cross, n2] 
                '''
                if cross in nn1: nn1.remove(cross)
                if cross in nn2: nn2.remove(cross)
                if len(nn1) == 1 and len(nn2) == 1:
                    target_path = [nn1[0], n1, n2, nn2[0]]
                elif len(nn1) == 1:
                    target_path = [nn1[0], n1, n2]
                elif len(nn2) == 1:
                    target_path = [n1, n2, nn2[0]]
                else:
                    target_path = [n1, cross, n2] 

                

                curvature_prob = self.curvature_predictor.computeScore(target_path)

                if len(target_path) > 2: # normalization
                    curvature_prob /= len(target_path) - 2
                else:
                    print("error: target path len less than 2!")
                #print(nn1, nn2, cross)
                #print(target_path)

                # total score
                tot_score = tripletnet_dict[(n1, n2)] * curvature_prob
                #print(n1, n2, tot_score, tripletnet_dict[(n1, n2)], curvature_prob)
                #print("   ")

                self.pred_matrix[self.pred_ids[n1], self.pred_ids[n2]] = tot_score





    def getPathsDict(self):
        return self.paths_dict


    def getPatch(self, label, crop_size=32):

        x, y = self.graph.getCentroidFromLabel(label)

        img = self.ariadne.image_masked.copy()
        img = Image.fromarray(img)

        crop = x - crop_size // 2, y - crop_size // 2, x + crop_size // 2, y + crop_size // 2,

        cropped = img.crop(crop)
        cropped = np.array(cropped)

        return cropped
    
    def tripletnetSinglePred(self, patches):
        

        # to tensor
        samples1 = [torch.tensor(p / 255.).permute(2, 0, 1).float().unsqueeze(0) for p in patches]

        sample1 = torch.cat(samples1, dim=0)
        sample1 = sample1.to(self.device)

        # Compute network output
        output = self.network_pred(sample1).cpu().detach().numpy()
        
        return output

        
    def computeDist(self, a,b):
        return np.sum(np.square(a-b))

    def draw(self, img, path, color=(0, 255, 0), thickness=2, seed_diff_color=True):
        color = (int(color[0]), int(color[1]), int(color[2]))
        for i in range(1, len(path)):
            if i == 1 and seed_diff_color:
                color_circle = (255, 255, 0)
            else:
                color_circle = color

            p1 = self.graph.getCentroidFromLabel(path[i - 1])
            p2 = self.graph.getCentroidFromLabel(path[i])
            cv2.circle(img, tuple(p1), color=color_circle,
                       radius=thickness + 2, thickness=-1)
            cv2.circle(img, tuple(p2), color=color_circle,
                       radius=thickness + 2, thickness=-1)
            cv2.line(img, tuple(p1), tuple(p2),
                     color=color, thickness=int(thickness))


    def drawMask(self, paths, colors):
        
        masks_colored = []
        for it, path in enumerate(paths):
            mask = np.zeros((self.ariadne.image.shape[0], self.ariadne.image.shape[1]), dtype=np.uint8)
            mask = self.ariadne.image_mask.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            if it >= len(colors):
                ii = it - len(colors)
            else:
                ii = it
                
            color = colors[ii]
            color = (int(color[0]), int(color[1]), int(color[2]))

            for p in path:
                if self.graph.getIntersectionFromLabel(p) == 0:
                    mask[self.graph.labels == p] = color
            
            masks_colored.append(mask)
            
        mask = np.zeros((self.ariadne.image.shape[0], self.ariadne.image.shape[1]), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        for img in masks_colored:
            mask = mask + img
        return mask


    ###################################################################
    # ------> CROSSNET - predict which wire is above
    ###################################################################
    def crossnetPredFast(self, paths, masks, crop_size=64):

        # get crosses in the graph
        crosses = self.graph.unique_ints_list
        crosses_list = [np.asscalar(el) for el in crosses]

        paths_dict = {i: p for i, p in enumerate(paths)}

        # loop through each cross node
        all_crosses_dict = {cross: {} for cross in crosses_list}
        for cross in crosses_list:

            cross_crops = {}

            x, y = self.graph.getCentroidFromLabel(cross) # centroid of cross

            for key, val in paths_dict.items():
                if cross in val:
                    img = Image.fromarray(self.ariadne.image.copy())
                    img_mask = Image.fromarray(masks[key].copy()) # spline mask of the path considered

                    # crop bounding-box
                    crop = x - crop_size // 2, y - crop_size // 2, x + crop_size // 2, y + crop_size // 2,

                    cropped = np.array(img.crop(crop))
                    cropped_mask =  np.array(img_mask.crop(crop)) 
                    cropped[cropped_mask < 127] = [0, 0, 0]

                    cross_crops[key] = cropped

            all_crosses_dict[cross] = cross_crops

        # make predictions
        cropss_all = []
        for key, _ in all_crosses_dict.items():
            for _, vval in all_crosses_dict[key].items():
                cropss_all.append(vval)

        #print(len(cropss_all))
        preds = self.crossNetInference(cropss_all)
        #print(preds)

        it = 0
        preds_all = {}
        for key, cross_dict in all_crosses_dict.items():
            preds_cross = {}
            for kkey, _ in cross_dict.items():
                preds_cross[kkey] = preds[it]
                it += 1
            preds_all[key] = preds_cross
        
        #print(preds_all)

        # make selection
        indices_dict = {i: 0 for i, _ in enumerate(paths)}
        for cross, d in preds_all.items():
            
            # get index of path with highest prediction
            v=list(d.values())
            k=list(d.keys())

            if len(v) != 0:
                path_id = k[v.index(max(v))]
                indices_dict[path_id] += 1 # increment counter of "above"

                # keep cross node only on found "path_id" and remove from all the others
                k.remove(path_id)
                for id in k:
                    paths[id].remove(cross)
        
        # sort dict by value from highest, return list of values
        indeces_dict = {k: v for k, v in sorted(indices_dict.items(), key=lambda item: item[1], reverse=False)}
        #print(indeces_dict)
        # order paths based on indeces_dict.values()
        #paths = [x for x,_ in sorted(zip(paths,indeces_dict.values()))]
        return indeces_dict

    def crossNetInference(self, crops_list):

        samples = [torch.tensor(crops_list[i] / 255.).permute(2, 0, 1).float().unsqueeze(0) for i in range(len(crops_list))]

        if samples:
            samples = torch.cat(samples, dim=0)
            samples = samples.to(self.device)
                
            # compute network output
            outputs = self.network(samples)

            return [torch.sigmoid(v).float().item() for v in outputs]        
        else:
            return None
