from scipy.interpolate import splprep, splev
import numpy as np
import cv2
from matplotlib import cm



class SplineMask():

    def __init__(self, ariadne, k=3, smoothing=0.0, periodic=0):
        self.ariadne = ariadne
        self.k = k
        self.smoothing = smoothing
        self.periodic = periodic

    def computeSpline(self, path):
        points = self.ariadne.graph.get2DPointsInv(path)
        if len(points) > self.k:
            pts = np.array(points)
            tck, u = splprep(pts.T,
                             u=None,
                             k=self.k,
                             s=self.smoothing,
                             per=self.periodic
                             )
            return {'tck': tck, 'u': u}
        return None

    def generateSplineMaskImage(self, path, radius=4, steps=1000, ref_image=None):
        if ref_image is None:
            ref_image = self.ariadne.image

        mask = np.zeros(
            (ref_image.shape[0], ref_image.shape[1]), dtype=np.uint8)
        spline = self.computeSpline(path)
        if spline is None:
            return mask
        u = spline['u']
        tck = spline['tck']
        u_new = np.linspace(u.min(), u.max(), steps)
        x_new, y_new = splev(u_new, tck, der=0)
        for i, x in enumerate(x_new):
            cv2.circle(mask, (int(x_new[i]), int(
                y_new[i])), radius, (255, 255, 255), -1)
        return mask

    
    def computeRadius(self, path):

        median = []
        for p in path:
            mask = np.zeros((self.ariadne.image.shape[0], self.ariadne.image.shape[1]), dtype=np.uint8)
            
            mask[self.ariadne.labels == p] = 255
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts[0]) < 5:
                continue

            ellipse = cv2.fitEllipse(cnts[0])

            if ellipse[1][1] > ellipse[1][0]:
                median.append(ellipse[1][0])
            else:
                median.append(ellipse[1][1])

        median.sort()
        diameter = median[int(len(median) / 2)]
        return int(diameter / 2)
    
    def generateImageLabels(self, paths, colors, radius=4, steps=1000):
        ref_image = self.ariadne.image
        mask = np.zeros((ref_image.shape[0], ref_image.shape[1]), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        for it, p in enumerate(paths):
            if it >= len(colors):
                ii = it - len(colors)
            else:
                ii = it

            color = colors[ii]
            color = (int(color[0]), int(color[1]), int(color[2]))

            mradius = self.computeRadius(p)

            spline = self.computeSpline(p)
            if spline is None:
                continue

            u = spline['u']
            tck = spline['tck']
            u_new = np.linspace(u.min(), u.max(), steps)
            x_new, y_new = splev(u_new, tck, der=0)
            for i, x in enumerate(x_new):
                cv2.circle(mask, (int(x_new[i]), int(y_new[i])), mradius, color, -1) 

        return mask

    def generateSingleLabels(self, paths, steps=1000):
        ref_image = self.ariadne.image

        masks_dict = {}
        for it, p in enumerate(paths):
            mask = np.zeros((ref_image.shape[0], ref_image.shape[1]), dtype=np.uint8)

          
            mradius = self.computeRadius(p)
            spline = self.computeSpline(p)
            if spline is None:
                for n in p:
                    mask[self.ariadne.new_labels == n] = 255
                
                masks_dict[it] = mask
                continue

            u = spline['u']
            tck = spline['tck']
            u_new = np.linspace(u.min(), u.max(), steps)
            x_new, y_new = splev(u_new, tck, der=0)

            for i, x in enumerate(x_new):
                cv2.circle(mask, (int(x_new[i]), int(y_new[i])), mradius, (255, 255, 255), -1) 

                masks_dict[it] = mask
            mask = np.zeros((ref_image.shape[0], ref_image.shape[1]), dtype=np.uint8)

        return masks_dict

    def drawFinalMaskSpline(self, paths, order_dict):  
        mask = self.ariadne.image_mask.copy()
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        colors = cm.get_cmap('tab20', 20)


        # draw from the top cable
        it = 0
        for key, _ in order_dict.items():

            c = colors(it)
            color = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
            it += 1        

            path = paths[key]
        
            mradius = self.computeRadius(path)
            spline = self.computeSpline(path)
            
            if spline is not None:
                u = spline['u']
                tck = spline['tck']
                u_new = np.linspace(u.min(), u.max(), 1000)
                x_new, y_new = splev(u_new, tck, der=0)

                for i, x in enumerate(x_new):
                    cv2.circle(mask, (int(x_new[i]), int(y_new[i])), mradius, (color[0], color[1], color[2]) , -1) 
                
            for p in path:
                if self.ariadne.graph.getIntersectionFromLabel(p) == 0:
                    mask[self.ariadne.graph.labels == p] = color

        
        mask[self.ariadne.image_mask == 0] = [0, 0, 0]
        mask[np.where((mask==[255,255,255]).all(axis=2))]=[0,0,0]

        return mask



class Spline():

    def __init__(self, ariadne, k=3, smoothing=0.0, periodic=0):
        self.ariadne = ariadne
        self.k = k
        self.smoothing = smoothing
        self.periodic = periodic

    def computeSpline(self, path):
        points = self.ariadne.graph.get2DPointsInv(path)
        if len(points) > self.k:
            pts = np.array(points)
            tck, u = splprep(pts.T,
                             u=None,
                             k=self.k,
                             s=self.smoothing,
                             per=self.periodic
                             )
            return {'tck': tck, 'u': u}
        return None

  
    def generatePathsSplines(self, paths, num_points=10, debug=False):
        points = []

        if debug:
            img_copy = self.ariadne.image.copy()
        
        for it, p in enumerate(paths):

            spline = self.computeSpline(p)
            if spline is None:
                continue

            u = spline['u']
            tck = spline['tck']
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck, der=0)

            x_approx = [int(x) for x in x_new] 
            y_approx = [int(y) for y in y_new] 
            points.append(list(zip(x_approx, y_approx)))
            if debug:
                for i, x in enumerate(x_new):
                    cv2.circle(img_copy, (int(x_new[i]), int(y_new[i])), 7, (255, 255, 0), -1) 

        if debug:
            cv2.imshow("discrete_spline_points", 0)
            cv2.imshow("discrete_spline_points", img_copy)
            
            cv2.imwrite("spline_points.png", img_copy)
            cv2.waitKey(0)



        return points

    def genereteOutputSplinesMsg(self, paths):
        tck_array = []

        for it, p in enumerate(paths):

            spline = self.computeSpline(p)
            if spline is None:
                continue

            u = spline['u']
            t, c, k = spline['tck']

            cx = c[0]
            cy = c[1]					
            tck_array.append([t, cx, cy, k])

        return tck_array