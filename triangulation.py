from lib.Delaunator import Delaunator
import cv2
import numpy as np
import random
from cvzone.FaceMeshModule import FaceMeshDetector

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )
 
def draw_delaunay(img, subdiv, delaunay_color) :
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in subdiv.getTriangleList():
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 
def draw_voronoi(img, subdiv) :
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
 
        ifacet = np.array(ifacet_arr, dtype=np.intp)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)

def get_control_points(face, out, draw=True):
    detector = FaceMeshDetector(maxFaces=3)
    _, faces = detector.findFaceMesh(face, draw=draw)
    if faces:
        for i in faces[0]:
            out.append((i[0], i[1]))
    del detector
    del _
    del faces

def check_in_screen(p, size):
    if p[0] >= size[1]:
        return False
    if p[0] < 0 or p[1] < 0:
        return False
    if p[1] >= size[0]:
        return False
    return True

def catch_face(img):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    points = []
    get_control_points(img, points)
    for p in points :
        if not check_in_screen(p, size):
            continue
        subdiv.insert(p)
    draw_delaunay( img, subdiv, (255, 255, 255))
    for p in points :
        draw_point(img, p, (0,0,255))
    del points
    del subdiv
    del size
    del rect

def create_del_vor(path, win_d, animate=False):
    img = cv2.imread(f"faces/{path}")

    img_orig = img.copy()
 
    size = img.shape
    rect = (0, 0, size[1], size[0]) 
    subdiv = cv2.Subdiv2D(rect)

    points = []
    get_control_points(img, points)
 
    for p in points :
        subdiv.insert(p)
 
        if animate :
            img_copy = img_orig.copy()
            draw_delaunay( img_copy, subdiv, (255, 255, 255))
            cv2.imshow(win_d, img_copy)
            cv2.waitKey(100)
    draw_delaunay( img, subdiv, (255, 255, 255))

    for p in points :
        draw_point(img, p, (0,0,255))
 
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)
    draw_voronoi(img_voronoi,subdiv)

    return img, img_voronoi, points


def applyAffineTransform(src, srcTri, dstTri, size) :
    # Вычисляется аффиная трансформация для двух треугольников
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Добавляется на img
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    r1 = cv2.boundingRect(np.array([t1], dtype=np.float32))
    r2 = cv2.boundingRect(np.array([t2], dtype=np.float32))
    r = cv2.boundingRect(np.array([t], dtype=np.float32))

    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def create_morph(img1, img2, imgMorph, alpha, coords, coords2):
    for i in range(min(len(coords), len(coords2))):
        t1 = [(coords[i][0][0], coords[i][0][1]), (coords[i][1][0], coords[i][1][1]), (coords[i][2][0], coords[i][2][1])]
        t2 = [(coords2[i][0][0], coords2[i][0][1]), (coords2[i][1][0], coords2[i][1][1]), (coords2[i][2][0], coords2[i][2][1])]

        morph_t = []
        for j in range(3):
            x = (1 - alpha) * t1[j][0] + alpha * t2[j][0]
            y = (1 - alpha) * t1[j][1] + alpha * t2[j][1]
            morph_t.append((x, y))

        morphTriangle(img1, img2, imgMorph, t1, t2, morph_t, alpha)

def create_cords(pts, triangles):
    coords = []
    for i in range(0, len(triangles), 3):
        coords.append([
            pts[triangles[i]],
            pts[triangles[i+1]],
            pts[triangles[i+2]]
        ])
    return coords

def screen_normalizator(*images):
    if isinstance(images[0], list):
        images = images[0]
    imgMorph = np.zeros((max(images[0].shape[0], images[1].shape[0]), max(images[1].shape[1],images[1].shape[1]), images[0].shape[2]), dtype=images[0].dtype)
    max_w, max_h = 0, 0
    for img in range(len(images)-1):
        imgMorph = np.zeros((max(images[img].shape[0], images[img+1].shape[0], max_h), max(images[img].shape[1],images[img+1].shape[1], max_w), images[img].shape[2]), dtype=images[img].dtype)
        if max_h < imgMorph.shape[0]:
            max_h = imgMorph.shape[0]
        if max_w < imgMorph.shape[1]:
            max_w = imgMorph.shape[1]

    return imgMorph

def create_morphs(*paths):
    window = "Morphs"
    cv2.namedWindow(window, cv2.WINDOW_GUI_NORMAL)

    images = []
    for path in paths:
        images.append(cv2.imread(f'faces/{path}'))
    alphas = np.linspace(0, 1, 50)
    imgMorph = screen_normalizator(images)

    points = []
    for img in images:
        pts = []
        get_control_points(img, pts, False)
        points.append(pts)

    triangles = Delaunator(points[0]).triangles
    coords = []

    for pts in points:
        coords.append(create_cords(pts, triangles))
    
    for index_img in range(len(images)-1):
        for alpha in alphas:
            imgMorph.fill(0)
            create_morph(images[index_img], images[index_img+1], imgMorph, alpha, coords[index_img], coords[index_img+1])

            cv2.imshow(window, imgMorph)
            cv2.waitKey(1)
    cv2.waitKey(0)
    
def main(path1, path2, alpha, using_alpha=False, animate=False):
    img1 = cv2.imread(f'faces/{path1}')
    img2 = cv2.imread(f'faces/{path2}')
    win_delaunay1 = "IMG1 Delaunay Triangulation"
    win_voronoi1 = "IMG1 Voronoi Diagram"
    win_delaunay2 = "IMG2 Delaunay Triangulation"
    win_voronoi2 = "IMG2 Voronoi Diagram"
    win_morphed = "IMG Morphed"
    cv2.namedWindow(win_morphed, cv2.WINDOW_GUI_NORMAL)

    alphas = np.linspace(0, 1, 50)
    image, image_voronoi, pts = create_del_vor(f"{path1}", win_delaunay1, animate)
    image2, image_voronoi2, pts2 = create_del_vor(f"{path2}", win_delaunay2, animate)

    imgMorph = screen_normalizator(img1, img2)

    triangles = Delaunator(pts).triangles
    coords = create_cords(pts, triangles)
    coords2 = create_cords(pts2, triangles)

    if not using_alpha:
        for alpha in alphas:
            imgMorph.fill(0)
            create_morph(img1, img2, imgMorph, alpha, coords, coords2)

            cv2.imshow(win_morphed, imgMorph)
            cv2.waitKey(1)
    else:
        create_morph(img1, img2, imgMorph, alpha, coords, coords2)

        cv2.imshow(win_morphed, imgMorph)
        cv2.waitKey(1)
        
    cv2.waitKey(0)

    cv2.namedWindow(win_delaunay1,cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(win_delaunay2,cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(win_voronoi1,cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(win_voronoi2,cv2.WINDOW_GUI_NORMAL)

    cv2.imshow(win_delaunay1, image)
    cv2.imshow(win_voronoi1, image_voronoi)
    cv2.imshow(win_delaunay2, image2)
    cv2.imshow(win_voronoi2, image_voronoi2)

    cv2.waitKey(0)

if __name__ == '__main__':
    animate = False

    file_name1 = "good_man.jpg"
    file_name2 = "giga_man.jpg"
    file_name3 = "Girl1.jpg"
    file_name4 = "Girl2.jpg"
    file_name5 = "girl.jpg"
    file_name6 = "putin.jpg"
    file_name7 = "rock.jpeg"
    file_name8 = "sveta1.jpg"
    file_name9 = "sveta2.jpg"
    # main(file_name1, file_name3, 0.5,using_alpha=False, animate=animate)
    create_morphs(file_name1, file_name2, file_name5, file_name6, file_name7)
