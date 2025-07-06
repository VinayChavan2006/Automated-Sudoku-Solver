import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from digit_predictor import MNIST_CNN
import torch
from sudoku_solver import solveSudoku
import matplotlib.pyplot as plt
import urllib.request


def imread_from_url(url):
    resp = urllib.request.urlopen(url)
    image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # or IMREAD_GRAYSCALE
    return img

def preprocessing(img):
    img = img.copy()
    # convert image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussain blur
    blurred = cv2.GaussianBlur(gray_img, (7,7), 0)

    # apply adaptive thresholding
    thres_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thres_img = cv2.bitwise_not(thres_img)
    return thres_img

def detect_sudoku_square(img, orig_image):
    contours, heirarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudokuContour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        # approx wrapper
        approx_poly = cv2.approxPolyDP(c,0.02 * perimeter, True)
        # print(approx_poly, len(approx_poly))
        # break
        if len(approx_poly) == 4:
            sudokuContour = approx_poly
            break
    # print(contours)
    cv2.drawContours(orig_image, [sudokuContour], -1, (255, 255, 255), 2)
    cv2.imshow("bfhjf", orig_image)
    cv2.waitKey(0)
    sudoku = four_point_transform(orig_image, sudokuContour.reshape(4,2))
    return sudoku, sudokuContour
def draw_contours_on_image(image, contour):
    # Make a copy so original isn't modified
    contour_img = image.copy()
    cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), 2)
    return contour_img



def extract_cells(sudoku_img, points):
    cells = []
    # Ensure consistent preprocessing size
    gray = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)

    # Enhance contrast
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    cell_h, cell_w = 28, 28  # Final size for NN
    for i in range(9):
        for j in range(9):
            x, y = j * 28, i * 28
            cell = adaptive[y:y+28, x:x+28].copy()

            # Remove border artifacts
            cell = clear_border(cell)

            # Resize for consistent shape (safe-guard)
            cell = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            # Normalize for NN input
            cell = cell.astype(np.float32) / 255.0
            cells.append(cell)
    return np.array(cells)


def getSudokuGrid(cells):
    sudoku_grid = []
    isModelLoaded = False
    # print(cells[0])
    for cell in cells:        
        if(np.sum(cell == 1)/(28*28) < 0.01):
            sudoku_grid.append('.')
        else:
            if not isModelLoaded:
                plt.imshow(cell)
                cv2.imshow(f"cell_{isModelLoaded}", cell)
                cv2.waitKey(0)
                model = MNIST_CNN(in_channels=1, num_classes=10)
                model.load_state_dict(torch.load("./model/mnist_cnn_weights.pth", map_location=torch.device('cpu')))
                model.eval()

                isModelLoaded = True
            plt.imshow(cell)
            cv2.imshow(f"cell_{isModelLoaded}", cell)
            img_tensor = torch.tensor(cell)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = output.argmax(1).item()
            print(str(predicted_class))
            sudoku_grid.append(str(predicted_class))
    sudoku_grid = np.array(sudoku_grid).reshape(9,9)
    print(sudoku_grid)
    return sudoku_grid


# board = solveSudoku(board)
# print(board)

image = cv2.imread("images/digits.jpg")
# image = imread_from_url("https://ph-static.z-dn.net/files/d4f/58849763dabfdfc22e6d24b74bd06160.jpg")
image = np.array(image)
processed_img = preprocessing(image)
sudoku, sudoku_points = detect_sudoku_square(processed_img, image)
cells = extract_cells(sudoku, sudoku_points)

getSudokuGrid(cells)
cv2.imshow("grid", sudoku)
cv2.waitKey(0)
