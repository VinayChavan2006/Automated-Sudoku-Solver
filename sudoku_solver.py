import numpy as np


def isValid(board, i, j, val):
    for col in range(9):
        if col != j and board[i][col] == val:
            return False
    for row in range(9):
        if row != i and board[row][j] == val:
            return False

    mini_grid_start_i = (i // 3) * 3
    mini_grid_start_j = (j // 3) * 3
    for p in range(3):
        for q in range(3):
            r = mini_grid_start_i + p
            c = mini_grid_start_j + q
            if (r != i or c != j) and board[r][c] == val:
                return False
    return True

def solve(board, i, j):
    if j > 8:
        i += 1
        j = 0
    
    if i == 9:
        return True 

    if board[i][j] == '.':
        for val in range(1, 10):
            if isValid(board, i, j, str(val)):
                board[i][j] = str(val)
                if solve(board, i, j + 1): 
                    return True
                board[i][j] = '.' 
        return False  
    else:
        return solve(board, i, j + 1)

def solveSudoku(board):
    solve(board, 0, 0)
    return board

# Input board
board = [["5","3",".",".","7",".",".",".","."],
         ["6",".",".","1","9","5",".",".","."],
         [".","9","8",".",".",".",".","6","."],
         ["8",".",".",".","6",".",".",".","3"],
         ["4",".",".","8",".","3",".",".","1"],
         ["7",".",".",".","2",".",".",".","6"],
         [".","6",".",".",".",".","2","8","."],
         [".",".",".","4","1","9",".",".","5"],
         [".",".",".",".","8",".",".","7","9"]]

# solved = solveSudoku(board)

# # Print result
# for row in solved:
#     print(row)
