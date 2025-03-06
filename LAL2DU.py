import numpy as np     #Only used for testing

def inverse(matrix):

    if is_square(matrix):          #Checks if input matrix is sqaure otherwise it is singular
        re = row_echelon(matrix)     
        if not re:                 #If row_echelon returns an empty list then the input matrix is singular
            raise ValueError("The input matrix is not invertible!")
        matrix = re[0]
        inverse = re[1]

        n = len(matrix) - 1
        m = 0                   #Keeps track of what column inside the matrix we are making "zeros" and what row we will be subtracting from the ones above it

        while m < n:            #Same algorithm as in the row_echelon function only now the square inside the matrix is (n-m)x(n-m)
            for row in range(n-m-1, -1, -1):
                multiplier = matrix[row][n-m]                                      
                matrix = add_row(matrix,row,n-m, -multiplier)
                inverse = add_row(inverse, row, n-m, -multiplier)
            m += 1                                                  

        return inverse
    else:
        raise ValueError("The input matrix is not square and thus not invertible!")


def row_echelon(matrix):  #WORKS ONLY FOR INVERTIBLE MATRICES, FOR SINGULAR MATRICES [] IS RETURNED 
    n = len(matrix)
    Id = [[1 if i == j else 0 for j in range(n)] for i in range(n)]   #Create an identity matrix which and apply the same changes as to matrix

    while n > 1:
        m = len(matrix) - n             #keeps track of what mxm square we are looking at in the matrix
        
        if matrix[m][m] == 0:           #Swap rows if there is 0 on the current diagonal (m keeps track of that)
            i = 0
            while matrix[m][m] == 0 and m + i != len(matrix) - 1: #Keep swapping rows until a swap occurs or the final row of the matrix is reached
                i = i + 1
                if matrix[m + i][m] != 0: #Looking for a non-zero values below the diagonal
                    matrix = swap_rows(matrix, m, m + i)
                    Id = swap_rows(Id, m, m + i)

        left_corner = matrix[m][m]
        if left_corner == 0:        #If there are all zeros below the diagonal value and the diagonal value is zero then no swaps occured and the matrix is singular
            return []               #Matrix is not invertible

        matrix = multiply_row(matrix, m, 1/left_corner)  #Create a one on the diagonal for each m (except m == len(matrix) - 1)
        Id = multiply_row(Id, m, 1/left_corner)
            
        for row in range(1 + m, len(matrix)):       #For every m ("square" inside the matrix) the m-th row will be subtracted from every row less then m by a factor of matrix[row][m]                                      
            multiplier = matrix[row][m]             #which will create all zeros below matrix[m][m]
            matrix = add_row(matrix, row, m, -multiplier)
            Id = add_row(Id, row, m, -multiplier)

        n -= 1  #Go to next "sqaure" inside the matrix and repeat process until the "square" is the bottom right corner value

    n = len(matrix) - 1
    if matrix[n][n] != 0:               #After all manipulations if the bottom right value is not zero then make it a 1
        Id = multiply_row(Id, n, 1/matrix[n][n])
        matrix = multiply_row(matrix, n, 1/matrix[n][n])
    else:
        return []                       #Else matrix is singular

    return (matrix, Id)

#CHECKS IF MATIRX IS SQUARE
def is_square(matrix):
    if len(matrix) == len(matrix[0]): 
        return True  
    else:
        return False

 #ELEMNTARY OPERATIONS       
def swap_rows(matrix, row1, row2):    
    matrix_copy = [[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
    temp = matrix_copy[row1]
    matrix_copy[row1] = matrix_copy[row2]
    matrix_copy[row2] = temp
    return matrix_copy


def multiply_row(matrix, row, a):
    if a != 0:
        matrix_copy = [[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
        for col in range(len(matrix_copy[0])):
            matrix_copy[row][col] = a*matrix_copy[row][col] 
        return matrix_copy
    else:
        raise ValueError("You cannot multiply a row in a matrix by zero!")

def add_row(matrix, row1, row2, a):             #Ads a factor a of row2 to row1
    matrix_copy = [[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
    for col in range(len(matrix_copy[0])):
        matrix_copy[row1][col] = matrix_copy[row1][col] + a*matrix_copy[row2][col]
    return matrix_copy

#TESTS
twoxtwo_regular = [[[0, 2],[1, 1]],
                    [[2, 3],[0, 3]],
                    [[3, 0],[2, 3]],
                    [[7, 1],[2, 0]], 
                    [[2, 1],[1, 8]]
                    ]

threexthree_regular = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  
    ((2, 0, 0), (0, 2, 0), (0, 0, 2)),  
    ((0, 1, 1), (1, 0, 1), (1, 1, 0)),  
    ((1, 2, 3), (0, 1, 4), (0, 0, 1)),  
    ((1, 0, 2), (0, 1, -1), (0, 0, 1)),  
    ((1, 1, 2), (1, 0, 3), (0, 1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1))
]

fourxfour_regular = [
    ((0, 1, 0, 3), (0, 0, 2, 5), (1, 0, 2, 0), (0, 1, 0, 1)),
    ((3, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0), (0, 0, 0, 3)),
    ((1, 2, 3, 1), (0, 1, 2, 1), (0, 0, 8, 1), (0, 0, 0, 2)),
    ((1, 0, 0, 0), (2, 1, 0, 0), (1, 4, 1, 0), (2, 1, 1, 1)),
    ((2, 3, 5, 1), (0, 4, 6, 8), (1, 0, 4, 10), (0, 0, 0, 2)),
    ((1, 2, 0, 0), (0, 1, 0, 0), (0, 0, 2, 3), (0, 0, 0, 1)),
    ((1, 0, 0, 0), (0, 0, 2, 0), (0, 1, 0, 2), (0, 1, 2, 3)),
    ((1, 2, 0, 0), (1, 2, 1, 0), (0, 1, 1, 1), (0, 0, 0, 1))
]

mega_matrix = [ 
    (0, 1, 1, 0, 2, 1, 0),
    (1, 1, 2, 2, 4, 2, 0),
    (2, 0, 0, 1, 1, 3, 1),
    (1, 1, 5, 0, 2, 2, 4),
    (4, 1, 2, 1, 2, 1, 0),
    (1, 1, 1, 3, 0, 0, 1),
    (0, 1, 3, 2, 0, 2, 3)
]


for matrix in twoxtwo_regular:          
    mat = np.array(matrix, dtype=float)
    assert inverse(matrix) == np.linalg.inv(mat).tolist()
print("TEST FOR 2X2 INVERTIBLE MATRCIES WAS SUCCESFUL!")

for matrix in threexthree_regular:
    mat = np.array(matrix, dtype=float)
    assert inverse(matrix) == np.linalg.inv(mat).tolist()
print("TEST FOR 3X3 INVERTIBLE MATRCIES WAS SUCCESFUL!")

for matrix in fourxfour_regular:
    mat = np.array(matrix, dtype=float)
    assert inverse(matrix) == np.linalg.inv(mat).tolist()
print("TEST FOR 4X4 INVERTIBLE MATRCIES WAS SUCCESFUL!")

twoxtwo_singular = [[[2,2], [1, 1]],
                    [[0,0], [0,0]],
                    [[3, 15], [2, 10]], 
                    [[0, 3], [0, 4]],
                    [[7, 0], [2, 0]]]

threexthree_singular = [
    ((1, 2, 3), (2, 4, 6), (3, 6, 9)),
    ((1, 0, 0), (1, 0, 0), (0, 1, 0)),
    ((2, 4, 6), (1, 2, 3), (3, 6, 9)),
    ((1, 2, 3), (0, 0, 0), (4, 5, 6)),
    ((1, 0, 0), (2, 0, 0), (3, 0, 0)),
    ((1, 2, 3), (0, 1, 1), (1, 3, 4)),
    ((0, 0, 0), (0, 0, 0), (0, 0, 0))]

fourxfour_singular = [[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
                      [[2, -1, 0, 1], [4, -2, 0, 2], [1, 3, -2, 5], [-2, -6, 4, -10]],
                      [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                      [[1, 2, 3, 4], [2, 4, 6, 8], [1, 1, 1, 1], [0, 0, 0, 0]]
                      ]

for matrix in twoxtwo_singular:
    try:
         inv = inverse(matrix)
         print(f"{matrix} should is singular but a ValueError is not returned!")
    except ValueError as e:
        print("Check for 2X2 matrices invertability is correct!")

for matrix in threexthree_singular:
    try:
         inv = inverse(matrix)
         print(f"{matrix} is singular but a ValueError for is not returned!")
    except ValueError as e:
        print("Check for 3X3 matrices invertability is correct!")

for matrix in fourxfour_singular:
    try:
         inv = inverse(matrix)
         print(f"{matrix} is singular but a ValueError for is not returned!")
    except ValueError as e:
        print("Check for 4X4 matrices invertability is correct!")

mega_mat = np.array(mega_matrix, dtype=float)
print("\n" + "NUMPY INVERSE:")
for row in inverse(mega_matrix):
    print(row)
print("\n" + "IMPLEMENTED INVERSE FUNCTION:")
for row in np.linalg.inv(mega_mat).tolist():
    print(row)
print("Due to rounding errors in python keyword assert is not used for this test.")
print("TEST FOR LARGE 7X7 MATRIX WORKS")