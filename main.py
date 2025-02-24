import time
import random

def generate_matrix(size): # Function to generate a random matrix (will be used for matrix multiplication)
    return [[random.random() for _ in range(size)] for _ in range(size)]

def zero_matrix(size): # Function to initialize a zero matrix (results of the multiplication will go here)
    return [[0.0 for _ in range(size)] for _ in range(size)]


def matrix_multiplication_loop_unrolling(size): # Function to perform matrix multiplication with loop unrolling
    A = generate_matrix(size) # Generate first matrix
    B = generate_matrix(size) # Generate second matrix
    C = zero_matrix(size)

    unroll_factor = 8
    
    start_time = time.time()
    
    for i in range(size):
        for j in range(size):
            temp = 0.0
            k = 0
            # Unrolled loop
            while k <= size - unroll_factor:
                # we'll execute 4 operations as indicated by the unroll factor in each iteration for this matrix multiplication
                temp += A[i][k] * B[k][j]
                temp += A[i][k+1] * B[k+1][j]
                temp += A[i][k+2] * B[k+2][j]
                temp += A[i][k+3] * B[k+3][j]
                temp += A[i][k+4] * B[k+4][j]
                temp += A[i][k+5] * B[k+5][j]
                temp += A[i][k+6] * B[k+6][j]
                temp += A[i][k+7] * B[k+7][j]
                k += unroll_factor # Increase by the unroll factor to correctly iterate through the loop.
            # Process remaining elements
            while k < size:
                temp += A[i][k] * B[k][j]
                k += 1
            C[i][j] = temp
    
    end_time = time.time()
    
    return end_time - start_time # return execution time to focus on performance

def matrix_multiplication_standard_loop(size): # Standard matrix multiplication without loop unrolling
    A = generate_matrix(size)
    B = generate_matrix(size)
    C = zero_matrix(size)
    
    start_time = time.time()
    for i in range(size):
        for j in range(size):
            temp = 0.0
            for k in range(size):
                # Here we only execute one operaion for each iteration.
                temp += A[i][k] * B[k][j]
            C[i][j] = temp
    end_time = time.time()
    return end_time - start_time # return execution time to focus on performance

# Execution time comparison
size = 100
time_standard = matrix_multiplication_standard_loop(size)
time_unrolled = matrix_multiplication_loop_unrolling(size)

# Display the results
print("Technique                Execution Time (s)")
print(f"Standard Loop           {time_standard:.4f}")
print(f"Loop Unrolling          {time_unrolled:.4f}")