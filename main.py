import numpy, random, math
from scipy . optimize import minimize
import matplotlib . pyplot as plt

#################ASSIGMENT 1####################################
def linear_kernel(x, y):
    return numpy.dot(x, y)

def poly_kernel(x, y, p):
    return math.pow((numpy.dot(x, y) + 1), p)

def radial_basis_function_kernel(x, y, smooth):
    x_minus_y = numpy.subtract(x, y)
    abs_x_minus_y = numpy.sqrt(x_minus_y.dot(x_minus_y))
    abs_x_minus_y_squared = math.pow(abs_x_minus_y, 2)
    final_exp = (abs_x_minus_y_squared/(2 * math.pow(smooth, 2))) * - 1
    return math.pow(math.e, final_exp)

#print("linear_kernel test", linear_kernel((2, 3), (1, 1)))
#print("poly_kernel test", poly_kernel((2, 3), (1, 1), 2))
#print("rbf_kernel test", radial_basis_function_kernel((2, 3), (1, 1), 1))


#################ASSIGMENT 2###########################################
#t_vector = (-1, -1, 1, 1, 1)
#x_vector = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
#N = 5

classA = numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5], numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.2+[0.0, -0.5]

x_vector = numpy.concatenate((classA, classB))
t_vector = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N=x_vector.shape[0] # Number of rows (samples) permute=list(range(N)) random. shuffle(permute) inputs = inputs [permute , :] targets = targets [permute]


matrix = numpy.zeros(shape=(N, N))

def calculate_matrix(x_vector, t_vector):
    for i in range(0, len(x_vector)):
        for j in range(0, len(x_vector)):
            matrix[i][j] = t_vector[i] * t_vector[j] * linear_kernel(x_vector[i], x_vector[j])

calculate_matrix(x_vector, t_vector)
start = numpy.zeros(N)
print("printing matrix")
print(matrix)

def objective(alpha):
    vector_sum = numpy.sum(alpha)
    scalar = 0
    for i in range(0, len(alpha)):
        for j in range(0, len(alpha)):
            scalar = scalar + (alpha[i] * alpha[j] * matrix[i][j])
    #scalar = numpy.sum(numpy.dot(alpha, matrix))
    scalar = (scalar/2) - vector_sum
    return scalar

##################ASSIGMENT 3################################

def zero_fun(vector):
    return numpy.dot(vector, t_vector)
    #return numpy.absolute(numpy.dot(vector, t_vector))


##################ASSIGMENT 4################################
XC = {'type': 'eq', 'fun': zero_fun}
B = [(0, None) for b in range(N)]
print("printing B")
print(B)
ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']
support_vector_list = []
print("printing ret")
print(ret)
print("printing alpha")
print(alpha)
if ret['success']:
    for i in range(0, len(alpha)):
        print("Examining alpha value =", alpha[i])
        if (alpha[i] >= 0.000001) or (alpha[i] <= -0.000001):
            support_vector_list.append([alpha[i], x_vector[i], t_vector[i]])
print("-----------> Support vector list:", support_vector_list)

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')
#plt.savefig('svmplot.pdf')
plt.show()

print("printing first support vector and its t value")
print(support_vector_list[0][1])
print(support_vector_list[0][2])
#calculate b
#point on margin: alpha[0]

##use all x_vector or only the 3 support vectors?
def calculate_b(x_vector, t_vector, sup_vector, sup_vector_t):
    sum = 0;
    for i in range(0, len(x_vector)):
        kern = linear_kernel(sup_vector, x_vector[i])
        a=alpha[i]*t_vector[i]
        sum+=(a*kern-sup_vector_t)

    return sum;

b=calculate_b(x_vector, t_vector, support_vector_list[0][1] ,support_vector_list[0][2])
print("b ", b)


def indicator_function(sv_list, new_point, b):
    ###sv_list[i][0] = alpha force
    ###sv_list[i][1] = x vector
    ###sv_list[i][2] = t value 1 / -1
    sum = 0;
    for i in range(0, len(sv_list)):
        kern = linear_kernel(sv_list[0][1], sv_list[i][1])
        a=sv_list[i][0]*sv_list[i][2]
        sum+=(a*kern-b)
    return sum;

result = indicator_function(support_vector_list, (-1,-1), b)
print("result ", result)