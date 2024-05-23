import fenics as fe
import dolfin as df
from fenics import MixedElement, FunctionSpace, TestFunctions, Function, split
from fenics import derivative, NonlinearVariationalProblem, NonlinearVariationalSolver
from fenics import UserExpression, sqrt, plot , LogLevel
from dolfin import MPI, MeshFunction, cells, refine, set_log_level, LogLevel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sympy as sp




#################### Define Parallel Variables ####################

# Get the global communicator
comm = MPI.comm_world 

# Get the rank of the process
rank = MPI.rank(comm)

# Get the size of the communicator (total number of processes)
size = MPI.size(comm)

#############################  END  ################################



set_log_level(LogLevel.ERROR)

def refine_mesh_local( mesh , rad , center , Max_level  ): 

    xc , yc = center

    mesh_itr = mesh

    for i in range(Max_level):

        mf = MeshFunction("bool", mesh_itr, mesh_itr.topology().dim() , False )

        cells_mesh = cells( mesh_itr )


        index = 0 

        for cell in cells_mesh :

            if ( cell.midpoint()[0] - xc ) **2  + ( cell.midpoint()[1] - yc ) **2  <   rad**2 : 





                mf.array()[ index ] = True

            index = index + 1 


        mesh_r = refine( mesh_itr, mf )

        # Update for next loop
        mesh_itr = mesh_r


    return mesh_itr 

#################### Define Domain ####################
Max_level = 1
dy = 0.4 

Nx= 20 
Ny = 20 

nx = (int)(Nx / dy ) + 1
ny = (int)(Ny / dy ) + 1




# print( "NY is : " + str(Ny ) + " NX is  : " + str( Nx ) )


center = [ Nx/2, Ny/2  ]  # Circle center

rad_init = 2  # Circle radius

mesh = fe.RectangleMesh(df.Point(0, 0), df.Point(Nx, Ny), nx, ny)

# mesh = refine_mesh_local( coarse_mesh , rad_init, center , Max_level  ) # Locally refined mesh

# mesh_2 = fe.RectangleMesh(df.Point(0, 0), df.Point(Nx, Ny), nx, ny)


#############################  END  #############################
 


#################### Define Constants ####################

D = 4  # Thermal diffusivity of steel, mm2.s-1

T_hot = 2000 

T_cold = 0 

dt = 1/(4*D) * dy**2





#############################  END  #############################


#################### Define Variables  ####################


def variable_define(mesh):
    
    """
    Define variables and function spaces for a given mesh.

    Parameters:
    mesh: The computational mesh

    Returns:
    Tuple containing various elements, function spaces, and test functions.
    """

    # Define finite element
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U or the Concentration

    # Create a mixed finite element
    element = MixedElement([P1, P1])

    # Define function space
    ME = FunctionSpace(mesh, element)

    # Define test function
    v_test , w_test= TestFunctions(ME)

    # Define current and previous solutions
    Sol_Func = Function(ME)  # Current solution
    Sol_Func_0 = Function(ME)  # Previous Solution

    # Split functions to get individual components
    U_answer , Phi = split(Sol_Func)  # current solution
    U_prev , Phi_0 = split(Sol_Func_0)  # last step solution



    # Extract subspaces and mappings for each subspace
    num_subs = ME.num_sub_spaces()
    spaces, maps = [], []
    for i in range(num_subs):
        space_i, map_i = ME.sub(i).collapse(collapsed_dofs=True)
        spaces.append(space_i)
        maps.append(map_i)

    # Return all the defined variables
    return U_answer, U_prev, Phi, Phi_0,  Sol_Func, Sol_Func_0, v_test, w_test, spaces, ME


def eq_diff(u_answer, u_prev, v_test, dt, D):

    """
    Calculate the differential equation term.

    Parameters:
    u_answer: The current solution
    u_prev: The solution from the previous time step
    v_test: Test function
    dt: Time step size
    D: Diffusion coefficient

    Returns:
    The finite element form of the differential equation.
    """

    # Calculate the time derivative term
    time_derivative = (u_answer - u_prev) / dt

    # Calculate the spatial derivative term
    spatial_derivative = fe.grad(u_answer)

    # Assemble the equation
    eq = ( fe.inner(time_derivative, v_test ) + 
          
          D * fe.inner(spatial_derivative, fe.grad(v_test ) ) ) * fe.dx
    

    return eq


def eq_Phi(u_answer, u_prev, v_test, dt, D):


    # Calculate the time derivative term
    time_derivative = (u_answer - u_prev) / dt


    eq = fe.inner(time_derivative, v_test ) * fe.dx
          
 
    

    return eq
#############################  END  #############################

#################### Define Problem and Solver  ####################


def problem_define(eq1, eq2,  sol_func):

    """
    Define a nonlinear variational problem and its solver.

    Parameters:
    eq1: The left-hand side of the equation (weak form)
    sol_func: The solution function

    Returns:
    A configured nonlinear variational solver.
    """

    # Define the variational problem
    L = eq1 + eq2
    J = derivative(L, sol_func)  # Compute the Jacobian of L
    problem = NonlinearVariationalProblem(L, sol_func, J=J)

    # Configure the solver for the problem
    solver = NonlinearVariationalSolver(problem)

    # Access and set solver parameters
    prm = solver.parameters
    prm["newton_solver"]["relative_tolerance"] = 1e-5
    prm["newton_solver"]["absolute_tolerance"] = 1e-6
    prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True

    return solver


#############################  END  #############################


#################### Define Initial Condition  ####################


class InitialConditions(UserExpression):
    def __init__(self, rad, center, T_hot, T_cold, **kwargs):
        super().__init__(**kwargs)
        self.rad = rad  # Initial circle radius
        self.center = center  # Center of the circle (xc, yc)
        self.T_hot = T_hot  # Temperature inside the circle
        self.T_cold = T_cold  # Temperature outside the circle

    def eval(self, values, x):
        xc, yc = self.center
        x, y = x[0], x[1]  # Coordinates
        dist = (x - xc)**2 + (y - yc)**2  # Distance squared from the center

        # Check if the point is inside the circle (dist <= rad^2)
        if dist <= self.rad**2:
            values[0] = self.T_hot  # Inside the circle
        else:
            values[0] = self.T_cold  # Outside the circle

        values[1] = 0  # If 'Phi' is the second variable and it's always 0

    def value_shape(self):
        return (2,)

    
def Initial_Interpolate( sol_func, Sol_Func_0, rad, center, T_hot, T_cold,  degree ):


    initial_conditions = InitialConditions(rad= rad, center = center, T_hot= T_hot,T_cold= T_cold ,  degree= degree)

    sol_func.interpolate(initial_conditions)

    Sol_Func_0.interpolate(initial_conditions)



#############################  END  ###############################


#################### Define Step 1 For Solving  ####################
    
U_answer, U_prev, Phi, Phi_0,  Sol_Func, Sol_Func_0, v_test, w_test, spaces, ME = variable_define(mesh= mesh )

eqdiff = eq_diff(u_answer= U_answer, u_prev= U_prev, v_test= v_test, dt= dt , D= D )

eqPhi = eq_Phi(u_answer= Phi , u_prev= Phi_0, v_test= w_test , dt = dt, D= D)

Initial_Interpolate(Sol_Func, Sol_Func_0 , rad_init, center, T_hot, T_cold, 2 )

solver = problem_define(eq1= eqdiff, eq2= eqPhi, sol_func= Sol_Func)

#############################  END  ###############################

############################ File Section #########################


file = fe.XDMFFile("Diffusion_AD.xdmf" ) # File Name To Save #


def write_simulation_data(Sol_Func, time, file, variable_names ):


    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()



T = 0

variable_names = [ "U", "Phi" ]  # Adjust as needed


write_simulation_data( Sol_Func, T, file , variable_names=variable_names )


#############################  END  ###############################



############################ Solever Loop #########################

############ Initialize for Adaptive Mesh #########################


V_project = spaces[0]

for it in tqdm(range(0, 2)):

    T = T + dt

    solver.solve()

    Sol_Func_0.vector()[:] = Sol_Func.vector()  # update the solution


    if it % 1 == 0 :

        write_simulation_data( Sol_Func_0,  T , file , variable_names )



#############################  END  ###############################



spaces_past = spaces
(U_answer, U_prev, Phi, Phi_0,  Phi_U, Phi_U_0, v_test, w_test, spaces, ME )= variable_define( mesh )

u_new, phi_new = Phi_U_0.split(deepcopy=True)
u_past, phi_past = Sol_Func_0.split(deepcopy=True)




def Value_Coor_dof(U_answer, v_phi, comm):
    """Return value of the solution at the degrees of freedom and corresponding coordinates."""
    coordinates_of_all = v_phi.tabulate_dof_coordinates()
    phi_value_on_dof = U_answer.vector().get_local()

    all_Val_dof = comm.gather(phi_value_on_dof, root=0)
    all_point = comm.gather(coordinates_of_all, root=0)

    # Broadcast the data to all processors
    all_point = comm.bcast(all_point, root=0)
    all_Val_dof = comm.bcast(all_Val_dof, root=0)

    # Combine the data from all processors
    all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
    all_point_1 = [point for sublist in all_point for point in sublist]

    point = np.array(all_point_1)
    Val_dof = np.array(all_Val_dof_1)

    # Generate the interpolating function
    interp_func = linear_lagrange_interpolation(point, Val_dof)

    return interp_func
from scipy.spatial import Delaunay


def linear_lagrange_interpolation(points, values):
    """
    Perform linear interpolation on a 2D grid using linear Lagrange elements.

    Parameters:
    - points: Array of shape (n, 2) with the coordinates of the interpolation nodes
    - values: Array of shape (n,) with the function values at the interpolation nodes

    Returns:
    - interp_func: A function that takes (x, y) and returns the interpolated value
    """
    points = np.asarray(points)
    values = np.asarray(values)
    
    # Create a Delaunay triangulation of the points
    tri = Delaunay(points)
    
    def interp_func(x, y):
        # Find the simplex containing the point
        simplex = tri.find_simplex((x, y))
        
        if simplex == -1:
            return np.nan  # Point is outside the convex hull
        
        # Get the vertices of the simplex
        vertices = tri.simplices[simplex]
        pts = points[vertices]
        vals = values[vertices]
        
        # Barycentric coordinates
        T = np.vstack((pts.T, np.ones(len(pts))))
        bary_coords = np.linalg.solve(T, [x, y, 1])
        
        # Interpolated value
        return np.dot(bary_coords, vals)
    
    return interp_func

interp_func = Value_Coor_dof(u_past, spaces_past[0], comm)



b = 5 # x + b
xmin=0.0
xmax= Nx
lx = xmax - xmin
# coordinates = mesh.coordinates()
# u_new, phi_new = Phi_U_0.split(deepcopy=True)
# u_past, phi_past = Sol_Func_0.split(deepcopy=True)

u_new_vector = u_new.vector().get_local()
phi_new_vector = phi_new.vector().get_local()
coordinates = spaces[0].tabulate_dof_coordinates()


for i, coord in enumerate(coordinates):

    x0_shift = coord[0] - 5
    x_shift = [x0_shift, coord[1]]

    # try : 
    #     u_eval = interp_func(x_shift[0], x_shift[1]) 
    # except: 
    #     u_eval = 0.0

    u_eval = interp_func(x_shift[0], x_shift[1]) 

    if x0_shift < 5:
        u_eval = 0.0
    else:
        u_new_vector[i] = u_eval




u_new.vector().set_local(u_new_vector)
u_new.vector().apply('insert')
phi_new.vector().set_local(phi_new_vector)
phi_new.vector().apply('insert')
fe.assign(Phi_U_0, [u_new, phi_new])
write_simulation_data( Phi_U_0,  T+1 , file , variable_names )





# # print( U_prev_past_mesh(fe.Point(0.2, 0.5)) ) 

# # for i, coord in enumerate(coordinates):

# #     # print(coord)

# #     try:

# #         Phi_0_vector[i] = Phi_0_past_mesh(df.Point(coord))
# #         U_prev_vector[i] = U_prev_past_mesh(df.Point(coord - np.array([10, 10])) ) 
# #         # print(coord)
# #         # print(i)

# #     except:

            
# #         Phi_0_vector[i] = 0
# #         U_prev_vector[i] = 0
        



# # Phi_0.vector().set_local(Phi_0_vector)
# # Phi_0.vector().apply('insert')
# # U_prev.vector().set_local(U_prev_vector)
# # U_prev.vector().apply('insert')

# # fe.assign(Phi_U_0, [U_prev, Phi_0])




# # fe.LagrangeInterpolator.interpolate(Phi_U, Sol_Func) # Phi_U is new and moved
# # fe.LagrangeInterpolator.interpolate(Phi_U_0, Sol_Func_0)


# # write_simulation_data( Phi_U_0,  T+1 , file , variable_names )



# # def move_the_fields( spaces, past_solu, new_solu , y_disp  ):

# #     coordinates = spaces[0].tabulate_dof_coordinates()

# #     u_past_mesh, phi_past_mesh  = past_solu.split( deepcopy=True ) # past mesh solution
# #     u_new , phi_new = new_solu.split(deepcopy=True)  # last step solution
# #     u_new_vector = phi_new.vector().get_local()
# #     phi_new_vector = u_new.vector().get_local() 

# #     for i, coord in enumerate(coordinates):
# #         try:
# #             phi_new_vector[i] = phi_past_mesh(df.Point(coord + np.array([0.0, y_disp]) ) )
# #             u_new_vector[i] = u_past_mesh(df.Point(coord + np.array([0.0, y_disp])) ) 
# #         except:
# #             phi_new_vector[i] = 0
# #             u_new_vector[i] = 0

# #     phi_new.vector().set_local(phi_new_vector)
# #     phi_new.vector().apply('insert')
# #     u_new.vector().set_local(u_new_vector)
# #     u_new.vector().apply('insert')

# #     fe.assign(new_solu, [u_new, phi_new])

# #     return new_solu



            



