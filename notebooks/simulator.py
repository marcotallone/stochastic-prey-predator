# Simulators ofr stochastic differential equations (SDEs)


# Imports
import numpy as np
from numpy.random import Generator, PCG64
from scipy.integrate import odeint


# Base simulation class
class Simulator():

    # Constructor
    def __init__(self, x0, T, N, f, g):

        # Random generator for gaussian noise
        self.rng = Generator(PCG64())

        # Initialize variables
        self.x0 = x0  # Initial state
        self.n = len(x0) # Number of states
        self.T = T  # Total time duration
        self.N = N  # Number of discretization time steps
        self.f = f  # Drift function
        self.g = g  # Diffusion function

    # Simulation method
    def simulate(self):
        """Method for simulating a single temporal evolution of stochastic differential equations.

        Parameters
        ----------
        x0 : float
            Initial state value of the process.
        T : float
            Total time duration for the simulation.
        N : int
            Number of discretization time steps to simulate.
        f : Callable[[float, float], float]
            Drift function of the SDE, which takes the current state and time as inputs.
        g : Callable[[float, float], float]
            Diffusion function of the SDE, which takes the current state and time as inputs.
        dg : Callable[[float, float], float], only for Milstein simulator
            Derivative of the diffusion function with respect to the state, which takes the current state and time as inputs.

        Returns
        -------
        x : np.ndarray
            Array of simulated state values of the process at each time point.
        t : np.ndarray
            Array of time points at which the process is evaluated.
        """

        # If T=0, return the initial state value
        if np.isclose(self.T, 0, atol=1e-6): 
            x = np.array([self.x0])
            t = np.array([0])

        # Initialize values
        dt = self.T / self.N
        x = np.zeros((self.N + 1, self.n), dtype=float)
        x[0] = self.x0
        t = np.linspace(0, self.T, self.N + 1)

        # Pre-generate Gaussian noise samples for all time steps
        dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=(self.N, self.n))

        # Main simulation loop
        for i in range(1, self.N + 1):

            # Update population
            x[i] = self._update(x[i-1], t[i-1], dW[i-1], dt)
            
            # Check population is > 0
            x[i][0] = max(0.0, x[i][0])
            x[i][1] = max(0.0, x[i][1])

        return x, t
    
    # Method to simulate the equivalent deterministic system with odeint
    def simulate_deterministic(self):
        """Method for simulating the equivalent deterministic system using odeint.

        Returns
        -------
        x : np.ndarray
            Array of simulated state values of the process at each time point.
        t : np.ndarray
            Array of time points at which the process is evaluated.
        """

        # If T=0, return the initial state value
        if np.isclose(self.T, 0, atol=1e-6): 
            x = np.array([self.x0])
            t = np.array([0])

        # Initialize values
        # dt = self.T / self.N
        x = np.zeros((self.N + 1, self.n), dtype=float)
        x[0] = self.x0
        t = np.linspace(0, self.T, self.N + 1)

        # Main simulation loop
        for i in range(1, self.N + 1):
            x[i] = odeint(self.f, x[i-1], [t[i-1], t[i]])[1]

            # Check population is > 0
            x[i][0] = max(0.0, x[i][0])
            x[i][1] = max(0.0, x[i][1])
            
        return x, t

    # Update method (base class)
    def _update(self, x, t, dW, dt):
        """Update method to be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Simulation method with fixed input noise
    def simulate_dW(self, dW):
        """Method for simulating a single temporal evolution of stochastic differential equations.

        Parameters
        ----------
        dW : np.ndarray
            Pre-generated array of Gaussian noise samples for each time step.

        Returns
        -------
        x : np.ndarray
            Array of simulated state values of the process at each time point.
        t : np.ndarray
            Array of time points at which the process is evaluated.
        """

        # If T=0, return the initial state value
        if np.isclose(self.T, 0, atol=1e-6): 
            x = np.array([self.x0])
            t = np.array([0])

        # Initialize values
        dt = self.T / self.N
        x = np.zeros((self.N + 1, self.n), dtype=float)
        x[0] = self.x0
        t = np.linspace(0, self.T, self.N + 1)

        # Main simulation loop
        for i in range(1, self.N + 1):

            # Update population
            x[i] = self._update(x[i-1], t[i-1], dW[i-1], dt)
            
            # Check population is > 0
            x[i][0] = max(0.0, x[i][0])
            x[i][1] = max(0.0, x[i][1])

        return x, t

    # Impulses generation method
    def __generate_impulses(self, times, lambd, intensity) -> None:
        """Generate the array of stochastic impulses over time.

        Raises
        ------
        ValueError
            If the initial time value is negative.
        """

        # Generate event times
        t_event: np.ndarray = np.cumsum(
            np.random.exponential(1/lambd, size = int(2 * self.T * lambd))
        )

        # Filter only the events that occur before the end of the simulation
        t_event = t_event[t_event < self.T]

        # Generate the stochastic impulse values uniformly sampled in [-intensity, +intensity]
        events: np.ndarray = np.random.uniform(-intensity, +intensity, size = (len(t_event), self.n))

        # Initialize the impulse array
        impulses = np.zeros((len(times), self.n), dtype=float)
        
        # Find idexes corresponding to impulsive events
        events_idx: np.ndarray = np.searchsorted(times, t_event)

        # Assign the impulse values to the corresponding indexes
        impulses[events_idx] = events
        
        return impulses

    # White-shot noise simulation method
    def simulate_shot_noise(self, lambd, intensity):

        # If T=0, return the initial state value
        if np.isclose(self.T, 0, atol=1e-6): 
            x = np.array([self.x0])
            t = np.array([0])

        # Initialize values
        dt = self.T / self.N
        x = np.zeros((self.N + 1, self.n), dtype=float)
        x[0] = self.x0
        t = np.linspace(0, self.T, self.N + 1)

        # Generate impulsive white shot noise
        self.impulses = self.__generate_impulses(t, lambd, intensity)

        # Main simulation loop
        for i in range(1, self.N + 1):

            # Update population
            x[i] = self._update(x[i-1], t[i-1], self.impulses[i-1], dt)
            
            # Check population is > 0
            x[i][0] = max(0.0, x[i][0])
            x[i][1] = max(0.0, x[i][1])

        return x, t

    # # Run multiple simulations for different initial conditions
    # def multi_simulate_IC(
    
    # # Simulation method
    # def simulate_linearized(self, x_eq):
    #     """Method for simulating a single temporal evolution of the linearized system.
        
    #     Prameters
    #     ---------
    #     x_eq : np.array
    #         Equilibrium point arount which the system has been linearized.
        
    #     Returns
    #     -------
    #     x : np.ndarray
    #         Array of simulated state values of the process at each time point.
    #     t : np.ndarray
    #         Array of time points at which the process is evaluated.
    #     """

    #     # If T=0, return the initial state value
    #     if np.isclose(self.T, 0, atol=1e-6): 
    #         x = np.array([self.x0])
    #         t = np.array([0])

    #     # Initialize values
    #     dt = self.T / self.N
    #     x = np.zeros((self.N + 1, self.n), dtype=float)
    #     x[0] = self.x0
    #     t = np.linspace(0, self.T, self.N + 1)

    #     # Pre-generate Gaussian noise samples for all time steps
    #     dW = self.rng.normal(loc=0.0, scale=np.sqrt(dt), size=(self.N, self.n))

    #     # Main simulation loop
    #     for i in range(1, self.N + 1):

    #         # Update population
    #         x[i] = x_eq + self._update(x[i-1], t[i-1], dW[i-1], dt)
            
    #         # Check population is > 0
    #         x[i][0] = max(0.0, x[i][0])
    #         x[i][1] = max(0.0, x[i][1])

    #     return x, t

    # TODO: method to simulate N runs and return interesting statistics...
    # def multisimulate(self, N_runs, ...):
    #     ...


# Euler-Maruyama simulator
class EulerMaruyama(Simulator):
    """Euler-Maruyama simulator for stochastic differential equations."""
    
    # Update method
    def _update(self, x, t, dW, dt):
        """Update the state using the Euler-Maruyama method.
        
        Parameters
        ----------
        x : float
            Current state value of the process.
        t : float
            Current time value.
        dW : float
            Sampled value from the normal distribution for the current time step.
        dt : float
            Time increment for the current step.
            
        Returns
        -------
        float
            Updated state value after one time step.
        """

        return x + self.f(x, t) * dt + self.g(x, t) * dW
    
    
# Milstein simulator
class Milstein(Simulator):
    """Milstein simulator for stochastic differential equations."""
    
    # Constructor
    def __init__(self, x0, T, N, f, g, dg):
        super().__init__(x0, T, N, f, g)
        self.dg = dg  # Derivative of the diffusion function
        
    # Update method
    def _update(self, x, t, dW, dt):
        """Update the state using the Milstein method.
        
        Parameters
        ----------
        x : float
            Current state value of the process.
        t : float
            Current time value.
        dW : float
            Sampled value from the normal distribution for the current time step.
        dt : float
            Time increment for the current step.
            
        Returns
        -------
        float
            Updated state value after one time step.
        """
        return (
            x + 
            self.f(x, t) * dt + 
            self.g(x, t) * dW + 
            0.5 * self.g(x, t) * self.dg(x, t) * (dW**2 - dt)
        )
    

# Stochastic Runge-Kutta simulator
class RungeKutta(Simulator):
    """Stochastic Runge-Kutta simulator for stochastic differential equations."""
    
    # Update method
    def _update(self, x, t, dW, dt):
        """Update the state using the Stochastic Runge-Kutta method.
        Source: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE)
        
        Parameters
        ----------
        x : float
            Current state value of the process.
        t : float
            Current time value.
        dW : float
            Sampled value from the normal distribution for the current time step.
        dt : float
            Time increment for the current step.
            
        Returns
        -------
        float
            Updated state value after one time step.
        """

        x_hat = x + self.f(x, t) * dt + self.g(x, t) * np.sqrt(dt)
        return (
            x + 
            self.f(x, t) * dt + 
            self.g(x, t) * dW + 
            0.5 * (self.g(x_hat, t) - self.g(x, t)) * (dW**2 - dt) / np.sqrt(dt)
        )