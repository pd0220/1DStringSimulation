// 1D elastic string simulation (solving the wave equation numerically)

// including useful headers and/or libraries
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
// Eigen library for matrices
#include <Eigen/Dense>

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// number of time (s) and space (cm) points on the lattice
const int numOfTimeSteps = 300;
const int numOfSpaceSteps = 20;
// speed of wave [sqrt(tension / linear density)] (cm / s)
const double cWave = 10.;
// starting and ending (x = L) points in space (cm)
const double xStart = 0., xStop = 100.;
// starting and ending points in time (s)
const double tStart = 0., tStop = 50.;
// step size for time (s) and space (cm)
const double deltaT = (tStop - tStart) / numOfTimeSteps;
const double deltaX = (xStop - xStart) / numOfSpaceSteps;
// characteristic constant for stepping
const double StepperConst = cWave * cWave * deltaT * deltaT / deltaX / deltaX;
// initial conditions --> x0 and y0 (cm)
const double X0 = 50.;
const double Y0 = 3.;

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// setting initial values (lambda)
auto SetInitialValue = [&](Eigen::MatrixXd &mat) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }
    // check space coordinate of initial value
    if (X0 <= xStart || X0 >= xStop)
    {
        std::cout << "ERROR\nGive space coordinate is not appropriate." << std::endl;
        std::exit(-1);
    }

    // y(x, t = 0) = given value
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
    {
        if (xIndex * deltaX <= X0)
        {
            mat(0, xIndex) = Y0 * xIndex * deltaX / X0;
        }
        else
        {
            mat(0, xIndex) = Y0 * (xStop - xIndex * deltaX) / (xStop - X0);
        }
    }

    // y(x = 0, t) = y(x = L, t) = 0
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        mat(tIndex, 0) = 0.;
        mat(tIndex, numOfSpaceSteps - 1);
    }
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// taking first step (lambda)
auto FirstStep = [](Eigen::MatrixXd &mat) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // initial velocity will be dy(x, t = 0) / dt = f(x) = 0 --> taking first step accordingly
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        mat(1, xIndex) = mat(0, xIndex) * (1 - StepperConst) + StepperConst / 2 * (mat(0, xIndex + 1) + mat(0, xIndex - 1));
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// taking a single step (from tIndex to tIndex + 1)
auto Step = [](Eigen::MatrixXd &mat, int tIndex) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // taking one timestep
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        mat(tIndex + 1, xIndex) = 2 * mat(tIndex, xIndex) * (1 - StepperConst) - mat(tIndex - 1, xIndex) + StepperConst * (mat(tIndex, xIndex + 1) + mat(tIndex, xIndex - 1));
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// main function
int main(int, char **)
{
    // check for stability
    if (StepperConst >= 1)
    {
        std::cout << "WARNING\nStability issues may occur." << std::endl;
    }

    // matrix for y(x, t) values
    Eigen::MatrixXd matSolution(numOfTimeSteps, numOfSpaceSteps);

    // setting initial values for y(x = 0, t) = y(x = L, t) = 0 and y(x, t = 0)
    SetInitialValue(matSolution);

    // taking the first step seperately --> estimating y(x, t - dt) from initial condition: dy(x, t = 0) / dt = f(x)
    FirstStep(matSolution);

    // taking timesteps (rest of the simulation)
    for (int tIndex{1}; tIndex < numOfTimeSteps - 1; tIndex++)
    {
        Step(matSolution, tIndex);
        //std::cout << matSolution(tIndex, 3) << std::endl;
    }

    std::ofstream data;
    data.open("animationData.txt");
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        for (int xIndex{0}; xIndex < numOfSpaceSteps; xIndex++)
        {
            data << matSolution(tIndex, xIndex) << " ";
        }
        data << "\n";
    }
    data.close();

    data.open("data.txt");
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        for (int xIndex{0}; xIndex < numOfSpaceSteps; xIndex++)
        {
            data << tIndex * deltaT << " " << xIndex * deltaX << " " << matSolution(tIndex, xIndex) << "\n";
        }
    }
    data.close();
}
